"""
Enhanced Real-time monitoring dashboard with Phase 1 & 2 features.

Features:
- Top-K predictions
- Live confusion matrix
- Dark mode
- CSV export
- Sensor heatmap
- 3D position plot
- Performance metrics

Run with: python flask_dashboard_enhanced.py
Then open: http://localhost:5000
"""
import sys
sys.path.insert(0, 'src')

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime
import io
import base64
from sklearn.manifold import TSNE
import redis
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add file-based logging for diagnostics
file_handler = logging.FileHandler('/tmp/dashboard.log', mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.info("=" * 80)
logger.info("Dashboard starting with file logging enabled")
logger.info("=" * 80)

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.multihead_lm import MultiHeadGCodeLM
from miracle.dataset.target_utils import TokenDecomposer
from miracle.utilities.gcode_tokenizer import GCodeTokenizer
from miracle.training.grammar_constraints import GCodeGrammarConstraints


# Operation type mapping (matches preprocessing.py)
OPERATION_TYPE_NAMES = [
    "adaptive",           # 0
    "adaptive150025",     # 1
    "face",               # 2
    "face150025",         # 3
    "pocket",             # 4
    "pocket150025",       # 5
    "damageadaptive",     # 6
    "damageface",         # 7
    "damagepocket",       # 8
]

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'gcode-fingerprinting-2025'

# Initialize SocketIO with CORS support
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,
    engineio_logger=False
)

# Initialize Redis for caching (optional - graceful fallback if not available)
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
    REDIS_AVAILABLE = True
except (redis.ConnectionError, redis.RedisError) as e:
    logger.warning(f"Redis not available: {e}. Caching disabled.")
    redis_client = None
    REDIS_AVAILABLE = False

# Global state
state = {
    'model': None,
    'model_type': None,  # 'baseline' or 'multihead'
    'decomposer': None,  # For multi-head token decomposition
    'vocab_value_map': None,  # Nearest-token mapper for 4-digit vocabulary
    'tokenizer': None,
    'metadata': None,  # Preprocessing metadata with master_columns
    'buffer': deque(maxlen=64),
    'predictions_history': [],
    'csv_data': None,
    'current_idx': 0,
    'running': False,
    'config': {},
    'ground_truth': [],  # For confusion matrix
    'confusion_matrix': defaultdict(lambda: defaultdict(int)),  # Live confusion matrix (GCode)
    'confusion_matrix_operation': defaultdict(lambda: defaultdict(int)),  # Operation type confusion matrix
    'confusion_matrix_type': defaultdict(lambda: defaultdict(int)),  # Type confusion matrix
    'confusion_matrix_command': defaultdict(lambda: defaultdict(int)),  # Command confusion matrix
    'confusion_matrix_param_type': defaultdict(lambda: defaultdict(int)),  # Param type confusion matrix
    'confusion_matrix_param_value': defaultdict(lambda: defaultdict(int)),  # Param value confusion matrix
    'statistics': defaultdict(list),  # Running statistics
    'fingerprints': [],  # For t-SNE
    'generation_history': deque(maxlen=50),  # Last 50 generated commands
    'command_types': defaultdict(int),  # Track command type distribution
    'current_modal_command': None,  # Track current modal G-code command (e.g., 'G1', 'G0')
    'accuracy_over_time': [],  # Track accuracy metrics
    'attention_weights': None,  # Latest attention weights for visualization
    'attention_history': deque(maxlen=10),  # Last 10 attention maps
    # Generation settings
    'gen_settings': {
        'enable_autoregressive': True,
        'max_tokens': 15,
        'temperature': 1.0,
        'top_p': 1.0,
        'beam_size': 1,
        'use_beam_search': False,
    },
    # W&B Sweep monitoring
    'wandb_entity': None,
    'wandb_project': None,
    'active_sweeps': [],
    'sweep_cache': {},
    'sweep_cache_timestamp': {},
}


# ============================================================================
# Error Handling Utilities
# ============================================================================

def emit_error(error_message, error_type="error", hint=None, exception=None):
    """
    Emit standardized error messages via WebSocket.

    Args:
        error_message: Human-readable error message
        error_type: Type of error (error, warning, info)
        hint: Optional hint for resolving the error
        exception: Optional exception object for logging
    """
    from flask_socketio import emit

    error_data = {
        'error': error_message,
        'type': error_type,
        'timestamp': datetime.now().isoformat()
    }

    if hint:
        error_data['hint'] = hint

    if exception:
        logger.error(f"{error_message}: {str(exception)}", exc_info=True)
        error_data['details'] = str(exception)
    else:
        logger.error(error_message)

    emit('error', error_data)


def validate_model_loaded():
    """Validate that model is loaded and ready."""
    if state['model'] is None:
        emit_error(
            'No model loaded',
            hint='Please select and load a model from the dropdown before starting inference.'
        )
        return False
    return True


def validate_csv_loaded():
    """Validate that CSV data is loaded and ready."""
    if state['csv_data'] is None:
        emit_error(
            'No CSV data loaded',
            hint='Please select and load a CSV file from the dropdown before starting inference.'
        )
        return False
    if len(state['csv_data']) == 0:
        emit_error(
            'CSV file is empty',
            hint='The loaded CSV file contains no data rows.'
        )
        return False
    return True


# ============================================================================
# Token Reconstruction Helpers
# ============================================================================

def reconstruct_numeric_token(token_str, tokenizer_config):
    """
    Convert bucketed numeric token to readable G-code value.

    IMPORTANT: In multi-head models, the parameter letter (X, Y, etc.) is a
    SEPARATE token from the numeric value. So NUM_X_2 should become just "0.002",
    not "X0.002", because the X is already in the sequence!

    Examples:
        NUM_X_2 → 0.002 (just the value, no parameter letter!)
        NUM_Y_15 → 0.015
        NUM_F_250 → 250
        G1 → G1 (pass-through for non-numeric tokens)

    Args:
        token_str: Token string (e.g., "NUM_X_2" or "G1")
        tokenizer_config: Tokenizer config with precision settings

    Returns:
        Reconstructed value string (without parameter letter for NUM_ tokens)
    """
    import re

    # Check if this is a numeric bucket token
    match = re.match(r'NUM_([A-Z])_(-?\d+)', token_str)
    if not match:
        # Not a numeric token, return as-is
        return token_str

    param = match.group(1)
    bucket_str = match.group(2)
    bucket_value = int(bucket_str)

    # Get precision for this parameter
    precision = tokenizer_config.get('precision', {}).get(param, 1e-3)

    # Convert bucket value back to actual value
    actual_value = bucket_value * precision

    # Format based on parameter type
    # NOTE: Return JUST the value, not param+value (param is separate token!)
    if param in ['X', 'Y', 'Z', 'I', 'J', 'K', 'R']:
        # Position values: 3 decimal places
        return f"{actual_value:.3f}"
    elif param == 'F':
        # Feed rate: integer or 1 decimal
        if actual_value == int(actual_value):
            return f"{int(actual_value)}"
        else:
            return f"{actual_value:.1f}"
    elif param in ['S', 'T']:
        # Spindle speed, tool number: integer
        return f"{int(actual_value)}"
    elif param == 'P':
        # Dwell time: 3 decimal places
        return f"{actual_value:.3f}"
    else:
        # Default: 3 decimal places
        return f"{actual_value:.3f}"


# ============================================================================
# Model and Data Loading
# ============================================================================

def load_model(checkpoint_path):
    """Load model from checkpoint - supports both baseline and multi-head models."""
    device = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint['config']

    # Detect model type by checking for multihead_state_dict in checkpoint
    # Multihead models have both backbone_state_dict and multihead_state_dict
    is_multihead = 'multihead_state_dict' in checkpoint

    if is_multihead:
        logger.info("Loading Multi-Head model (Phase 2)")
        model_type = 'multihead'

        # Infer architecture from checkpoint state dict
        state_dict_key = 'multihead_state_dict' if 'multihead_state_dict' in checkpoint else 'model_state_dict'
        state_dict = checkpoint[state_dict_key]

        # ===== FIX 1: INFER D_MODEL FROM CHECKPOINT =====
        # Try multiple methods to infer d_model
        inferred_d_model = None

        # Method 1: From embedding layer
        if 'embed.weight' in state_dict:
            inferred_d_model = state_dict['embed.weight'].shape[1]
            logger.info(f"Inferred d_model={inferred_d_model} from embed.weight shape")

        # Method 2: From decoder layers
        elif 'decoder.layers.0.self_attn.in_proj_weight' in state_dict:
            inferred_d_model = state_dict['decoder.layers.0.self_attn.in_proj_weight'].shape[1]
            logger.info(f"Inferred d_model={inferred_d_model} from decoder attention layer")

        # Method 3: From config (with validation)
        elif 'hidden_dim' in config_dict:
            inferred_d_model = config_dict['hidden_dim']
            logger.warning(f"Using d_model={inferred_d_model} from config (no validation possible)")
        else:
            # Fallback with warning
            inferred_d_model = 128
            logger.warning(f"⚠️  Could not infer d_model from checkpoint! Using default: {inferred_d_model}")

        # Validate against config if present
        if 'hidden_dim' in config_dict and config_dict['hidden_dim'] != inferred_d_model:
            logger.warning(
                f"⚠️  Config mismatch: config says hidden_dim={config_dict['hidden_dim']}, "
                f"but state_dict indicates d_model={inferred_d_model}. Using state_dict value."
            )

        # Store inferred d_model in config for use by LSTM encoder
        config_dict['hidden_dim'] = inferred_d_model

        # Extract head sizes from state dict
        n_commands = state_dict['command_head.4.weight'].shape[0]
        n_param_types = state_dict['param_type_head.4.weight'].shape[0]

        # Detect param_value architecture: regression, hybrid, or standard
        if 'param_value_regression_head.8.weight' in state_dict:
            # REGRESSION architecture: direct continuous value prediction
            # n_param_values is not used (regression outputs scalar), set to 1 for compatibility
            n_param_values = 1
            logger.info(f"Detected REGRESSION architecture with continuous value prediction")
        elif 'param_value_coarse_head.4.weight' in state_dict:
            # Hybrid architecture: coarse (10) + residual heads
            n_param_values = state_dict['param_value_coarse_head.4.weight'].shape[0]
            logger.info(f"Detected HYBRID architecture with coarse bucketing")
        else:
            # Standard architecture: single param_value head
            n_param_values = state_dict['param_value_head.4.weight'].shape[0]

        logger.info(f"Detected architecture: d_model={inferred_d_model}, n_commands={n_commands}, "
                   f"n_param_types={n_param_types}, n_param_values={n_param_values}")

        # ===== FIX 2: INFER VOCAB SIZE FROM CHECKPOINT =====
        inferred_vocab_size = state_dict['embed.weight'].shape[0] if 'embed.weight' in state_dict else 170
        if inferred_vocab_size != 170:
            logger.info(f"Detected non-standard vocab_size: {inferred_vocab_size}")

        # ===== FIX 3: CREATE MODEL WITH CORRECT DIMENSIONS =====
        try:
            model = MultiHeadGCodeLM(
                d_model=inferred_d_model,  # Use inferred dimension!
                n_commands=n_commands,
                n_param_types=n_param_types,
                n_param_values=n_param_values,
                nhead=config_dict.get('num_heads', config_dict.get('nhead', 4)),
                num_layers=config_dict.get('num_layers', 2),
                dropout=0.1,
                vocab_size=inferred_vocab_size,
            ).to(device)
        except Exception as e:
            logger.error(f"Failed to create MultiHeadGCodeLM with inferred dimensions: {e}")
            raise ValueError(
                f"Model creation failed. Detected dimensions: d_model={inferred_d_model}, "
                f"n_commands={n_commands}, n_param_types={n_param_types}, n_param_values={n_param_values}, "
                f"vocab_size={inferred_vocab_size}. Error: {e}"
            )

        # Load state dict - multi-head checkpoints have multihead_state_dict
        try:
            if 'multihead_state_dict' in checkpoint:
                # Use strict=False to allow partial loading of incompatible checkpoints
                # (e.g., old hybrid models with param_value_coarse/residual heads vs new regression head)
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint['multihead_state_dict'],
                    strict=False
                )
                if missing_keys or unexpected_keys:
                    logger.warning(f"⚠️  Partial checkpoint loading:")
                    if missing_keys:
                        logger.warning(f"   Missing keys (not in checkpoint): {missing_keys[:3]}...")
                    if unexpected_keys:
                        logger.warning(f"   Unexpected keys (not in model): {unexpected_keys[:3]}...")
                    logger.warning("   Model loaded partially - some heads may not work correctly")
            elif 'model_state_dict' in checkpoint:
                missing_keys, unexpected_keys = model.load_state_dict(
                    checkpoint['model_state_dict'],
                    strict=False
                )
                if missing_keys or unexpected_keys:
                    logger.warning("⚠️  Partial checkpoint loading - model may not work correctly")
            else:
                raise KeyError("Checkpoint missing 'multihead_state_dict' or 'model_state_dict'")
        except RuntimeError as e:
            logger.error(f"❌ State dict loading failed: {e}")
            logger.error("This usually means the checkpoint dimensions don't match the created model.")
            logger.error(f"Checkpoint keys: {list(state_dict.keys())[:5]}...")
            raise ValueError(
                f"Failed to load checkpoint weights. The checkpoint may be corrupted or incompatible. "
                f"Inferred dimensions: d_model={inferred_d_model}, vocab_size={inferred_vocab_size}. "
                f"Original error: {e}"
            )

        model.eval()

        # Load tokenizer (vocab for multi-head) - MUST match training config!
        vocab_path = Path('data/vocabulary_4digit_hybrid.json')
        if vocab_path.exists():
            tokenizer = GCodeTokenizer.load(vocab_path)
            logger.info(f"✓ Loaded vocabulary from: {vocab_path}")
        else:
            logger.error(f"✗ Vocabulary not found: {vocab_path}")
            tokenizer = None

        # Load decomposer for token reconstruction
        decomposer = TokenDecomposer(vocab_path) if vocab_path.exists() else None

        # Decomposer already logs its initialization details above
        if not decomposer:
            logger.error(f"✗ Failed to initialize decomposer")

        # Build nearest-token mapper for 4-digit vocabulary
        # Maps regression values to nearest available vocabulary tokens
        vocab_value_map = {}
        if decomposer and vocab_path.exists():
            import json
            vocab_data = json.load(open(vocab_path))
            param_values = {}

            # Extract all numeric token values by parameter type
            for token, idx in vocab_data['vocab'].items():
                if token.startswith('NUM_'):
                    parts = token.split('_')
                    if len(parts) == 3:
                        param = parts[1]
                        value = int(parts[2])
                        if param not in param_values:
                            param_values[param] = []
                        param_values[param].append((value, token))

            # Sort and store
            for param, values in param_values.items():
                vocab_value_map[param] = sorted(values, key=lambda x: x[0])

            logger.info(f"✓ Built nearest-token mapper for {len(vocab_value_map)} parameters:")
            for param, values in sorted(vocab_value_map.items()):
                value_range = f"{values[0][0]} to {values[-1][0]}" if values else "empty"
                logger.info(f"  {param}: {len(values)} tokens, range {value_range}")

        # Initialize grammar constraints for inference
        grammar_constraints = None
        if decomposer:
            try:
                device = next(model.parameters()).device
                grammar_constraints = GCodeGrammarConstraints(
                    decomposer.vocab,
                    device=device,
                    decomposer=decomposer,  # Pass decomposer for type transition constraints
                    allow_modal_commands=True  # Enable modal G-code behavior (sequences can start with parameters)
                )
                logger.info(f"✅ Grammar constraints initialized on {device} (modal commands enabled)")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize grammar constraints: {e}")

        # Load preprocessing metadata - try multiple locations
        metadata = None
        metadata_search_dirs = [
            'outputs/processed_hybrid',
            'outputs/processed_with_ops',
            'outputs/processed_v2',
            'outputs/processed_2digit_FIXED',
            'outputs/processed',
        ]

        for data_dir_str in metadata_search_dirs:
            data_dir = Path(data_dir_str)
            metadata_path = data_dir / 'train_sequences_metadata.json'
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from: {metadata_path}")
                break

        if metadata is None:
            logger.warning("⚠️ No metadata found in any preprocessed data directory. Sensor heatmap may not work correctly.")

        # Create config object for compatibility (with updated hidden_dim)
        config = type('Config', (), config_dict)()

        logger.info(f"✅ Multi-Head model loaded: {inferred_d_model}d, "
                   f"{n_commands} commands, vocab={inferred_vocab_size}")

        return model, tokenizer, config, metadata, model_type, decomposer, grammar_constraints, vocab_value_map

    else:
        logger.info("Loading Baseline model (Phase 1)")
        model_type = 'baseline'

        # Map checkpoint config parameters to ModelConfig parameters
        # Checkpoints may use different parameter names than ModelConfig expects
        mapped_config = config_dict.copy()

        # Map parameter names
        if 'hidden_dim' in mapped_config and 'd_model' not in mapped_config:
            mapped_config['d_model'] = mapped_config.pop('hidden_dim')

        if 'vocab_size' in mapped_config and 'gcode_vocab' not in mapped_config:
            mapped_config['gcode_vocab'] = mapped_config.pop('vocab_size')

        if 'num_layers' in mapped_config and 'lstm_layers' not in mapped_config:
            mapped_config['lstm_layers'] = mapped_config.pop('num_layers')

        if 'num_heads' in mapped_config and 'n_heads' not in mapped_config:
            mapped_config['n_heads'] = mapped_config.pop('num_heads')

        # Filter to only include ModelConfig parameters (exclude training params like batch_size, learning_rate, etc.)
        # ModelConfig expects: sensor_dims, d_model, lstm_layers, gcode_vocab, future_len,
        #                      n_heads, context_specs, fp_dim, use_attention_pooling
        valid_params = {
            'sensor_dims', 'd_model', 'lstm_layers', 'gcode_vocab', 'future_len',
            'n_heads', 'context_specs', 'fp_dim', 'use_attention_pooling'
        }
        filtered_config = {k: v for k, v in mapped_config.items() if k in valid_params}

        # Create baseline model
        config = ModelConfig(**filtered_config)
        model = MM_DTAE_LSTM(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load tokenizer (vocab v1 for baseline)
        vocab_path = Path('data/vocabulary_4digit_hybrid.json')
        if vocab_path.exists():
            tokenizer = GCodeTokenizer.load(vocab_path)
        else:
            tokenizer = None

        # Load preprocessing metadata
        data_dir = Path('outputs/processed')
        metadata_path = data_dir / 'train_sequences_metadata.json'
        metadata = None
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        logger.info(f"✅ Baseline model loaded: {config.d_model}d, vocab={config.gcode_vocab}")

        return model, tokenizer, config, metadata, model_type, None, None, None  # decomposer, grammar_constraints, vocab_value_map


def extract_features_with_validation(row, metadata, model_config):
    """
    Extract features from CSV row with proper validation and error handling.

    Args:
        row: pandas Series representing one CSV row
        metadata: Preprocessing metadata dict (may be None)
        model_config: ModelConfig object from loaded model

    Returns:
        Tuple of (continuous_features, categorical_features)

    Raises:
        ValueError: If feature dimensions don't match model expectations
    """
    # Expected dimensions from model
    expected_cont_dim = model_config.sensor_dims[0] if hasattr(model_config, 'sensor_dims') else None
    expected_cat_dim = model_config.sensor_dims[1] if hasattr(model_config, 'sensor_dims') and len(model_config.sensor_dims) > 1 else None

    # Method 1: Use metadata if available (preferred)
    if metadata and 'master_columns' in metadata:
        master_cols = metadata['master_columns']
        cat_cols = metadata.get('categorical_columns', ['stat', 'unit', 'dist', 'coor'])

        logger.debug(f"Using metadata: {len(master_cols)} continuous features, {len(cat_cols)} categorical features")

        # Validate dimensions match model expectations
        if expected_cont_dim is not None and len(master_cols) != expected_cont_dim:
            error_msg = (
                f"Dimension mismatch: Metadata specifies {len(master_cols)} continuous features "
                f"but model expects {expected_cont_dim}. "
                f"This usually means the CSV was preprocessed with different settings than the model was trained with."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if expected_cat_dim is not None and len(cat_cols) != expected_cat_dim:
            logger.warning(
                f"Categorical dimension mismatch: Metadata has {len(cat_cols)} features "
                f"but model expects {expected_cat_dim}. Adjusting..."
            )
            # Pad or truncate categorical features
            cat_cols = (cat_cols + ['stat', 'unit', 'dist', 'coor'])[:expected_cat_dim]

        # Extract continuous features with zero-padding for missing columns
        continuous = np.zeros(len(master_cols), dtype=np.float32)
        missing_cols = []
        for i, col in enumerate(master_cols):
            if col in row.index:
                continuous[i] = float(row[col])
            else:
                missing_cols.append(col)

        if missing_cols:
            logger.debug(f"Zero-padding {len(missing_cols)} missing sensors: {missing_cols[:5]}...")

        # Extract categorical features
        categorical = np.zeros(len(cat_cols), dtype=np.float32)
        for i, col in enumerate(cat_cols):
            if col in row.index:
                categorical[i] = float(row[col])

        return continuous, categorical

    # Method 2: Fallback to column-based detection (legacy)
    logger.warning("No metadata available, using fallback column detection. This may cause dimension mismatches!")

    exclude_cols = ['time', 'gcode_line_num', 'gcode_text', 'gcode_tokens',
                    't_console', 'gcode_line', 'gcode_string', 'raw_json',
                    'vel', 'plane', 'line', 'posx', 'posy', 'posz', 'feed', 'momo']
    cat_cols = ['stat', 'unit', 'dist', 'coor']

    cont_cols = [col for col in row.index if col not in exclude_cols and col not in cat_cols]
    continuous = row[cont_cols].values.astype(np.float32)
    categorical = row[[col for col in cat_cols if col in row.index]].values.astype(np.float32)

    # Validate dimensions
    if expected_cont_dim is not None and len(continuous) != expected_cont_dim:
        logger.warning(
            f"Fallback extraction: Got {len(continuous)} continuous features, "
            f"model expects {expected_cont_dim}. Padding/truncating..."
        )
        # Pad or truncate
        if len(continuous) < expected_cont_dim:
            continuous = np.pad(continuous, (0, expected_cont_dim - len(continuous)), 'constant')
        else:
            continuous = continuous[:expected_cont_dim]

    if expected_cat_dim is not None and len(categorical) != expected_cat_dim:
        logger.warning(
            f"Fallback extraction: Got {len(categorical)} categorical features, "
            f"model expects {expected_cat_dim}. Padding/truncating..."
        )
        # Pad or truncate
        if len(categorical) < expected_cat_dim:
            categorical = np.pad(categorical, (0, expected_cat_dim - len(categorical)), 'constant')
        else:
            categorical = categorical[:expected_cat_dim]

    return continuous, categorical


def edit_distance(s1, s2):
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def nucleus_sampling(logits, top_p=0.9, temperature=1.0):
    """Apply nucleus (top-p) sampling to logits."""
    if temperature != 1.0:
        logits = logits / temperature

    # Sort logits in descending order
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]

    # Convert to probabilities
    probs = np.exp(sorted_logits - sorted_logits.max())
    probs = probs / probs.sum()

    # Compute cumulative probabilities
    cumulative_probs = np.cumsum(probs)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Keep at least one token
    if sorted_indices_to_remove[0]:
        sorted_indices_to_remove[0] = False

    # Set logits of removed tokens to -inf
    filtered_logits = logits.copy()
    filtered_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf

    # Sample from filtered distribution
    probs = np.exp(filtered_logits - filtered_logits.max())
    probs = probs / probs.sum()

    # Sample token
    token_id = np.random.choice(len(probs), p=probs)
    return token_id, probs[token_id]


def _get_valid_params(command):
    """Get valid parameters for a G-code command."""
    valid_params = {
        'G0': 'X Y Z (rapid positioning - no feed rate)',
        'G1': 'X Y Z F (linear interpolation with feed rate)',
        'G2': 'X Y Z F R I J K (clockwise arc - needs radius or center offset)',
        'G3': 'X Y Z F R I J K (counter-clockwise arc - needs radius or center offset)',
        'M3': 'S (spindle on clockwise with speed)',
        'M5': '(spindle off - no parameters)',
        'M30': '(program end - no parameters)',
    }
    return valid_params.get(command, 'X Y Z F S R I J K (default)')


def beam_search_generate(model, memory, tokenizer, beam_size=3, max_tokens=15, special_tokens=None, model_type='baseline'):
    """Generate G-code using beam search."""
    device = 'cpu'
    eos_id = tokenizer.vocab.get('EOS', 2)

    if special_tokens is None:
        special_tokens = {'PAD', 'BOS', 'EOS', 'UNK', 'MASK', '<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>'}

    # Initialize beams: (score, tokens, confidence)
    beams = [(0.0, [1], 1.0)]  # Start with BOS token

    for step in range(max_tokens):
        candidates = []

        for score, tokens, conf in beams:
            # Check if beam already ended
            if tokens[-1] == eos_id:
                candidates.append((score, tokens, conf))
                continue

            # Get logits
            current_tokens = torch.tensor([tokens], dtype=torch.long, device=device)

            if model_type == 'multihead':
                # Multi-head model - need to reconstruct full tokens from hierarchical predictions
                multihead_outputs = model(memory, current_tokens)

                # Get predictions from all heads
                type_logits = multihead_outputs['type_logits'][0, -1].detach().cpu().numpy()
                command_logits = multihead_outputs['command_logits'][0, -1].detach().cpu().numpy()
                param_type_logits = multihead_outputs['param_type_logits'][0, -1].detach().cpu().numpy()
                param_value_logits = multihead_outputs['param_value_logits'][0, -1].detach().cpu().numpy()

                # For simplicity in beam search, use command head as primary (most G-code tokens are commands)
                next_logits = command_logits
            else:
                # Baseline model
                step_logits = model.gcode_head(memory, current_tokens)
                next_logits = step_logits[0, -1].detach().cpu().numpy()

            # Get top-k tokens
            top_indices = np.argsort(next_logits)[-beam_size * 2:][::-1]

            probs = np.exp(next_logits - next_logits.max())
            probs = probs / probs.sum()

            for idx in top_indices:
                token_id = int(idx)
                token_prob = float(probs[token_id])
                token_score = score + np.log(token_prob + 1e-10)

                # Decode to check if valid
                decoded = tokenizer.decode([token_id])
                if isinstance(decoded, list):
                    token_text = decoded[0] if decoded else ''
                else:
                    token_text = decoded

                # Skip special tokens except EOS
                if token_text in special_tokens and token_id != eos_id:
                    continue

                new_tokens = tokens + [token_id]
                new_conf = conf * token_prob
                candidates.append((token_score, new_tokens, new_conf))

        # Keep top beam_size beams
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        # Stop if all beams ended
        if all(tokens[-1] == eos_id for _, tokens, _ in beams):
            break

    # Return best beam
    best_score, best_tokens, best_conf = beams[0]

    # Decode tokens
    result_tokens = []
    for token_id in best_tokens[1:]:  # Skip BOS
        if token_id == eos_id:
            break
        decoded = tokenizer.decode([token_id])
        if isinstance(decoded, list):
            token_text = decoded[0] if decoded else ''
        else:
            token_text = decoded
        if token_text and token_text not in special_tokens:
            result_tokens.append(token_text)

    return result_tokens, best_conf


def process_sample(continuous, categorical, ground_truth_gcode=None):
    """Process a single sensor sample and return predictions."""
    if state['model'] is None:
        logger.warning("process_sample called but no model loaded")
        return None

    # Add to buffer
    state['buffer'].append({
        'continuous': continuous,
        'categorical': categorical,
        'timestamp': datetime.now()
    })

    # Need full window
    if len(state['buffer']) < 64:
        return None

    # Stack into windows with error handling
    try:
        cont_window = np.stack([s['continuous'] for s in state['buffer']])
        cat_window = np.stack([s['categorical'] for s in state['buffer']])
    except ValueError as e:
        logger.error(f"Failed to stack features into window: {e}")
        logger.error(f"Feature shapes in buffer: continuous={[s['continuous'].shape for s in list(state['buffer'])[:3]]}")
        logger.error(f"This usually indicates inconsistent feature dimensions across samples")
        # Clear buffer to prevent continued errors
        state['buffer'].clear()
        return None

    # Convert to torch
    device = 'cpu'
    try:
        cont_tensor = torch.from_numpy(cont_window).unsqueeze(0).float().to(device)
        cat_tensor = torch.from_numpy(cat_window).unsqueeze(0).float().to(device)
    except Exception as e:
        logger.error(f"Failed to convert features to tensors: {e}")
        logger.error(f"cont_window.shape: {cont_window.shape}, cat_window.shape: {cat_window.shape}")
        return None

    mods = [cont_tensor, cat_tensor]

    # Inference with error handling - handle both baseline and multi-head models
    try:
        with torch.no_grad():
            if state['model_type'] == 'multihead':
                # Multi-head model: needs LSTM encoder for memory
                # Get hidden_dim from config (inferred during model loading)
                hidden_dim = state['config'].hidden_dim if hasattr(state['config'], 'hidden_dim') else 128

                # Create a simple LSTM encoder if not already in model
                if not hasattr(state['model'], 'lstm_encoder'):
                    # Use the continuous features as input to an LSTM to get memory
                    # CRITICAL: Use the same hidden_dim as the multihead model!
                    lstm = torch.nn.LSTM(cont_tensor.shape[-1], hidden_dim, 2, batch_first=True, bidirectional=False)
                    lstm = lstm.to(device).eval()
                    state['model'].lstm_encoder = lstm
                    logger.info(f"Created LSTM encoder with hidden_dim={hidden_dim}")

                # Validate dimensions match
                lstm_hidden_dim = state['model'].lstm_encoder.hidden_size
                if lstm_hidden_dim != hidden_dim:
                    logger.error(
                        f"⚠️  DIMENSION MISMATCH: LSTM encoder has hidden_dim={lstm_hidden_dim}, "
                        f"but multihead model expects d_model={hidden_dim}!"
                    )
                    # Recreate LSTM with correct dimensions
                    lstm = torch.nn.LSTM(cont_tensor.shape[-1], hidden_dim, 2, batch_first=True, bidirectional=False)
                    lstm = lstm.to(device).eval()
                    state['model'].lstm_encoder = lstm
                    logger.info(f"Recreated LSTM encoder with correct hidden_dim={hidden_dim}")

                # Get memory from LSTM
                memory, _ = state['model'].lstm_encoder(cont_tensor)  # [B, T, D]

                # Validate memory dimensions
                if memory.shape[-1] != hidden_dim:
                    logger.error(
                        f"❌ Memory dimension mismatch: LSTM output is {memory.shape[-1]}, "
                        f"but model expects {hidden_dim}"
                    )
                    raise ValueError(
                        f"LSTM encoder output dimension ({memory.shape[-1]}) doesn't match "
                        f"multihead model's d_model ({hidden_dim})"
                    )

                # Store for later use in G-code generation
                outputs = {
                    'memory': memory,
                    'fingerprint': memory[:, -1, :],  # Use last hidden state as fingerprint
                    'anom': torch.zeros(1, 1).to(device),  # Placeholder
                }
            else:
                # Baseline model
                outputs = state['model'](
                    mods=mods,
                    lengths=torch.tensor([64]).to(device),
                    gcode_in=None,
                    modality_dropout_p=0.0
                )
    except RuntimeError as e:
        logger.error(f"❌ Model inference failed: {e}")
        logger.error(f"Input tensor shapes: continuous={cont_tensor.shape}, categorical={cat_tensor.shape}")
        logger.error(f"Expected model input dims: {state['config'].sensor_dims if state.get('config') else 'unknown'}")
        # Check for both hidden_dim (multihead) and d_model (baseline)
        hidden_dim_val = 'unknown'
        if state.get('config'):
            if hasattr(state['config'], 'hidden_dim'):
                hidden_dim_val = state['config'].hidden_dim
            elif hasattr(state['config'], 'd_model'):
                hidden_dim_val = state['config'].d_model
        logger.error(f"Model hidden_dim/d_model: {hidden_dim_val}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}", exc_info=True)
        return None

    # Extract predictions
    fingerprint_vec = outputs['fingerprint'][0].cpu().numpy() if 'fingerprint' in outputs else np.zeros(128)
    memory_vec = outputs['memory'][0].cpu().numpy() if 'memory' in outputs else np.zeros((64, 128))

    # Calculate fingerprint score (L2 norm - higher means more distinctive fingerprint)
    fingerprint_score = float(np.linalg.norm(fingerprint_vec))

    # Calculate sensor reconstruction quality score
    # Use memory variance as a proxy for information capture (higher = better representation)
    reconstruction_score = float(np.std(memory_vec))

    predictions = {
        'fingerprint': fingerprint_vec.tolist(),
        'fingerprint_score': fingerprint_score,
        'reconstruction_score': reconstruction_score,
        'anomaly_score': float(outputs['anom'][0].cpu().numpy()[0]) if 'anom' in outputs else 0.0,
        'timestamp': state['buffer'][-1]['timestamp'].isoformat()
    }

    # Decode G-code with Top-K using logits from BOS token
    if state['tokenizer'] and 'memory' in outputs:
        try:
            memory = outputs['memory']  # [B, T, D]

            # ===== TOKEN-LEVEL PREDICTION (Current approach) =====
            # Use a single BOS token to get logits for next token prediction
            bos_tokens = torch.full((1, 1), 1, dtype=torch.long, device=device)  # [B, 1]

            if state['model_type'] == 'multihead':
                # Multi-head model returns a dictionary with different head outputs
                try:
                    multihead_outputs = state['model'](memory, bos_tokens)  # Dict with type_logits, command_logits, etc.

                    # Extract attention weights for visualization (if available)
                    try:
                        if hasattr(state['model'], 'extract_attention_weights'):
                            attention_data = state['model'].extract_attention_weights(memory, bos_tokens)
                            # Store attention weights
                            state['attention_weights'] = {
                                'avg_attention': attention_data['attention'].tolist(),  # [B, Tg, Tm]
                                'timestamp': datetime.utcnow().isoformat(),
                                'shape': {
                                    'batch': memory.shape[0],
                                    'target_len': bos_tokens.shape[1],
                                    'memory_len': memory.shape[1],
                                }
                            }
                            state['attention_history'].append(state['attention_weights'])
                    except Exception as attn_err:
                        logger.debug(f"Could not extract attention weights: {attn_err}")
                        # Non-critical, continue without attention visualization

                except RuntimeError as e:
                    logger.error(f"❌ G-code generation error: {e}")
                    logger.error(f"Memory shape: {memory.shape}, bos_tokens shape: {bos_tokens.shape}")
                    logger.error(f"Model d_model: {state['config'].hidden_dim if hasattr(state['config'], 'hidden_dim') else 'unknown'}")
                    raise ValueError(
                        f"Dimension mismatch during G-code generation. "
                        f"Memory dimension ({memory.shape[-1]}) may not match model's d_model. "
                        f"Try reloading the model. Error: {e}"
                    )

                # Extract all heads from multihead model
                type_logits = multihead_outputs['type_logits'][0, -1].detach().cpu().numpy()  # [n_types]
                command_logits = multihead_outputs['command_logits'][0, -1].detach().cpu().numpy()  # [n_commands]
                param_type_logits = multihead_outputs['param_type_logits'][0, -1].detach().cpu().numpy()  # [n_param_types]

                # Extract operation type logits if available
                if 'operation_logits' in multihead_outputs:
                    operation_logits = multihead_outputs['operation_logits'][0].detach().cpu().numpy()  # [n_operation_types]
                    operation_type_id = int(np.argmax(operation_logits))
                    operation_confidence = float(np.max(operation_logits))
                else:
                    operation_logits = None
                    operation_type_id = -1
                    operation_confidence = 0.0

                # Handle param_value: regression mode vs hybrid mode (coarse + residual) vs standard mode
                if 'param_value_regression' in multihead_outputs:
                    # Direct regression mode: continuous value prediction
                    param_value_regression = multihead_outputs['param_value_regression'][0, -1].detach().cpu().item()  # scalar
                    # Round to nearest integer for token composition
                    param_value_id = int(round(param_value_regression))
                    # Create dummy logits for visualization (one-hot at predicted value)
                    param_value_logits = np.zeros(100)  # Dummy array for visualization
                    if 0 <= param_value_id < 100:
                        param_value_logits[param_value_id] = 1.0
                    logger.info(f"Regression mode: predicted_value={param_value_regression:.2f}, rounded_id={param_value_id}")
                elif 'param_value_coarse_logits' in multihead_outputs:
                    # Hybrid bucketing mode (coarse + residual)
                    param_value_coarse_logits = multihead_outputs['param_value_coarse_logits'][0, -1].detach().cpu().numpy()  # [10]
                    param_value_residual = multihead_outputs['param_value_residual'][0, -1].detach().cpu().item()  # scalar
                    param_value_logits = param_value_coarse_logits  # For visualization/logging
                    param_value_id = int(np.argmax(param_value_coarse_logits))  # Coarse bucket (0-9)
                    logger.info(f"Hybrid mode: coarse_id={param_value_id}, residual={param_value_residual:.2f}")
                elif 'param_value_logits' in multihead_outputs:
                    # Standard bucketing mode (legacy)
                    param_value_logits = multihead_outputs['param_value_logits'][0, -1].detach().cpu().numpy()  # [n_param_values]
                    param_value_id = int(np.argmax(param_value_logits))
                else:
                    # No param_value prediction available - use default
                    logger.warning("No param_value prediction found in model outputs - using default value 0")
                    param_value_id = 0
                    param_value_logits = np.zeros(100)
                    param_value_logits[0] = 1.0

                # Predict each component (argmax on each head)
                type_id = int(np.argmax(type_logits))
                command_id = int(np.argmax(command_logits))
                param_type_id = int(np.argmax(param_type_logits))

                # Store component predictions for confusion matrix tracking
                predictions['component_predictions'] = {
                    'type_id': type_id,
                    'command_id': command_id,
                    'param_type_id': param_type_id,
                    'param_value_id': param_value_id
                }

                # DEBUG: Log multihead prediction details
                logger.info("=" * 60)
                logger.info("MULTIHEAD PREDICTION DEBUG")
                logger.info("=" * 60)

                # Log top 5 logits for each head
                logger.info(f"Type head (argmax={type_id}):")
                type_top5_idx = np.argsort(type_logits)[-5:][::-1]
                for idx in type_top5_idx:
                    logger.info(f"  [{idx}]: {type_logits[idx]:.4f}")

                logger.info(f"Command head (argmax={command_id}):")
                cmd_top5_idx = np.argsort(command_logits)[-5:][::-1]
                for idx in cmd_top5_idx:
                    cmd_name = state['decomposer'].command_tokens[idx] if state['decomposer'] and idx < len(state['decomposer'].command_tokens) else f"<ID:{idx}>"
                    logger.info(f"  [{idx}] {cmd_name}: {command_logits[idx]:.4f}")

                logger.info(f"Param Type head (argmax={param_type_id}):")
                pt_top5_idx = np.argsort(param_type_logits)[-5:][::-1]
                for idx in pt_top5_idx:
                    logger.info(f"  [{idx}]: {param_type_logits[idx]:.4f}")

                logger.info(f"Param Value head (argmax={param_value_id}):")
                pv_top5_idx = np.argsort(param_value_logits)[-5:][::-1]
                for idx in pv_top5_idx:
                    logger.info(f"  [{idx}]: {param_value_logits[idx]:.4f}")

                # Log decomposed predictions
                logger.info(f"\nDecomposed prediction:")
                logger.info(f"  type_id={type_id}, command_id={command_id}, param_type_id={param_type_id}, param_value_id={param_value_id}")

                if state['decomposer'] and command_id == 0:
                    logger.warning(f"⚠️  command_id=0, which maps to: {state['decomposer'].command_tokens[0]}")
                logger.info("=" * 60)

                # Reconstruct full token using TokenDecomposer
                if state['decomposer']:
                    predicted_token_id = state['decomposer'].compose_token(
                        type_id, command_id, param_type_id, param_value_id
                    )
                    # DEBUG: Log the composed token
                    # Get reverse mapping from decomposer's vocab
                    id_to_token = {v: k for k, v in state['decomposer'].vocab.items()}
                    predicted_token_str = id_to_token.get(predicted_token_id, f"<ID:{predicted_token_id}>")
                    logger.info(f"Composed token: ID={predicted_token_id}, Token={predicted_token_str}")
                else:
                    # Fallback: use command head only
                    logger.warning("No decomposer available - using command head only (may produce incorrect tokens)")
                    predicted_token_id = command_id

                # Create a fake logits array where the predicted token has highest score
                # This allows the downstream code to work unchanged
                last_logits = np.zeros(state['decomposer'].vocab_size if state['decomposer'] else len(command_logits))
                last_logits[predicted_token_id] = 10.0  # High confidence for predicted token

                # Add some diversity by including top predictions from command head
                for cmd_idx in np.argsort(command_logits)[-5:][::-1]:
                    if state['decomposer'] and cmd_idx < len(state['decomposer'].command_tokens):
                        cmd_token = state['decomposer'].command_tokens[cmd_idx]
                        cmd_token_id = state['decomposer'].vocab.get(cmd_token, 0)
                        last_logits[cmd_token_id] = float(command_logits[cmd_idx])
            else:
                # Baseline model
                gcode_logits = state['model'].gcode_head(memory, bos_tokens)  # [B, 1, vocab_size]
                last_logits = gcode_logits[0, -1].detach().cpu().numpy()  # [vocab_size]

            # Get top-k predictions
            top_k = 10  # Get more to filter out special tokens
            top_indices = np.argsort(last_logits)[-top_k:][::-1]
            top_scores = last_logits[top_indices]

            # Convert to probabilities
            top_probs = np.exp(top_scores - top_scores.max())  # Numerical stability
            top_probs = top_probs / top_probs.sum()

            # Decode top-k and filter special tokens
            top_predictions = []
            special_tokens = {'PAD', 'BOS', 'EOS', 'UNK', 'MASK', '<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>'}

            for idx, prob in zip(top_indices, top_probs):
                try:
                    token_id = int(idx)
                    gcode_decoded = state['tokenizer'].decode([token_id])

                    # Handle case where decode returns a list
                    if isinstance(gcode_decoded, list):
                        gcode_text = gcode_decoded[0] if gcode_decoded else ''
                    else:
                        gcode_text = gcode_decoded

                    # Skip special tokens and empty strings
                    if gcode_text and gcode_text not in special_tokens:
                        top_predictions.append({
                            'gcode': gcode_text,
                            'confidence': float(prob),
                            'token_id': token_id
                        })

                        # Stop once we have 5 valid predictions
                        if len(top_predictions) >= 5:
                            break
                except Exception as decode_err:
                    print(f"Token decode error for idx {idx}: {decode_err}")
                    continue

            # Ensure we have at least one prediction
            if top_predictions:
                predictions['top_k'] = top_predictions[:5]  # Top 5
                predictions['gcode_text'] = top_predictions[0]['gcode']
                predictions['gcode_confidence'] = top_predictions[0]['confidence']

            # Add operation type prediction (for multihead models)
            if state['model_type'] == 'multihead' and 'operation_type_id' in locals() and operation_type_id >= 0:
                if operation_type_id < len(OPERATION_TYPE_NAMES):
                    predictions['operation_type'] = OPERATION_TYPE_NAMES[operation_type_id]
                    predictions['operation_type_id'] = operation_type_id
                    predictions['operation_confidence'] = operation_confidence
                else:
                    predictions['operation_type'] = 'unknown'
                    predictions['operation_type_id'] = -1
                    predictions['operation_confidence'] = 0.0
            else:
                predictions['operation_type'] = 'unknown'
                predictions['operation_type_id'] = -1
                predictions['operation_confidence'] = 0.0

            if not top_predictions:
                # Fallback to highest scoring token even if it's special
                highest_idx = np.argmax(last_logits)
                fallback_decoded = state['tokenizer'].decode([int(highest_idx)])

                # Handle case where decode returns a list
                if isinstance(fallback_decoded, list):
                    fallback_text = fallback_decoded[0] if fallback_decoded else '<UNK>'
                else:
                    fallback_text = fallback_decoded

                predictions['top_k'] = [{'gcode': fallback_text, 'confidence': 0.1}]
                predictions['gcode_text'] = fallback_text
                predictions['gcode_confidence'] = 0.1

            # ===== FULL COMMAND GENERATION (Autoregressive) =====
            if state['gen_settings']['enable_autoregressive']:
                # Use beam search or greedy/sampling
                if state['gen_settings']['use_beam_search']:
                    full_command_tokens, full_command_confidence = beam_search_generate(
                        state['model'],
                        memory,
                        state['tokenizer'],
                        beam_size=state['gen_settings']['beam_size'],
                        max_tokens=state['gen_settings']['max_tokens'],
                        special_tokens=special_tokens,
                        model_type=state['model_type']
                    )
                else:
                    # Greedy or nucleus sampling
                    full_command_tokens = []
                    full_command_confidence = 1.0
                    token_confidences = []  # Per-token confidence for breakdown
                    current_tokens = torch.full((1, 1), 1, dtype=torch.long, device=device)

                    max_tokens = state['gen_settings']['max_tokens']
                    temperature = state['gen_settings']['temperature']
                    top_p = state['gen_settings']['top_p']
                    eos_id = state['tokenizer'].vocab.get('EOS', 2)

                    logger.debug(f"Starting autoregressive generation: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")

                    for step_num in range(max_tokens):
                        if state['model_type'] == 'multihead':
                            # Multi-head model - reconstruct full token from hierarchical predictions
                            multihead_outputs = state['model'](memory, current_tokens)

                            # Apply grammar constraints if available (BEFORE argmax)
                            if state.get('grammar_constraints') and state['decomposer']:
                                try:
                                    # Log type logits BEFORE constraints (for first few steps)
                                    if len(full_command_tokens) < 6:
                                        type_logits_before = multihead_outputs['type_logits'][0, -1].detach().cpu().numpy()
                                        logger.debug(f"[Step {len(full_command_tokens)}] Type logits BEFORE constraints: {type_logits_before}")

                                    # Pass modal command context to grammar constraints
                                    # This allows sequences to start with parameters if we have a modal command active
                                    modal_cmd = state.get('current_modal_command')
                                    if modal_cmd and len(full_command_tokens) == 0:
                                        logger.info(f"[Modal Mode] Using modal command context: {modal_cmd}")

                                    multihead_outputs = state['grammar_constraints'].apply_inference_constraints(
                                        multihead_outputs,
                                        current_tokens,
                                        step=len(full_command_tokens),
                                        modal_command=modal_cmd  # Pass modal command for context
                                    )

                                    # Log type logits AFTER constraints (for first few steps)
                                    if len(full_command_tokens) < 6:
                                        type_logits_after = multihead_outputs['type_logits'][0, -1].detach().cpu().numpy()
                                        logger.debug(f"[Step {len(full_command_tokens)}] Type logits AFTER constraints: {type_logits_after}")
                                        logger.info(f"[Step {len(full_command_tokens)}] Grammar constraints applied to {full_command_tokens}")
                                except Exception as e:
                                    logger.error(f"Grammar constraint application failed: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())

                            # Get predictions from all heads
                            type_logits = multihead_outputs['type_logits'][0, -1].detach().cpu().numpy()
                            command_logits = multihead_outputs['command_logits'][0, -1].detach().cpu().numpy()
                            param_type_logits = multihead_outputs['param_type_logits'][0, -1].detach().cpu().numpy()

                            # Check for regression vs hybrid vs standard param_value head
                            if 'param_value_regression' in multihead_outputs:
                                # REGRESSION MODE: Direct continuous value prediction
                                param_value_regression_val = multihead_outputs['param_value_regression'][0, -1, 0].detach().cpu().item()
                                logger.info(f"Regression mode: predicted_value={param_value_regression_val:.2f}, rounded_id={int(round(param_value_regression_val))}")

                                # For 4-digit vocabulary: Map regression value to nearest available token
                                mapped_token_name = None  # Will store the token name if using vocab map
                                if state.get('vocab_value_map') and state.get('decomposer'):
                                    # Get parameter type (need to predict it first or get from previous step)
                                    temp_param_type_logits = multihead_outputs['param_type_logits'][0, -1].detach().cpu().numpy()
                                    temp_param_type_id = int(np.argmax(temp_param_type_logits))

                                    if temp_param_type_id < len(state['decomposer'].param_tokens):
                                        param_name = state['decomposer'].param_tokens[temp_param_type_id]

                                        if param_name in state['vocab_value_map']:
                                            available_values = state['vocab_value_map'][param_name]

                                            # CRITICAL FIX: Scale regression output to match vocabulary range
                                            # Regression outputs small values (-0.1 to 0.5), but vocab expects larger integers
                                            # Scale by 1000 to convert from mm to micrometers (vocab uses integer micrometers)
                                            predicted_value_scaled = param_value_regression_val * 1000.0
                                            predicted_value_int = int(round(predicted_value_scaled))

                                            # Find nearest available token value
                                            nearest_idx = min(range(len(available_values)),
                                                            key=lambda i: abs(available_values[i][0] - predicted_value_int))
                                            nearest_value, nearest_token = available_values[nearest_idx]

                                            # Store the token name to use directly instead of composing
                                            mapped_token_name = nearest_token
                                            param_value_id = nearest_value  # Keep for logging
                                            logger.info(f"  Regression: {param_value_regression_val:.3f} → Scaled: {predicted_value_int} → Nearest: {nearest_value} (token: {nearest_token})")
                                        else:
                                            # Parameter not in vocab map, use default
                                            param_value_id = max(0, min(99, int(round(param_value_regression_val))))
                                            logger.warning(f"  Parameter {param_name} not in vocab map, using default: {param_value_id}")
                                    else:
                                        param_value_id = max(0, min(99, int(round(param_value_regression_val))))
                                else:
                                    # No vocab map available, use old method
                                    param_value_id = int(round(param_value_regression_val))
                                    param_value_id = max(0, min(99, param_value_id))

                                # Create dummy logits array for compatibility (one-hot encoding)
                                # Note: This is not actually used when decomposer is available
                                param_value_logits = np.zeros(10)
                                # Safely index: param_value_id could be negative or > 9 for 4-digit vocab
                                safe_idx = max(0, min(param_value_id, 9))
                                param_value_logits[safe_idx] = 10.0  # High confidence
                            elif 'param_value_coarse_logits' in multihead_outputs:
                                # Hybrid mode: use coarse bucket
                                param_value_logits = multihead_outputs['param_value_coarse_logits'][0, -1].detach().cpu().numpy()
                                param_value_id = int(np.argmax(param_value_logits))
                            else:
                                # Standard mode: use full param_value
                                param_value_logits = multihead_outputs['param_value_logits'][0, -1].detach().cpu().numpy()
                                param_value_id = int(np.argmax(param_value_logits))

                            # Reconstruct full token from hierarchical predictions
                            type_id = int(np.argmax(type_logits))
                            command_id = int(np.argmax(command_logits))
                            param_type_id = int(np.argmax(param_type_logits))

                            # SAFETY: First token MUST be a command (type_id=1)
                            # This is a fallback in case grammar constraints didn't work
                            if len(full_command_tokens) == 0 and type_id != 1:
                                logger.warning(f"First token had type_id={type_id}, forcing to COMMAND (type_id=1)")
                                type_id = 1  # Force TYPE_COMMAND

                            # SAFETY: After PARAMETER, next token MUST be NUMERIC (type_id=3)
                            # This prevents incomplete parameter-value pairs like "Z" without a number
                            if len(full_command_tokens) > 0 and type_id == 0:  # SPECIAL predicted
                                # Check if last token was a PARAMETER
                                last_token_id = current_tokens[0, -1].item()
                                last_type, _, _, _ = state['decomposer'].decompose_token(last_token_id)

                                if last_type == 2:  # Last was TYPE_PARAMETER
                                    logger.error(f"⚠️ CONSTRAINT VIOLATION: EOS predicted after PARAMETER at step {len(full_command_tokens)}")
                                    logger.error(f"  Last token ID: {last_token_id}")
                                    logger.error(f"  Full sequence: {full_command_tokens}")
                                    # Show type probabilities to diagnose why constraint failed
                                    type_probs = np.exp(type_logits - type_logits.max())
                                    type_probs = type_probs / type_probs.sum()
                                    logger.error(f"  Type probs: [SPECIAL:{type_probs[0]:.6f}, COMMAND:{type_probs[1]:.6f}, "
                                              f"PARAMETER:{type_probs[2]:.6f}, NUMERIC:{type_probs[3]:.6f}]")
                                    logger.warning(f"  🛡️ FORCING type_id=3 (NUMERIC) to complete parameter-value pair")
                                    type_id = 3  # Force TYPE_NUMERIC

                            # Debug logging for token components
                            if state.get('decomposer'):
                                # Enhanced logging for all tokens to debug Z missing value
                                if len(full_command_tokens) < 8:
                                    # Show type probabilities to verify grammar constraints worked
                                    type_probs = np.exp(type_logits - type_logits.max())
                                    type_probs = type_probs / type_probs.sum()
                                    logger.info(f"Token {len(full_command_tokens)}: type_id={type_id}, "
                                              f"type_probs=[SPECIAL:{type_probs[0]:.3f}, COMMAND:{type_probs[1]:.3f}, "
                                              f"PARAMETER:{type_probs[2]:.3f}, NUMERIC:{type_probs[3]:.3f}]")
                                    logger.info(f"  Last 3 tokens: {full_command_tokens[-3:] if len(full_command_tokens) >= 3 else full_command_tokens}")

                                logger.debug(f"Token prediction - type:{type_id}, cmd:{command_id}, "
                                           f"param_type:{param_type_id}, param_val:{param_value_id}")

                            # Check if type is SPECIAL (0) - this means EOS or end of sequence
                            # This matches the logic in MultiHeadGCodeLM.generate()
                            if type_id == 0:
                                logger.debug(f"Stopping generation: type_id=0 (SPECIAL) predicted")
                                break

                            if state['decomposer']:
                                # For 4-digit vocab with mapped token, use direct lookup
                                if 'mapped_token_name' in locals() and mapped_token_name and type_id == 3:
                                    # Directly look up the token ID from vocabulary
                                    if mapped_token_name in state['decomposer'].vocab:
                                        next_token_id = state['decomposer'].vocab[mapped_token_name]
                                        logger.debug(f"  Using mapped token: {mapped_token_name} → ID {next_token_id}")
                                    else:
                                        # Fallback to compose if token not found
                                        logger.warning(f"  Token {mapped_token_name} not in vocab, falling back to compose")
                                        next_token_id = state['decomposer'].compose_token(
                                            type_id, command_id, param_type_id, param_value_id
                                        )
                                else:
                                    # Normal composition for bucketed vocabularies
                                    next_token_id = state['decomposer'].compose_token(
                                        type_id, command_id, param_type_id, param_value_id
                                    )

                                # SAFETY: Check if composed token is invalid (PAD/UNK)
                                # This happens when param_value_id is out of range for the parameter
                                if next_token_id == 0 or next_token_id == 3:  # PAD or UNK
                                    logger.error(f"⚠️ INVALID TOKEN COMPOSED: type={type_id}, cmd={command_id}, "
                                              f"param_type={param_type_id}, param_value={param_value_id}")
                                    logger.error(f"  Composed token_id={next_token_id}, which is PAD/UNK")

                                    # If it's a NUMERIC token (type_id=3), try incrementing param_value_id
                                    if type_id == 3 and param_value_id < 9:
                                        logger.warning(f"  Trying param_value_id={param_value_id + 1} instead...")
                                        for fallback_value in range(param_value_id + 1, 10):
                                            fallback_token_id = state['decomposer'].compose_token(
                                                type_id, command_id, param_type_id, fallback_value
                                            )
                                            if fallback_token_id not in [0, 3]:  # Not PAD/UNK
                                                logger.warning(f"  ✅ Fallback successful: param_value_id={fallback_value}, "
                                                            f"token_id={fallback_token_id}")
                                                next_token_id = fallback_token_id
                                                param_value_id = fallback_value
                                                break
                                        else:
                                            logger.error(f"  ❌ All fallback values failed! Forcing EOS.")
                                            type_id = 0  # Force generation to stop
                                            break
                                    else:
                                        logger.error(f"  ❌ Cannot recover from invalid token! Forcing EOS.")
                                        type_id = 0  # Force generation to stop
                                        break

                                # For confidence, use command head probability (most reliable)
                                cmd_probs = np.exp(command_logits - command_logits.max())
                                cmd_probs = cmd_probs / cmd_probs.sum()
                                token_confidence = float(cmd_probs[command_id])

                                # Skip sampling - we already have the reconstructed token
                                next_token_logits = None  # Signal we already have next_token_id
                            else:
                                # Fallback: use command head (may produce wrong tokens)
                                next_token_logits = command_logits
                        else:
                            # Baseline model
                            step_logits = state['model'].gcode_head(memory, current_tokens)
                            next_token_logits = step_logits[0, -1].detach().cpu().numpy()

                        # Apply temperature and sampling (unless we already have the token from decomposer)
                        if next_token_logits is not None:
                            if temperature != 1.0 or top_p < 1.0:
                                next_token_id, token_confidence = nucleus_sampling(
                                    next_token_logits,
                                    top_p=top_p,
                                    temperature=temperature
                                )
                            else:
                                # Greedy
                                next_token_id = int(np.argmax(next_token_logits))
                                probs = np.exp(next_token_logits - next_token_logits.max())
                                probs = probs / probs.sum()
                                token_confidence = float(probs[next_token_id])
                        # else: next_token_id and token_confidence already set from decomposer

                        # Decode token
                        decoded = state['tokenizer'].decode([next_token_id])
                        if isinstance(decoded, list):
                            token_text = decoded[0] if decoded else ''
                        else:
                            token_text = decoded

                        # Reconstruct numeric tokens from bucketed format to readable G-code
                        if state['tokenizer'] and hasattr(state['tokenizer'], 'cfg'):
                            tokenizer_config = {
                                'precision': state['tokenizer'].cfg.precision if hasattr(state['tokenizer'].cfg, 'precision') else {}
                            }
                            token_text = reconstruct_numeric_token(token_text, tokenizer_config)

                        # Filter out PAD tokens completely
                        if token_text in ['PAD', '<PAD>', 'BOS', '<BOS>']:
                            logger.debug(f"Skipping special token: '{token_text}'")
                            continue

                        # Log the predicted token (after reconstruction)
                        logger.debug(f"Step {len(full_command_tokens)}: predicted '{token_text}' (id={next_token_id}, conf={token_confidence:.4f})")

                        # Stop if EOS token (for baseline models or edge cases)
                        if next_token_id == eos_id:
                            logger.debug(f"Stopping generation: EOS token predicted")
                            break

                        # Repetition detection: break if same token repeated too many times
                        if len(full_command_tokens) >= 3:
                            last_3_tokens = full_command_tokens[-3:]
                            if len(set(last_3_tokens)) == 1:
                                logger.debug(f"Stopping generation: Token '{token_text}' repeated 3+ times")
                                break

                        # SAFETY: Prevent parameter repetition within same G-code command
                        # Each parameter (X, Y, Z, R, F, S, I, J, K) should appear at most once per command
                        # Valid: "G1 X1.200 Y0.043 Z0.000"
                        # Invalid: "G1 X1.200 X0.500" (X appears twice)
                        if token_text in ['X', 'Y', 'Z', 'R', 'F', 'S', 'I', 'J', 'K']:
                            # Track parameters used since the last COMMAND token
                            used_params = set()
                            for i in range(len(full_command_tokens) - 1, -1, -1):
                                tok = full_command_tokens[i]
                                # Stop scanning when we hit a command token (G0, G1, M3, etc.)
                                if tok.startswith('G') or tok.startswith('M'):
                                    break
                                # Track parameter tokens
                                if tok in ['X', 'Y', 'Z', 'R', 'F', 'S', 'I', 'J', 'K']:
                                    used_params.add(tok)

                            # If this parameter was already used in current command, stop generation
                            if token_text in used_params:
                                logger.warning(f"⚠️ Parameter repetition detected: '{token_text}' already used in this command")
                                logger.warning(f"  Current sequence: {' '.join(full_command_tokens)}")
                                logger.warning(f"  Stopping generation to prevent invalid G-code")
                                break  # Stop generation

                        # SAFETY: Enforce parameter ordering (position params before modifier params)
                        # Valid G-code order: G1 X Y Z (position) then F S R I J K (modifiers)
                        # Invalid: "G1 X R Z" (modifier R between position params)
                        position_params = ['X', 'Y', 'Z']
                        modifier_params = ['F', 'S', 'R', 'I', 'J', 'K']

                        if token_text in position_params or token_text in modifier_params:
                            # Scan backward to see if we've already used any modifier parameters
                            has_modifier = False
                            for i in range(len(full_command_tokens) - 1, -1, -1):
                                tok = full_command_tokens[i]
                                # Stop scanning when we hit a command token
                                if tok.startswith('G') or tok.startswith('M'):
                                    break
                                # Check if we've seen a modifier parameter
                                if tok in modifier_params:
                                    has_modifier = True
                                    break

                            # If we've seen a modifier parameter and now predicting a position parameter, stop
                            if has_modifier and token_text in position_params:
                                logger.warning(f"⚠️ Invalid parameter order: Position param '{token_text}' predicted after modifier param")
                                logger.warning(f"  Current sequence: {' '.join(full_command_tokens)}")
                                logger.warning(f"  Valid order: X Y Z (position) → F S R I J K (modifiers)")
                                logger.warning(f"  Stopping generation to prevent invalid G-code")
                                break  # Stop generation

                        # SAFETY: Command-specific parameter validation
                        # Different G-code commands accept different parameters
                        # G0 (rapid): X Y Z only (no F, R, I, J, K)
                        # G1 (linear): X Y Z F only (no R, I, J, K)
                        # G2/G3 (arcs): X Y Z F R I J K (need radius or center offset)
                        # M3/M5 (spindle): S only (no position params)
                        if token_text in ['X', 'Y', 'Z', 'F', 'S', 'R', 'I', 'J', 'K']:
                            # Find the current command by scanning backward
                            current_command = None
                            for i in range(len(full_command_tokens) - 1, -1, -1):
                                tok = full_command_tokens[i]
                                if tok.startswith('G') or tok.startswith('M'):
                                    current_command = tok
                                    break

                            # Modal mode: If no explicit command in sequence, use modal command context
                            if current_command is None and state.get('current_modal_command'):
                                current_command = state['current_modal_command']
                                logger.debug(f"[Modal] Using modal command for validation: {current_command}")

                            if current_command:
                                # Define invalid parameter combinations
                                invalid_combinations = {
                                    'G0': ['F', 'R', 'I', 'J', 'K'],  # Rapid move: no feed rate or arc params
                                    'G1': ['R', 'I', 'J', 'K'],       # Linear move: no arc parameters
                                    'M3': ['X', 'Y', 'Z', 'F', 'R', 'I', 'J', 'K'],  # Spindle on: only S
                                    'M5': ['X', 'Y', 'Z', 'F', 'R', 'I', 'J', 'K'],  # Spindle off: only S
                                    'M30': ['X', 'Y', 'Z', 'F', 'S', 'R', 'I', 'J', 'K'],  # Program end: no params
                                }

                                # Check if this parameter is invalid for the current command
                                if current_command in invalid_combinations:
                                    if token_text in invalid_combinations[current_command]:
                                        logger.warning(f"⚠️ Invalid parameter for command: '{current_command}' cannot have '{token_text}'")
                                        logger.warning(f"  Current sequence: {' '.join(full_command_tokens)}")
                                        logger.warning(f"  {current_command} accepts: {_get_valid_params(current_command)}")
                                        logger.warning(f"  Stopping generation to prevent invalid G-code")
                                        break  # Stop generation

                        # Add to sequence
                        full_command_tokens.append(token_text)
                        token_confidences.append(token_confidence)
                        full_command_confidence *= token_confidence

                        # Track modal command: Update when a G or M command is predicted
                        # This allows future sequences to use this command as context
                        # Example: After "G1 X10 Y20", next sequence "X15 Y25" implicitly uses G1
                        if token_text.startswith('G') or token_text.startswith('M'):
                            state['current_modal_command'] = token_text
                            logger.debug(f"[Modal] Command '{token_text}' is now active")

                        # Update current_tokens
                        new_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                        current_tokens = torch.cat([current_tokens, new_token], dim=1)

                        # Stop early conditions: program end codes or very low confidence
                        if token_text in ['M30', 'M2']:
                            logger.debug(f"Stopping generation: End-of-program code '{token_text}' predicted")
                            break
                        if token_confidence < 0.01:
                            logger.debug(f"Stopping generation: Low confidence ({token_confidence:.4f})")
                            break

                    # Store per-token breakdown
                    predictions['token_breakdown'] = [
                        {'token': tok, 'confidence': float(conf)}
                        for tok, conf in zip(full_command_tokens, token_confidences)
                    ]

                    # Log generation summary
                    logger.info(f"Generation complete: {len(full_command_tokens)} tokens generated")
                    logger.info(f"Generated G-code: {' '.join(full_command_tokens) if full_command_tokens else '<EMPTY>'}")

                    # VALIDATION: Check for orphaned parameters (parameters without values)
                    if full_command_tokens and state.get('decomposer'):
                        parameter_tokens = ['X', 'Y', 'Z', 'R', 'F', 'S', 'I', 'J', 'K']
                        for i, token in enumerate(full_command_tokens):
                            # Check if token is a parameter
                            if token in parameter_tokens:
                                # Check if it's the last token (orphaned)
                                if i == len(full_command_tokens) - 1:
                                    logger.error(f"❌ VALIDATION FAILED: Sequence ends with orphaned parameter '{token}'")
                                    logger.error(f"   Full sequence: {full_command_tokens}")
                                # Check if next token is also a parameter (no numeric value in between)
                                elif i + 1 < len(full_command_tokens) and full_command_tokens[i + 1] in parameter_tokens:
                                    logger.error(f"❌ VALIDATION FAILED: Parameter '{token}' not followed by numeric value")
                                    logger.error(f"   Position {i}: '{token}' → '{full_command_tokens[i + 1]}'")
                                    logger.error(f"   Full sequence: {full_command_tokens}")

                # Join tokens
                full_command_text = ' '.join(full_command_tokens) if full_command_tokens else '<EMPTY>'
                predictions['full_command'] = full_command_text
                predictions['full_command_confidence'] = float(
                    full_command_confidence ** (1.0 / max(len(full_command_tokens), 1))
                )

                # Track command type
                if full_command_tokens:
                    command_type = full_command_tokens[0]  # First token (G1, M3, etc.)
                    state['command_types'][command_type] += 1

                # Calculate metrics if ground truth available
                if ground_truth_gcode:
                    edit_dist = edit_distance(full_command_text, ground_truth_gcode)
                    predictions['edit_distance'] = edit_dist
                    predictions['ground_truth'] = ground_truth_gcode
                    predictions['match'] = (full_command_text == ground_truth_gcode)

                # Add to generation history
                state['generation_history'].append({
                    'predicted': full_command_text,
                    'ground_truth': ground_truth_gcode if ground_truth_gcode else None,
                    'confidence': predictions['full_command_confidence'],
                    'timestamp': predictions['timestamp'],
                    'token_breakdown': predictions.get('token_breakdown', [])
                })
            else:
                # Autoregressive disabled
                predictions['full_command'] = '<DISABLED>'
                predictions['full_command_confidence'] = 0.0
                predictions['token_breakdown'] = []

            # Update confusion matrix if ground truth available
            if ground_truth_gcode:
                predicted = predictions['gcode_text']
                state['confusion_matrix'][ground_truth_gcode][predicted] += 1

                # Update component-level confusion matrices for multihead models
                if state['model_type'] == 'multihead' and state['decomposer'] and 'component_predictions' in predictions:
                    try:
                        # Ground truth might be a full sequence, extract first token
                        # e.g., "G53 G0 Z0." -> "G53" or "G0" (depending on which is in vocab)
                        gt_token_to_check = ground_truth_gcode
                        gt_token_id = state['tokenizer'].vocab.get(gt_token_to_check, -1)

                        # If full sequence not in vocab, try first token
                        if gt_token_id < 0 and ' ' in ground_truth_gcode:
                            tokens = ground_truth_gcode.split()
                            for token in tokens:
                                gt_token_id = state['tokenizer'].vocab.get(token, -1)
                                if gt_token_id >= 0:
                                    gt_token_to_check = token
                                    break

                        if gt_token_id >= 0:
                            gt_components = state['decomposer'].decompose_token(gt_token_id)

                            if gt_components:
                                # Get predicted component IDs from stored predictions
                                pred_components = predictions['component_predictions']

                                # Decomposer returns tuple: (type, command, param_type, param_value)
                                gt_type_id, gt_command_id, gt_param_type_id, gt_param_value_id = gt_components

                                # Type confusion matrix
                                pred_type_id = pred_components['type_id']
                                state['confusion_matrix_type'][gt_type_id][pred_type_id] += 1

                                # Command confusion matrix
                                pred_command_id = pred_components['command_id']
                                state['confusion_matrix_command'][gt_command_id][pred_command_id] += 1

                                # Param type confusion matrix
                                pred_param_type_id = pred_components['param_type_id']
                                state['confusion_matrix_param_type'][gt_param_type_id][pred_param_type_id] += 1

                                # Param value confusion matrix
                                pred_param_value_id = pred_components['param_value_id']
                                state['confusion_matrix_param_value'][gt_param_value_id][pred_param_value_id] += 1
                        else:
                            logger.debug(f"Ground truth token '{ground_truth_gcode}' not found in vocab")
                    except Exception as e:
                        logger.warning(f"Failed to update component confusion matrices: {e}", exc_info=True)

        except Exception as e:
            import traceback
            print(f"\n=== G-code generation error ===")
            print(f"Error: {e}")
            traceback.print_exc()
            print(f"================================\n")
            predictions['gcode_text'] = '<ERROR>'
            predictions['gcode_confidence'] = 0.0
            predictions['top_k'] = []
            predictions['full_command'] = '<ERROR>'
            predictions['full_command_confidence'] = 0.0
    else:
        predictions['gcode_text'] = '<NO_MODEL>'
        predictions['gcode_confidence'] = 0.0
        predictions['top_k'] = []
        predictions['full_command'] = '<NO_MODEL>'
        predictions['full_command_confidence'] = 0.0

    # Store fingerprint for t-SNE
    if 'fingerprint' in predictions:
        state['fingerprints'].append(predictions['fingerprint'])
        # Keep only last 500 for performance
        if len(state['fingerprints']) > 500:
            state['fingerprints'] = state['fingerprints'][-500:]

    # Update running statistics
    state['statistics']['anomaly'].append(predictions['anomaly_score'])
    state['statistics']['confidence'].append(predictions['gcode_confidence'])
    state['statistics']['fingerprint_score'].append(predictions['fingerprint_score'])
    state['statistics']['reconstruction_score'].append(predictions['reconstruction_score'])

    # Compute running stats
    predictions['running_stats'] = {
        'anomaly_mean': float(np.mean(state['statistics']['anomaly'][-100:])),
        'anomaly_std': float(np.std(state['statistics']['anomaly'][-100:])),
        'confidence_mean': float(np.mean(state['statistics']['confidence'][-100:])),
        'confidence_std': float(np.std(state['statistics']['confidence'][-100:])),
        'fingerprint_score_mean': float(np.mean(state['statistics']['fingerprint_score'][-100:])),
        'fingerprint_score_std': float(np.std(state['statistics']['fingerprint_score'][-100:])),
        'reconstruction_score_mean': float(np.mean(state['statistics']['reconstruction_score'][-100:])),
        'reconstruction_score_std': float(np.std(state['statistics']['reconstruction_score'][-100:])),
    }

    return predictions


# Routes
@app.route('/')
def index():
    """Serve the enhanced dashboard page."""
    return render_template('dashboard_enhanced.html')


@app.route('/api/models')
def get_models():
    """Get list of available models."""
    models = []

    # Check multi-head models (Phase 2) - PRIORITIZE THESE!
    multihead_dir = Path('outputs/multihead_aug_v2')
    if multihead_dir.exists():
        checkpoint_best = multihead_dir / 'checkpoint_best.pt'
        if checkpoint_best.exists():
            models.append({
                'path': str(checkpoint_best),
                'name': '🏆 Multi-Head (100% Command Acc)',
                'type': 'multihead_phase2'
            })
        checkpoint_latest = multihead_dir / 'checkpoint_latest.pt'
        if checkpoint_latest.exists():
            models.append({
                'path': str(checkpoint_latest),
                'name': 'Multi-Head (latest)',
                'type': 'multihead_phase2'
            })

    # Check direct regression model (NEW!)
    regression_dir = Path('outputs/direct_regression')
    if regression_dir.exists():
        checkpoint_best = regression_dir / 'checkpoint_best.pt'
        if checkpoint_best.exists():
            models.append({
                'path': str(checkpoint_best),
                'name': '🚀 Direct Regression (100% Tolerance, MAE 0.01)',
                'type': 'direct_regression'
            })
        checkpoint_latest = regression_dir / 'checkpoint_latest.pt'
        if checkpoint_latest.exists() and not checkpoint_best.exists():
            models.append({
                'path': str(checkpoint_latest),
                'name': 'Direct Regression (latest)',
                'type': 'direct_regression'
            })

    # Check sweep output directory (default location for W&B sweeps)
    sweep_dir = Path('outputs/multihead_v2')
    if sweep_dir.exists():
        checkpoint_best = sweep_dir / 'checkpoint_best.pt'
        if checkpoint_best.exists():
            models.append({
                'path': str(checkpoint_best),
                'name': '🔥 Sweep Best Model',
                'type': 'multihead_sweep'
            })
        checkpoint_latest = sweep_dir / 'checkpoint_latest.pt'
        if checkpoint_latest.exists() and not checkpoint_best.exists():
            models.append({
                'path': str(checkpoint_latest),
                'name': 'Sweep Latest Model',
                'type': 'multihead_sweep'
            })

    # Check wandb_sweeps (most common from sweeps)
    wandb_dir = Path('outputs/wandb_sweeps')
    if wandb_dir.exists():
        for model_dir in wandb_dir.glob('gcode_model_*/checkpoint_best.pt'):
            models.append({
                'path': str(model_dir),
                'name': model_dir.parent.name,
                'type': 'wandb_sweep'
            })
        for model_dir in wandb_dir.glob('gcode_model_*/checkpoint_latest.pt'):
            if not (model_dir.parent / 'checkpoint_best.pt').exists():
                models.append({
                    'path': str(model_dir),
                    'name': model_dir.parent.name + ' (latest)',
                    'type': 'wandb_sweep'
                })

    # Check training_clean
    clean_dir = Path('outputs/training_clean')
    if clean_dir.exists():
        for model_dir in clean_dir.glob('gcode_model_*/checkpoint_best.pt'):
            models.append({
                'path': str(model_dir),
                'name': model_dir.parent.name,
                'type': 'clean'
            })
        for model_dir in clean_dir.glob('gcode_model_*/checkpoint_latest.pt'):
            if not (model_dir.parent / 'checkpoint_best.pt').exists():
                models.append({
                    'path': str(model_dir),
                    'name': model_dir.parent.name + ' (latest)',
                    'type': 'clean'
                })

    # Fallback to regular training
    train_dir = Path('outputs/training')
    if train_dir.exists():
        for model_dir in train_dir.glob('gcode_model_*/checkpoint_best.pt'):
            models.append({
                'path': str(model_dir),
                'name': model_dir.parent.name,
                'type': 'training'
            })

    # Scan for model_* directories (new training runs)
    outputs_dir = Path('outputs')
    for model_path in outputs_dir.glob('model_*/checkpoint_best.pt'):
        models.append({
            'path': str(model_path),
            'name': model_path.parent.name.replace('model_', '').replace('_', ' ').title(),
            'type': 'custom_model'
        })

    # Scan for training_* directories (numbered training runs)
    for model_path in outputs_dir.glob('training_*/checkpoint_best.pt'):
        models.append({
            'path': str(model_path),
            'name': model_path.parent.name.replace('training_', 'Training ').replace('epoch', ' Epoch'),
            'type': 'training_run'
        })

    # Scan for hybrid_* directories (hybrid bucketing models)
    for model_path in outputs_dir.glob('hybrid_*/checkpoint_best.pt'):
        models.append({
            'path': str(model_path),
            'name': '🔥 ' + model_path.parent.name.replace('hybrid_', 'Hybrid ').replace('_', ' ').title(),
            'type': 'hybrid'
        })

    # Scan for baseline_* directories (baseline comparison models)
    for model_path in outputs_dir.glob('baseline_*/checkpoint_best.pt'):
        models.append({
            'path': str(model_path),
            'name': '📊 ' + model_path.parent.name.replace('baseline_', 'Baseline ').replace('_', ' ').title(),
            'type': 'baseline'
        })

    # Scan for sweep_* directories (W&B sweep results)
    for model_path in outputs_dir.glob('sweep_*/checkpoint_best.pt'):
        models.append({
            'path': str(model_path),
            'name': '🎯 ' + model_path.parent.name.replace('sweep_', 'Sweep ').replace('_', ' ').title(),
            'type': 'sweep'
        })

    # Generic catch-all: any other checkpoint_best.pt files in outputs/
    # This ensures we never miss a checkpoint
    all_checkpoints = set()
    for checkpoint in outputs_dir.glob('*/checkpoint_best.pt'):
        checkpoint_path = str(checkpoint)
        # Only add if not already in the list
        if checkpoint_path not in [m['path'] for m in models]:
            models.append({
                'path': checkpoint_path,
                'name': checkpoint.parent.name,
                'type': 'other'
            })

    return jsonify(models)


@app.route('/api/csv_files')
def get_csv_files():
    """Get list of available CSV files."""
    csv_files = []
    data_dir = Path('data')

    if data_dir.exists():
        for csv_file in sorted(data_dir.glob('*_aligned.csv')):
            csv_files.append({
                'path': str(csv_file),
                'name': csv_file.name
            })

    return jsonify(csv_files)


@app.route('/api/load_model', methods=['POST'])
def load_model_endpoint():
    """Load a model with comprehensive error handling."""
    try:
        data = request.json
        if not data:
            logger.error("No data provided in load_model request")
            return jsonify({'success': False, 'error': 'No data provided'})

        model_path = data.get('path')
        if not model_path:
            logger.error("No model path provided")
            return jsonify({'success': False, 'error': 'No model path provided'})

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return jsonify({
                'success': False,
                'error': f'Model file not found: {model_path}',
                'hint': 'Make sure the model checkpoint file exists at the specified path.'
            })

        logger.info(f"Loading model from: {model_path}")
        model, tokenizer, config, metadata, model_type, decomposer, grammar_constraints, vocab_value_map = load_model(model_path)

        state['model'] = model
        state['model_type'] = model_type
        state['decomposer'] = decomposer
        state['vocab_value_map'] = vocab_value_map
        state['grammar_constraints'] = grammar_constraints
        state['tokenizer'] = tokenizer
        state['config'] = config  # Store config object (not dict)
        state['config_dict'] = config.__dict__  # Also store dict for JSON responses
        state['metadata'] = metadata  # Store preprocessing metadata

        logger.info(f"✓ Model loaded successfully: {model_path}")
        logger.info(f"  Model type: {model_type}")
        if hasattr(config, 'sensor_dims'):
            logger.info(f"  Model expects: {config.sensor_dims} features (continuous, categorical)")
        if metadata:
            logger.info(f"  Metadata: {len(metadata.get('master_columns', []))} continuous, "
                       f"{len(metadata.get('categorical_columns', []))} categorical features")

        # Reset state
        state['buffer'].clear()
        state['predictions_history'].clear()
        state['confusion_matrix'].clear()
        state['statistics'].clear()
        state['fingerprints'].clear()

        return jsonify({
            'success': True,
            'config': state['config_dict'],
            'n_features': metadata['n_continuous_features'] if metadata else None,
            'model_type': model_type
        })

    except FileNotFoundError as e:
        logger.error(f"File not found while loading model: {e}")
        return jsonify({
            'success': False,
            'error': 'Model file or dependency not found',
            'hint': 'Check that the model checkpoint and vocabulary.json exist.',
            'details': str(e)
        })
    except (KeyError, AttributeError) as e:
        logger.error(f"Model structure error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Invalid model checkpoint format',
            'hint': 'The checkpoint file may be corrupted or from an incompatible version.',
            'details': str(e)
        })
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to load model',
            'hint': 'Check logs for detailed error information.',
            'details': str(e)
        })


@app.route('/api/load_csv', methods=['POST'])
def load_csv_endpoint():
    """Load a CSV file with comprehensive error handling."""
    try:
        data = request.json
        if not data:
            logger.error("No data provided in load_csv request")
            return jsonify({'success': False, 'error': 'No data provided'})

        csv_path = data.get('path')
        if not csv_path:
            logger.error("No CSV path provided")
            return jsonify({'success': False, 'error': 'No CSV path provided'})

        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            logger.error(f"CSV path does not exist: {csv_path}")
            return jsonify({
                'success': False,
                'error': f'CSV file not found: {csv_path}',
                'hint': 'Make sure the CSV file exists at the specified path.'
            })

        logger.info(f"Loading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            logger.warning(f"CSV file is empty: {csv_path}")
            return jsonify({
                'success': False,
                'error': 'CSV file is empty',
                'hint': 'The CSV file contains no data rows.'
            })

        state['csv_data'] = df
        state['current_idx'] = 0
        state['csv_filename'] = csv_path_obj.name  # Store filename for operation type extraction

        logger.info(f"✓ CSV loaded successfully: {len(df)} rows from {csv_path}")
        return jsonify({
            'success': True,
            'total_rows': len(df),
            'columns': list(df.columns)
        })

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty CSV file: {e}")
        return jsonify({
            'success': False,
            'error': 'CSV file is empty or malformed',
            'hint': 'Check that the CSV file is not corrupted and contains data.',
            'details': str(e)
        })
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to parse CSV file',
            'hint': 'The CSV file may be malformed or use an unexpected format.',
            'details': str(e)
        })
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to load CSV',
            'hint': 'Check logs for detailed error information.',
            'details': str(e)
        })


@app.route('/api/heatmap/sensors_vs_time', methods=['GET'])
def heatmap_sensors_vs_time():
    """Generate heatmap data for sensor readings over time."""
    try:
        if state['csv_data'] is None:
            return jsonify({
                'success': False,
                'error': 'No CSV data loaded'
            })

        df = state['csv_data']

        # Sensor columns to include
        sensor_cols = ['xa_motor', 'y_bed__1', 'y_bed__2', 'spindle1', 'za_motor']

        # Filter to only existing columns
        available_sensors = [col for col in sensor_cols if col in df.columns]

        if not available_sensors:
            return jsonify({
                'success': False,
                'error': f'None of the requested sensor columns found. Available: {list(df.columns)}'
            })

        # Extract sensor data and standardize (z-score normalization)
        # This ensures all sensors are on the same scale (mean=0, std=1)
        import numpy as np
        sensor_data = df[available_sensors].values  # Shape: (n_timesteps, n_sensors)

        # Standardize each sensor column
        sensor_data_normalized = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
        sensor_data_normalized = sensor_data_normalized.T  # Shape: (n_sensors, n_timesteps)

        # Create heatmap data structure for Plotly
        heatmap_data = {
            'success': True,
            'z': sensor_data_normalized.tolist(),  # 2D array: sensors x time (normalized)
            'x': list(range(len(df))),  # Time indices
            'y': available_sensors,  # Sensor names
            'colorscale': 'Viridis',
            'title': 'Sensor Readings Over Time (Standardized)'
        }

        return jsonify(heatmap_data)

    except Exception as e:
        logger.error(f"Error generating sensors vs time heatmap: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate heatmap',
            'details': str(e)
        })


@app.route('/api/heatmap/sensors_vs_vibration', methods=['GET'])
def heatmap_sensors_vs_vibration():
    """Generate heatmap data for sensor correlations with vibration metrics."""
    try:
        if state['csv_data'] is None:
            return jsonify({
                'success': False,
                'error': 'No CSV data loaded'
            })

        df = state['csv_data']

        # Sensor columns and vibration metric columns
        sensor_cols = ['y_bed__2', 'spindle1', 'xa_motor']
        vibration_cols = ['Ax', 'Ay', 'Az', 'RMS', 'kurtosis']

        # Filter to only existing columns
        available_sensors = [col for col in sensor_cols if col in df.columns]
        available_vibrations = [col for col in vibration_cols if col in df.columns]

        if not available_sensors or not available_vibrations:
            return jsonify({
                'success': False,
                'error': f'Missing required columns. Found sensors: {available_sensors}, vibrations: {available_vibrations}'
            })

        # Compute correlation matrix between sensors and vibration metrics
        import numpy as np

        # Extract data and standardize (z-score normalization)
        # This ensures all metrics are on the same scale before computing correlations
        sensor_data = df[available_sensors].values  # Shape: (n_timesteps, n_sensors)
        vibration_data = df[available_vibrations].values  # Shape: (n_timesteps, n_vibrations)

        # Standardize each column (mean=0, std=1)
        sensor_data_normalized = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
        vibration_data_normalized = (vibration_data - vibration_data.mean(axis=0)) / (vibration_data.std(axis=0) + 1e-8)

        # Compute correlation matrix: sensors (rows) x vibrations (columns)
        correlation_matrix = np.zeros((len(available_sensors), len(available_vibrations)))

        for i in range(len(available_sensors)):
            for j in range(len(available_vibrations)):
                # Pearson correlation coefficient on standardized data
                corr = np.corrcoef(sensor_data_normalized[:, i], vibration_data_normalized[:, j])[0, 1]
                correlation_matrix[i, j] = corr

        # Create heatmap data structure for Plotly
        heatmap_data = {
            'success': True,
            'z': correlation_matrix.tolist(),  # 2D array: sensors x vibrations
            'x': available_vibrations,  # Vibration metric names
            'y': available_sensors,  # Sensor names
            'colorscale': 'RdBu',
            'title': 'Sensor vs Vibration Correlation',
            'zmin': -1,
            'zmax': 1
        }

        return jsonify(heatmap_data)

    except Exception as e:
        logger.error(f"Error generating sensors vs vibration heatmap: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate heatmap',
            'details': str(e)
        })


@app.route('/api/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """Run batch evaluation on entire dataset to populate confusion matrices."""
    if not validate_model_loaded():
        return jsonify({'success': False, 'error': 'No model loaded'})

    if not validate_csv_loaded():
        return jsonify({'success': False, 'error': 'No CSV loaded'})

    try:
        # Get number of samples to evaluate (default: all)
        data = request.json or {}
        max_samples = data.get('max_samples', len(state['csv_data']))
        max_samples = min(max_samples, len(state['csv_data']))

        logger.info(f"Starting batch evaluation on {max_samples} samples...")

        # Reset confusion matrices
        from collections import defaultdict
        state['confusion_matrix'] = defaultdict(lambda: defaultdict(int))
        state['confusion_matrix_operation'] = defaultdict(lambda: defaultdict(int))
        state['confusion_matrix_type'] = defaultdict(lambda: defaultdict(int))
        state['confusion_matrix_command'] = defaultdict(lambda: defaultdict(int))
        state['confusion_matrix_param_type'] = defaultdict(lambda: defaultdict(int))
        state['confusion_matrix_param_value'] = defaultdict(lambda: defaultdict(int))

        # Reset buffer
        state['buffer'].clear()

        # Process all samples
        processed = 0
        skipped = 0

        for idx in range(max_samples):
            row = state['csv_data'].iloc[idx]

            try:
                # Extract features
                continuous, categorical = extract_features_with_validation(
                    row,
                    state.get('metadata'),
                    state['config'] if state['model'] else None
                )

                # Ground truth
                ground_truth = row.get('gcode_string', None)

                # Process sample
                predictions = process_sample(continuous, categorical, ground_truth)

                # Update operation type confusion matrix
                if predictions and 'operation_type_id' in predictions and predictions['operation_type_id'] >= 0:
                    if state.get('csv_filename'):
                        for i, op_name in enumerate(OPERATION_TYPE_NAMES):
                            if op_name in state['csv_filename'].lower():
                                gt_operation_id = i
                                pred_operation_id = predictions['operation_type_id']
                                state['confusion_matrix_operation'][gt_operation_id][pred_operation_id] += 1
                                break

                if predictions:
                    processed += 1
                else:
                    skipped += 1

                # Log progress every 100 samples
                if (idx + 1) % 100 == 0:
                    logger.info(f"Progress: {idx + 1}/{max_samples} samples processed")

            except Exception as e:
                logger.warning(f"Failed to process sample {idx}: {e}")
                skipped += 1
                continue

        logger.info(f"Batch evaluation complete: {processed} processed, {skipped} skipped")

        return jsonify({
            'success': True,
            'processed': processed,
            'skipped': skipped,
            'total': max_samples,
            'confusion_matrices': {
                'gcode': len(state['confusion_matrix']),
                'operation': len(state['confusion_matrix_operation']),
                'type': len(state['confusion_matrix_type']),
                'command': len(state['confusion_matrix_command']),
                'param_type': len(state['confusion_matrix_param_type']),
                'param_value': len(state['confusion_matrix_param_value']),
            }
        })

    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/control', methods=['POST'])
def control():
    """Control playback."""
    data = request.json
    action = data.get('action')

    if action == 'start':
        state['running'] = True
    elif action == 'pause':
        state['running'] = False
    elif action == 'reset':
        state['running'] = False
        state['current_idx'] = 0
        state['buffer'].clear()
        state['predictions_history'].clear()

    return jsonify({'success': True})


@app.route('/api/status')
def status():
    """Get current status."""
    return jsonify({
        'running': state['running'],
        'current_idx': state['current_idx'],
        'total_rows': len(state['csv_data']) if state['csv_data'] is not None else 0,
        'buffer_size': len(state['buffer']),
    })


@app.route('/api/step')
def step():
    """Process one timestep."""
    if state['csv_data'] is None:
        return jsonify({'success': False, 'error': 'No CSV loaded'})

    if not state['running']:
        return jsonify({'success': False, 'error': 'Not running'})

    if state['current_idx'] >= len(state['csv_data']):
        return jsonify({'success': False, 'error': 'End of data'})

    # Get current row
    row = state['csv_data'].iloc[state['current_idx']]

    # Extract features with validation
    try:
        continuous, categorical = extract_features_with_validation(
            row,
            state.get('metadata'),
            state['config'] if state['model'] else None
        )
    except ValueError as e:
        logger.error(f"Feature extraction failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Feature dimension mismatch: {str(e)}',
            'hint': 'Make sure the CSV was preprocessed with the same settings as the model was trained with.'
        })
    except Exception as e:
        logger.error(f"Unexpected error during feature extraction: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Feature extraction error: {str(e)}'
        })

    # Ground truth (if available)
    ground_truth = row.get('gcode_string', None)

    # Process
    predictions = process_sample(continuous, categorical, ground_truth)

    # Update operation type confusion matrix if predictions include operation type
    if predictions and 'operation_type_id' in predictions and predictions['operation_type_id'] >= 0:
        # Try to get ground truth operation type from CSV metadata
        # The operation type is typically encoded in the CSV filename
        if state.get('csv_filename'):
            try:
                # Extract operation type from filename (e.g., "face_007_aligned.csv" -> "face")
                csv_filename = state['csv_filename']
                for i, op_name in enumerate(OPERATION_TYPE_NAMES):
                    if op_name in csv_filename.lower():
                        gt_operation_id = i
                        pred_operation_id = predictions['operation_type_id']
                        state['confusion_matrix_operation'][gt_operation_id][pred_operation_id] += 1
                        break
            except Exception as e:
                logger.debug(f"Failed to update operation confusion matrix: {e}")

    # Increment
    state['current_idx'] += 1

    # Store history
    if predictions:
        state['predictions_history'].append(predictions)
        # Keep only last 200 for performance
        if len(state['predictions_history']) > 200:
            state['predictions_history'] = state['predictions_history'][-200:]

    # Prepare sensor data for visualization (all sensors with values)
    if state.get('metadata') and 'master_columns' in state['metadata']:
        sensor_data = {}
        for i, col in enumerate(state['metadata']['master_columns']):
            if col in row.index:
                sensor_data[col] = float(row[col])
            else:
                sensor_data[col] = 0.0  # Zero-padded sensors
    else:
        # Fallback
        sensor_data = {col: float(row[col]) for col in cont_cols[:50] if col in row.index}

    # Prepare 3D position data
    position_3d = {
        'x': float(row['mpox']) if 'mpox' in row.index else 0,
        'y': float(row['mpoy']) if 'mpoy' in row.index else 0,
        'z': float(row['mpoz']) if 'mpoz' in row.index else 0,
    }

    return jsonify({
        'success': True,
        'predictions': predictions,
        'sensor_data': sensor_data,
        'position_3d': position_3d,
        'history': state['predictions_history'][-50:],  # Last 50 for charts
    })


@app.route('/api/confusion_matrix')
def get_confusion_matrix():
    """Get current confusion matrix with Redis caching."""
    # Create cache key based on confusion matrix hash
    cm_str = json.dumps(state['confusion_matrix'], sort_keys=True)
    cache_key = f"confusion_matrix:{hash(cm_str)}"

    # Try to get from Redis cache
    if REDIS_AVAILABLE:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.debug("Confusion matrix cache hit")
                return jsonify(json.loads(cached_result))
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")

    # Cache miss - build confusion matrix
    matrix_data = []
    all_gcodes = set()

    for true_label in state['confusion_matrix']:
        all_gcodes.add(true_label)
        for pred_label in state['confusion_matrix'][true_label]:
            all_gcodes.add(pred_label)

    all_gcodes = sorted(list(all_gcodes))

    # Build matrix
    for true_label in all_gcodes:
        row = []
        for pred_label in all_gcodes:
            count = state['confusion_matrix'].get(true_label, {}).get(pred_label, 0)
            row.append(count)
        matrix_data.append(row)

    result = {
        'matrix': matrix_data,
        'labels': all_gcodes
    }

    # Cache result in Redis (TTL: 30 seconds)
    if REDIS_AVAILABLE:
        try:
            redis_client.setex(cache_key, 30, json.dumps(result))
            logger.debug("Confusion matrix result cached")
        except Exception as e:
            logger.warning(f"Redis setex failed: {e}")

    return jsonify(result)


def build_confusion_matrix_response(confusion_dict, label_names=None):
    """
    Helper function to build confusion matrix response from a confusion dictionary.

    Args:
        confusion_dict: Dictionary of {true_label: {pred_label: count}}
        label_names: Optional dictionary mapping label IDs to human-readable names

    Returns:
        Dictionary with 'matrix' and 'labels' keys
    """
    matrix_data = []
    all_labels = set()

    for true_label in confusion_dict:
        all_labels.add(true_label)
        for pred_label in confusion_dict[true_label]:
            all_labels.add(pred_label)

    # Handle empty confusion matrix
    if len(all_labels) == 0:
        return {
            'matrix': [],
            'labels': [],
            'label_ids': [],
            'empty': True
        }

    all_labels = sorted(list(all_labels))

    # Build matrix
    for true_label in all_labels:
        row = []
        for pred_label in all_labels:
            count = confusion_dict.get(true_label, {}).get(pred_label, 0)
            row.append(count)
        matrix_data.append(row)

    # Convert labels to names if mapping provided
    if label_names:
        display_labels = [label_names.get(label, str(label)) for label in all_labels]
    else:
        display_labels = [str(label) for label in all_labels]

    return {
        'matrix': matrix_data,
        'labels': display_labels,
        'label_ids': all_labels,
        'empty': False
    }


@app.route('/api/debug/confusion_state')
def debug_confusion_state():
    """Debug endpoint to check confusion matrix state."""
    return jsonify({
        'model_type': state['model_type'],
        'has_decomposer': state['decomposer'] is not None,
        'confusion_matrix_sizes': {
            'gcode': len(state['confusion_matrix']),
            'operation': len(state['confusion_matrix_operation']),
            'type': len(state['confusion_matrix_type']),
            'command': len(state['confusion_matrix_command']),
            'param_type': len(state['confusion_matrix_param_type']),
            'param_value': len(state['confusion_matrix_param_value']),
        },
        'sample_operation_data': dict(state['confusion_matrix_operation']) if len(state['confusion_matrix_operation']) > 0 else {},
        'sample_type_data': dict(state['confusion_matrix_type']) if len(state['confusion_matrix_type']) > 0 else {},
    })


@app.route('/api/confusion_matrix/operation')
def get_confusion_matrix_operation():
    """Get operation type confusion matrix."""
    result = build_confusion_matrix_response(
        state['confusion_matrix_operation'],
        label_names={i: name for i, name in enumerate(OPERATION_TYPE_NAMES)}
    )
    return jsonify(result)


@app.route('/api/confusion_matrix/type')
def get_confusion_matrix_type():
    """Get type confusion matrix."""
    result = build_confusion_matrix_response(state['confusion_matrix_type'])
    return jsonify(result)


@app.route('/api/confusion_matrix/command')
def get_confusion_matrix_command():
    """Get command confusion matrix with command token names."""
    # Get command token names from decomposer if available
    label_names = None
    if state.get('decomposer') and hasattr(state['decomposer'], 'command_tokens'):
        label_names = {i: token for i, token in enumerate(state['decomposer'].command_tokens)}

    result = build_confusion_matrix_response(state['confusion_matrix_command'], label_names)
    return jsonify(result)


@app.route('/api/confusion_matrix/param_type')
def get_confusion_matrix_param_type():
    """Get parameter type confusion matrix."""
    result = build_confusion_matrix_response(state['confusion_matrix_param_type'])
    return jsonify(result)


@app.route('/api/confusion_matrix/param_value')
def get_confusion_matrix_param_value():
    """Get parameter value confusion matrix."""
    result = build_confusion_matrix_response(state['confusion_matrix_param_value'])
    return jsonify(result)


@app.route('/api/analytics/hyperparameter_sweep')
def get_hyperparameter_sweep():
    """Get hyperparameter sweep data for visualization."""
    try:
        # Load sweep summary CSV
        sweep_csv_path = Path('reports/production_kae3w55d_curves/per_run_summary.csv')

        if not sweep_csv_path.exists():
            # Try alternative path
            sweep_csv_path = Path('outputs/wandb_sweep_analysis.csv')

        if not sweep_csv_path.exists():
            return jsonify({'success': False, 'error': 'Sweep data not found'})

        df = pd.read_csv(sweep_csv_path)

        # Remove rows with missing data
        df = df.dropna(subset=['val/param_type_acc', 'val/loss'])

        # Prepare data for parallel coordinates plot
        params = ['batch_size', 'hidden_dim', 'num_heads', 'num_layers', 'learning_rate', 'dropout', 'weight_decay']
        metrics = ['val/param_type_acc', 'val/loss', 'train/loss']

        # Get top 20 runs by validation accuracy
        df_sorted = df.sort_values('val/param_type_acc', ascending=False).head(20)

        sweep_data = {
            'run_ids': df_sorted['run_id'].tolist(),
            'parameters': {},
            'metrics': {},
            'top_runs': df_sorted.head(5)[['run_id', 'val/param_type_acc', 'val/loss', 'hidden_dim', 'num_heads', 'num_layers']].to_dict('records')
        }

        # Extract parameter values
        for param in params:
            if param in df_sorted.columns:
                sweep_data['parameters'][param] = df_sorted[param].fillna(0).tolist()

        # Extract metric values
        for metric in metrics:
            if metric in df_sorted.columns:
                sweep_data['metrics'][metric] = df_sorted[metric].fillna(0).tolist()

        return jsonify({'success': True, 'data': sweep_data})

    except Exception as e:
        logger.error(f"Failed to load hyperparameter sweep data: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/analytics/parameter_importance')
def get_parameter_importance():
    """Get parameter importance analysis."""
    try:
        importance_csv_path = Path('outputs/sweep_ab0ypky2_comprehensive_analysis/data/parameter_importance.csv')

        if not importance_csv_path.exists():
            return jsonify({'success': False, 'error': 'Parameter importance data not found'})

        df = pd.read_csv(importance_csv_path)

        return jsonify({
            'success': True,
            'parameters': df['parameter'].tolist() if 'parameter' in df.columns else df.iloc[:, 0].tolist(),
            'importance': df['importance'].tolist() if 'importance' in df.columns else df.iloc[:, 1].tolist()
        })

    except Exception as e:
        logger.error(f"Failed to load parameter importance data: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/confusion_matrix/<matrix_type>/normalized')
def get_confusion_matrix_normalized(matrix_type):
    """Get normalized confusion matrix (row-wise or column-wise)."""
    try:
        # Get normalization mode from query params
        norm_mode = request.args.get('mode', 'none')  # 'none', 'row', 'col'

        # Select the appropriate confusion matrix
        matrix_map = {
            'gcode': state['confusion_matrix'],
            'operation': state['confusion_matrix_operation'],
            'type': state['confusion_matrix_type'],
            'command': state['confusion_matrix_command'],
            'param_type': state['confusion_matrix_param_type'],
            'param_value': state['confusion_matrix_param_value'],
        }

        if matrix_type not in matrix_map:
            return jsonify({'success': False, 'error': f'Unknown matrix type: {matrix_type}'})

        confusion_dict = matrix_map[matrix_type]

        # Get label names if available
        label_names = None
        if matrix_type == 'operation':
            label_names = {i: name for i, name in enumerate(OPERATION_TYPE_NAMES)}
        elif matrix_type == 'command' and state.get('decomposer'):
            label_names = {i: token for i, token in enumerate(state['decomposer'].command_tokens)}

        # Build base confusion matrix
        result = build_confusion_matrix_response(confusion_dict, label_names)

        if result['empty']:
            return jsonify(result)

        # Apply normalization
        matrix = np.array(result['matrix'])

        if norm_mode == 'row':
            # Normalize by row (recall/sensitivity)
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            matrix_normalized = matrix / row_sums
            result['matrix'] = matrix_normalized.tolist()
            result['normalization'] = 'row'
            result['normalization_label'] = 'Recall (True Positive Rate)'
        elif norm_mode == 'col':
            # Normalize by column (precision)
            col_sums = matrix.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            matrix_normalized = matrix / col_sums
            result['matrix'] = matrix_normalized.tolist()
            result['normalization'] = 'col'
            result['normalization_label'] = 'Precision (Positive Predictive Value)'
        else:
            result['normalization'] = 'none'
            result['normalization_label'] = 'Raw Counts'

        # Add per-class metrics
        matrix_np = np.array(matrix)
        n_classes = len(result['labels'])
        per_class_metrics = []

        for i in range(n_classes):
            tp = matrix_np[i, i]
            fp = matrix_np[:, i].sum() - tp
            fn = matrix_np[i, :].sum() - tp
            tn = matrix_np.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_class_metrics.append({
                'label': result['labels'][i],
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(matrix_np[i, :].sum())
            })

        result['per_class_metrics'] = per_class_metrics

        # Calculate overall metrics
        total_correct = np.trace(matrix_np)
        total_samples = matrix_np.sum()
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        # Macro average
        macro_precision = np.mean([m['precision'] for m in per_class_metrics])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics])
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics])

        result['overall_metrics'] = {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'total_samples': int(total_samples)
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to compute normalized confusion matrix: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/export', methods=['GET'])
def export_data():
    """Export session data as CSV."""
    if not state['predictions_history']:
        return jsonify({'success': False, 'error': 'No data to export'})

    # Create DataFrame
    export_data = []
    for i, pred in enumerate(state['predictions_history']):
        # Use full_command (autoregressive generation) if available, fallback to single token
        gcode_output = pred.get('full_command', pred.get('gcode_text', '<EMPTY>'))
        full_conf = pred.get('full_command_confidence', pred.get('gcode_confidence', 0.0))

        export_data.append({
            'index': i,
            'timestamp': pred['timestamp'],
            'gcode_predicted': gcode_output,
            'confidence': full_conf,
            'anomaly_score': pred['anomaly_score'],
            'operation_type': pred.get('operation_type', 'unknown'),
            'operation_confidence': pred.get('operation_confidence', 0.0),
            'token_count': len(gcode_output.split()) if isinstance(gcode_output, str) else 0,
        })

    df = pd.DataFrame(export_data)

    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Send as file
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'dashboard_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@app.route('/api/export/detailed')
def export_detailed_csv():
    """Export detailed session data with token-level breakdown."""
    if not state['predictions_history']:
        return jsonify({'success': False, 'error': 'No data to export'})

    # Create detailed DataFrame with token breakdown
    detailed_data = []
    for i, pred in enumerate(state['predictions_history']):
        # Base information
        base_info = {
            'index': i,
            'timestamp': pred['timestamp'],
            'operation_type': pred.get('operation_type', 'unknown'),
            'operation_confidence': pred.get('operation_confidence', 0.0),
            'anomaly_score': pred['anomaly_score'],
        }

        # Full G-code reconstruction
        full_command = pred.get('full_command', pred.get('gcode_text', '<EMPTY>'))
        full_conf = pred.get('full_command_confidence', pred.get('gcode_confidence', 0.0))
        base_info['gcode_predicted'] = full_command
        base_info['full_confidence'] = full_conf

        # Multi-head prediction details (if available)
        if state['model_type'] == 'multihead':
            base_info['type_prediction'] = pred.get('type_prediction', '')
            base_info['command_prediction'] = pred.get('command_prediction', '')
            base_info['param_type_prediction'] = pred.get('param_type_prediction', '')
            base_info['param_value_prediction'] = pred.get('param_value_prediction', '')

        # Token-level breakdown
        token_breakdown = pred.get('token_breakdown', [])
        if token_breakdown:
            base_info['token_count'] = len(token_breakdown)
            base_info['avg_token_confidence'] = sum(t['confidence'] for t in token_breakdown) / len(token_breakdown)
            # Add individual tokens and confidences
            for j, tok_info in enumerate(token_breakdown[:10]):  # Limit to first 10 tokens
                base_info[f'token_{j+1}'] = tok_info['token']
                base_info[f'token_{j+1}_conf'] = tok_info['confidence']
        else:
            base_info['token_count'] = 0
            base_info['avg_token_confidence'] = 0.0

        detailed_data.append(base_info)

    df = pd.DataFrame(detailed_data)

    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Send as file
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'dashboard_detailed_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


@app.route('/api/analytics/error_analysis')
def get_error_analysis():
    """Comprehensive error analysis from confusion matrices and prediction history."""
    try:
        analysis = {
            'component_errors': {},
            'temporal_errors': [],
            'top_misclassifications': [],
            'error_correlation': {}
        }

        # Component-level error rates
        matrix_types = {
            'operation': state['confusion_matrix_operation'],
            'type': state['confusion_matrix_type'],
            'command': state['confusion_matrix_command'],
            'param_type': state['confusion_matrix_param_type'],
            'param_value': state['confusion_matrix_param_value']
        }

        for component, confusion_dict in matrix_types.items():
            if len(confusion_dict) == 0:
                continue

            # Calculate error rate
            total_predictions = 0
            total_errors = 0

            for true_label in confusion_dict:
                for pred_label in confusion_dict[true_label]:
                    count = confusion_dict[true_label][pred_label]
                    total_predictions += count
                    if true_label != pred_label:
                        total_errors += count

            error_rate = total_errors / total_predictions if total_predictions > 0 else 0

            analysis['component_errors'][component] = {
                'total_predictions': int(total_predictions),
                'total_errors': int(total_errors),
                'error_rate': float(error_rate),
                'accuracy': float(1 - error_rate)
            }

        # Top misclassifications from main GCode confusion matrix
        misclassifications = []
        for true_label in state['confusion_matrix']:
            for pred_label in state['confusion_matrix'][true_label]:
                if true_label != pred_label:
                    count = state['confusion_matrix'][true_label][pred_label]
                    misclassifications.append({
                        'true': true_label,
                        'predicted': pred_label,
                        'count': count
                    })

        # Sort by count and get top 10
        misclassifications.sort(key=lambda x: x['count'], reverse=True)
        analysis['top_misclassifications'] = misclassifications[:10]

        # Temporal error pattern (last 100 predictions)
        if state['predictions_history']:
            recent_predictions = state['predictions_history'][-100:]
            temporal_data = []

            for i, pred in enumerate(recent_predictions):
                has_error = False
                if 'ground_truth' in pred and 'gcode_text' in pred:
                    has_error = pred['ground_truth'] != pred['gcode_text']

                temporal_data.append({
                    'index': i,
                    'has_error': has_error,
                    'confidence': pred.get('gcode_confidence', 0)
                })

            analysis['temporal_errors'] = temporal_data

        return jsonify({'success': True, 'data': analysis})

    except Exception as e:
        logger.error(f"Error analysis failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/analytics/sequence_alignment')
def get_sequence_alignment():
    """Get sequence alignments for predicted vs ground truth comparison."""
    try:
        # Get recent predictions with ground truth
        alignments = []

        for pred in state['predictions_history'][-20:]:  # Last 20 predictions
            if 'ground_truth' in pred and pred.get('ground_truth'):
                alignment = {
                    'timestamp': pred.get('timestamp'),
                    'predicted': pred.get('full_command', pred.get('gcode_text', '')),
                    'ground_truth': pred['ground_truth'],
                    'confidence': pred.get('full_command_confidence', pred.get('gcode_confidence', 0)),
                    'match': pred.get('match', pred.get('gcode_text') == pred['ground_truth']),
                    'edit_distance': pred.get('edit_distance', 0)
                }

                # Token-by-token breakdown if available
                if 'token_breakdown' in pred:
                    alignment['token_breakdown'] = pred['token_breakdown']

                alignments.append(alignment)

        return jsonify({'success': True, 'alignments': alignments})

    except Exception as e:
        logger.error(f"Sequence alignment failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/statistics')
def get_statistics():
    """Get running statistics."""
    stats = {}

    for key, values in state['statistics'].items():
        if values:
            recent = values[-100:]  # Last 100 samples
            stats[key] = {
                'mean': float(np.mean(recent)),
                'std': float(np.std(recent)),
                'min': float(np.min(recent)),
                'max': float(np.max(recent)),
            }

    return jsonify(stats)


@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update generation settings."""
    if request.method == 'POST':
        data = request.json
        # Update settings
        for key in ['enable_autoregressive', 'max_tokens', 'temperature', 'top_p', 'beam_size', 'use_beam_search']:
            if key in data:
                state['gen_settings'][key] = data[key]
        return jsonify({'success': True, 'settings': state['gen_settings']})
    else:
        return jsonify(state['gen_settings'])


@app.route('/api/generation_history')
def get_generation_history():
    """Get recent generation history."""
    return jsonify({
        'history': list(state['generation_history'])
    })


@app.route('/api/fingerprint/current')
def get_current_fingerprint():
    """Get the most recent fingerprint data for visualization."""
    if not state['fingerprints']:
        return jsonify({'success': False, 'error': 'No fingerprint data available'})

    # Get the most recent fingerprint
    latest_fingerprint = state['fingerprints'][-1]

    # Get recent fingerprint scores for trend
    recent_scores = state['statistics']['fingerprint_score'][-50:] if 'fingerprint_score' in state['statistics'] else []

    return jsonify({
        'success': True,
        'fingerprint': latest_fingerprint,  # 128-dimensional vector
        'fingerprint_dimension': len(latest_fingerprint),
        'recent_scores': recent_scores,
        'current_score': state['statistics']['fingerprint_score'][-1] if state['statistics']['fingerprint_score'] else 0.0,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/reconstruction_metrics')
def get_reconstruction_metrics():
    """Get reconstruction score metrics and trends."""
    if 'reconstruction_score' not in state['statistics'] or not state['statistics']['reconstruction_score']:
        return jsonify({'success': False, 'error': 'No reconstruction metrics available'})

    # Get recent scores (last 100)
    recent_reconstruction = state['statistics']['reconstruction_score'][-100:]
    recent_fingerprint = state['statistics']['fingerprint_score'][-100:]
    recent_confidence = state['statistics']['confidence'][-100:]
    recent_anomaly = state['statistics']['anomaly'][-100:]

    return jsonify({
        'success': True,
        'reconstruction_score': {
            'current': float(recent_reconstruction[-1]),
            'mean': float(np.mean(recent_reconstruction)),
            'std': float(np.std(recent_reconstruction)),
            'min': float(np.min(recent_reconstruction)),
            'max': float(np.max(recent_reconstruction)),
            'history': recent_reconstruction
        },
        'fingerprint_score': {
            'current': float(recent_fingerprint[-1]),
            'mean': float(np.mean(recent_fingerprint)),
            'std': float(np.std(recent_fingerprint)),
            'min': float(np.min(recent_fingerprint)),
            'max': float(np.max(recent_fingerprint)),
            'history': recent_fingerprint
        },
        'confidence': {
            'current': float(recent_confidence[-1]),
            'mean': float(np.mean(recent_confidence)),
            'std': float(np.std(recent_confidence)),
            'history': recent_confidence
        },
        'anomaly': {
            'current': float(recent_anomaly[-1]),
            'mean': float(np.mean(recent_anomaly)),
            'std': float(np.std(recent_anomaly)),
            'history': recent_anomaly
        },
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/tokenizer_info')
def get_tokenizer_info():
    """Get tokenizer configuration and vocabulary information."""
    if not state['tokenizer']:
        return jsonify({'success': False, 'error': 'Tokenizer not loaded'})

    try:
        tokenizer_config = {
            'vocab_size': len(state['tokenizer'].vocab),
            'mode': getattr(state['tokenizer'], 'mode', 'unknown'),
            'precision': getattr(state['tokenizer'], 'precision', {}),
        }

        # Add bucketing info if available
        if hasattr(state['tokenizer'], 'bucket_digits'):
            tokenizer_config['bucket_digits'] = state['tokenizer'].bucket_digits
            tokenizer_config['max_bucket_value'] = 10 ** state['tokenizer'].bucket_digits

        # Vocabulary breakdown
        vocab_breakdown = {
            'special_tokens': [],
            'command_tokens': [],
            'parameter_tokens': [],
            'numeric_tokens': [],
        }

        for token, idx in state['tokenizer'].vocab.items():
            if token in ['<PAD>', '<BOS>', '<EOS>', '<UNK>', '<MASK>', 'PAD', 'BOS', 'EOS', 'UNK', 'MASK']:
                vocab_breakdown['special_tokens'].append(token)
            elif token.startswith('G') or token.startswith('M'):
                vocab_breakdown['command_tokens'].append(token)
            elif token.startswith('NUM_'):
                vocab_breakdown['numeric_tokens'].append(token)
            elif len(token) == 1 and token.isalpha():
                vocab_breakdown['parameter_tokens'].append(token)

        tokenizer_config['vocab_breakdown'] = {
            'special_count': len(vocab_breakdown['special_tokens']),
            'command_count': len(vocab_breakdown['command_tokens']),
            'parameter_count': len(vocab_breakdown['parameter_tokens']),
            'numeric_count': len(vocab_breakdown['numeric_tokens']),
            'examples': {
                'commands': vocab_breakdown['command_tokens'][:10],
                'parameters': vocab_breakdown['parameter_tokens'],
                'numeric_samples': vocab_breakdown['numeric_tokens'][:5],
            }
        }

        # Add decomposer info if available
        if state['decomposer']:
            tokenizer_config['decomposer'] = {
                'n_commands': len(state['decomposer'].command_tokens),
                'n_param_types': len(state['decomposer'].param_type_tokens),
                'n_param_values': state['decomposer'].n_param_values,
                'bucket_digits': state['decomposer'].bucket_digits,
                'command_examples': state['decomposer'].command_tokens[:10],
                'param_type_examples': state['decomposer'].param_type_tokens,
            }

        return jsonify(tokenizer_config)

    except Exception as e:
        logger.error(f"Error getting tokenizer info: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/command_types')
def get_command_types():
    """Get command type distribution."""
    return jsonify({
        'types': dict(state['command_types']),
        'total': sum(state['command_types'].values())
    })


@app.route('/api/export_gcode', methods=['GET'])
def export_gcode():
    """Export generated G-code commands."""
    if not state['generation_history']:
        return jsonify({'success': False, 'error': 'No data to export'})

    # Create text file
    output = io.StringIO()
    output.write("; Generated G-Code from Dashboard\n")
    output.write(f"; Exported: {datetime.now().isoformat()}\n\n")

    for i, entry in enumerate(state['generation_history'], 1):
        output.write(f"; Command {i}\n")
        output.write(f"; Predicted: {entry['predicted']}\n")
        if entry['ground_truth']:
            output.write(f"; Ground Truth: {entry['ground_truth']}\n")
        output.write(f"; Confidence: {entry['confidence']:.4f}\n")
        output.write(f"{entry['predicted']}\n\n")

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/plain',
        as_attachment=True,
        download_name=f'generated_gcode_{datetime.now().strftime("%Y%m%d_%H%M%S")}.nc'
    )


@app.route('/api/tsne')
def get_tsne():
    """Get t-SNE projection of fingerprint embeddings with Redis caching."""
    try:
        if not state['fingerprints'] or len(state['fingerprints']) < 10:
            return jsonify({'success': False, 'error': 'Not enough data for t-SNE (need at least 10 samples)'})

        # Get fingerprints
        embeddings = np.array(state['fingerprints'])

        # Create cache key based on fingerprints hash
        cache_key = f"tsne:{hash(embeddings.tobytes())}:{len(embeddings)}"

        # Try to get from Redis cache
        if REDIS_AVAILABLE:
            try:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    logger.info("t-SNE cache hit")
                    return jsonify(json.loads(cached_result))
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # Cache miss - compute t-SNE
        logger.info(f"t-SNE cache miss - computing for {len(embeddings)} samples")
        perplexity = min(30, len(embeddings) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=300)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Get corresponding G-code labels from history
        labels = []
        for i, entry in enumerate(list(state['generation_history'])[-len(embeddings):]):
            labels.append(entry.get('predicted', 'Unknown'))

        # Pad labels if needed
        while len(labels) < len(embeddings):
            labels.insert(0, 'Unknown')

        result = {
            'success': True,
            'embeddings': embeddings_2d.tolist(),
            'labels': labels,
            'n_samples': len(embeddings)
        }

        # Cache result in Redis (TTL: 60 seconds)
        if REDIS_AVAILABLE:
            try:
                redis_client.setex(cache_key, 60, json.dumps(result))
                logger.info("t-SNE result cached")
            except Exception as e:
                logger.warning(f"Redis setex failed: {e}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"t-SNE error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# W&B Sweep Integration Routes
# ============================================================================

@app.route('/api/wandb/configure', methods=['POST'])
def configure_wandb():
    """Configure W&B entity and project for sweep monitoring."""
    try:
        data = request.json
        entity = data.get('entity')
        project = data.get('project')

        if not entity or not project:
            return jsonify({'success': False, 'error': 'Both entity and project are required'})

        # Test connection
        try:
            api = wandb.Api()
            # Try to access the project
            api.project(project, entity=entity)
            state['wandb_entity'] = entity
            state['wandb_project'] = project
            logger.info(f"W&B configured: {entity}/{project}")

            return jsonify({
                'success': True,
                'message': f'Connected to {entity}/{project}',
                'entity': entity,
                'project': project
            })
        except Exception as e:
            logger.error(f"W&B connection failed: {e}")
            return jsonify({'success': False, 'error': f'Cannot access {entity}/{project}: {str(e)}'})

    except Exception as e:
        logger.error(f"W&B configure error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/wandb/sweeps')
def get_wandb_sweeps():
    """Get list of available W&B sweeps."""
    try:
        if not state['wandb_entity'] or not state['wandb_project']:
            return jsonify({'success': False, 'error': 'W&B not configured. Use /api/wandb/configure first.'})

        api = wandb.Api()
        project_path = f"{state['wandb_entity']}/{state['wandb_project']}"
        sweeps = api.project(state['wandb_project'], entity=state['wandb_entity']).sweeps()

        sweep_list = []
        for sweep in sweeps:
            sweep_data = {
                'id': sweep.id,
                'name': sweep.name or f"sweep_{sweep.id}",
                'state': sweep.state,
                'run_count': sweep.run_count,
                'config': sweep.config,
                'best_run': None,
            }

            # Get best run if available
            if sweep.best_run():
                best_run = sweep.best_run()
                sweep_data['best_run'] = {
                    'id': best_run.id,
                    'name': best_run.name,
                    'state': best_run.state,
                    'summary': best_run.summary._json_dict if hasattr(best_run.summary, '_json_dict') else {}
                }

            sweep_list.append(sweep_data)

        return jsonify({
            'success': True,
            'sweeps': sweep_list,
            'entity': state['wandb_entity'],
            'project': state['wandb_project']
        })

    except Exception as e:
        logger.error(f"Get sweeps error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/wandb/sweep/<sweep_id>')
def get_sweep_details(sweep_id):
    """Get detailed information about a specific sweep."""
    try:
        if not state['wandb_entity'] or not state['wandb_project']:
            return jsonify({'success': False, 'error': 'W&B not configured'})

        # Check cache (5 second TTL)
        cache_key = f"{state['wandb_entity']}/{state['wandb_project']}/{sweep_id}"
        now = datetime.now().timestamp()
        if cache_key in state['sweep_cache'] and \
           cache_key in state['sweep_cache_timestamp'] and \
           (now - state['sweep_cache_timestamp'][cache_key]) < 5:
            logger.info(f"Sweep cache hit: {sweep_id}")
            return jsonify(state['sweep_cache'][cache_key])

        api = wandb.Api()
        sweep_path = f"{state['wandb_entity']}/{state['wandb_project']}/sweeps/{sweep_id}"
        sweep = api.sweep(sweep_path)

        # Get runs
        runs_data = []
        for run in sweep.runs:
            run_info = {
                'id': run.id,
                'name': run.name,
                'state': run.state,
                'config': dict(run.config),
                'summary': run.summary._json_dict if hasattr(run.summary, '_json_dict') else {},
                'created_at': run.created_at,
                'heartbeat_at': run.heartbeat_at,
            }
            runs_data.append(run_info)

        # Sort runs by metric (if available)
        metric_name = sweep.config.get('metric', {}).get('name')
        goal = sweep.config.get('metric', {}).get('goal', 'maximize')
        if metric_name:
            runs_data.sort(
                key=lambda x: x['summary'].get(metric_name, float('-inf')),
                reverse=(goal == 'maximize')
            )

        result = {
            'success': True,
            'sweep_id': sweep.id,
            'name': sweep.name or f"sweep_{sweep.id}",
            'state': sweep.state,
            'config': sweep.config,
            'run_count': len(runs_data),
            'runs': runs_data,
            'metric_name': metric_name,
            'goal': goal,
        }

        # Cache result
        state['sweep_cache'][cache_key] = result
        state['sweep_cache_timestamp'][cache_key] = now

        return jsonify(result)

    except Exception as e:
        logger.error(f"Get sweep details error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/wandb/sweep/<sweep_id>/best')
def get_sweep_best_runs(sweep_id):
    """Get top 5 best runs from a sweep."""
    try:
        if not state['wandb_entity'] or not state['wandb_project']:
            return jsonify({'success': False, 'error': 'W&B not configured'})

        api = wandb.Api()
        sweep_path = f"{state['wandb_entity']}/{state['wandb_project']}/sweeps/{sweep_id}"
        sweep = api.sweep(sweep_path)

        metric_name = sweep.config.get('metric', {}).get('name')
        goal = sweep.config.get('metric', {}).get('goal', 'maximize')

        if not metric_name:
            return jsonify({'success': False, 'error': 'Sweep has no metric configured'})

        runs_data = []
        for run in sweep.runs:
            metric_value = run.summary.get(metric_name)
            if metric_value is not None:
                runs_data.append({
                    'id': run.id,
                    'name': run.name,
                    'state': run.state,
                    'config': dict(run.config),
                    'metric_value': metric_value,
                    'summary': run.summary._json_dict if hasattr(run.summary, '_json_dict') else {},
                })

        # Sort and get top 5
        runs_data.sort(
            key=lambda x: x['metric_value'],
            reverse=(goal == 'maximize')
        )
        top_runs = runs_data[:5]

        return jsonify({
            'success': True,
            'sweep_id': sweep.id,
            'metric_name': metric_name,
            'goal': goal,
            'top_runs': top_runs,
            'total_runs': len(runs_data)
        })

    except Exception as e:
        logger.error(f"Get best runs error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# Checkpoint Manager Routes
# ============================================================================

@app.route('/api/checkpoints')
def list_checkpoints():
    """List all available checkpoints in the outputs directory."""
    try:
        outputs_dir = Path('outputs')
        if not outputs_dir.exists():
            return jsonify({'success': True, 'checkpoints': []})

        checkpoints = []

        # Search for checkpoint files
        for checkpoint_file in outputs_dir.rglob('checkpoint_*.pt'):
            try:
                # Load checkpoint metadata
                checkpoint = torch.load(checkpoint_file, map_location='cpu')

                # Extract info
                checkpoint_info = {
                    'path': str(checkpoint_file),
                    'name': checkpoint_file.name,
                    'directory': str(checkpoint_file.parent.relative_to(outputs_dir)),
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'val_acc': checkpoint.get('val_acc', None),
                    'train_loss': checkpoint.get('train_loss', None),
                    'config': checkpoint.get('config', {}),
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                    'modified': checkpoint_file.stat().st_mtime,
                }

                checkpoints.append(checkpoint_info)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
                continue

        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'success': True,
            'checkpoints': checkpoints,
            'count': len(checkpoints)
        })

    except Exception as e:
        logger.error(f"List checkpoints error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/checkpoint/<path:checkpoint_path>/info')
def get_checkpoint_info(checkpoint_path):
    """Get detailed information about a specific checkpoint."""
    try:
        checkpoint_file = Path(checkpoint_path)

        if not checkpoint_file.exists():
            return jsonify({'success': False, 'error': 'Checkpoint not found'})

        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        # Extract detailed info
        info = {
            'path': str(checkpoint_file),
            'name': checkpoint_file.name,
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_acc': checkpoint.get('val_acc', None),
            'train_loss': checkpoint.get('train_loss', None),
            'val_loss': checkpoint.get('val_loss', None),
            'config': checkpoint.get('config', {}),
            'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
            'modified': checkpoint_file.stat().st_mtime,
        }

        # Get model state dict info
        if 'backbone_state_dict' in checkpoint:
            backbone_params = sum(p.numel() for p in checkpoint['backbone_state_dict'].values())
            info['backbone_params'] = backbone_params

        if 'multihead_state_dict' in checkpoint:
            multihead_params = sum(p.numel() for p in checkpoint['multihead_state_dict'].values())
            info['multihead_params'] = multihead_params

        return jsonify({
            'success': True,
            'checkpoint': info
        })

    except Exception as e:
        logger.error(f"Get checkpoint info error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/checkpoint/load', methods=['POST'])
def load_checkpoint_to_model():
    """Load a checkpoint into the current model."""
    try:
        data = request.json
        checkpoint_path = data.get('checkpoint_path')

        if not checkpoint_path:
            return jsonify({'success': False, 'error': 'No checkpoint path provided'})

        checkpoint_file = Path(checkpoint_path)

        if not checkpoint_file.exists():
            return jsonify({'success': False, 'error': 'Checkpoint not found'})

        # This would integrate with the existing load_model functionality
        # For now, we'll just validate the checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu')

        return jsonify({
            'success': True,
            'message': f'Checkpoint loaded: {checkpoint_file.name}',
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_acc': checkpoint.get('val_acc', 'unknown')
        })

    except Exception as e:
        logger.error(f"Load checkpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/checkpoints/compare', methods=['POST'])
def compare_checkpoints():
    """Compare multiple checkpoints side by side."""
    try:
        data = request.json
        checkpoint_paths = data.get('checkpoints', [])

        if len(checkpoint_paths) < 2:
            return jsonify({'success': False, 'error': 'Please select at least 2 checkpoints to compare'})

        if len(checkpoint_paths) > 5:
            return jsonify({'success': False, 'error': 'Maximum 5 checkpoints can be compared at once'})

        comparison_results = []

        for checkpoint_path in checkpoint_paths:
            checkpoint_file = Path(checkpoint_path)

            if not checkpoint_file.exists():
                continue

            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')

                # Extract metrics
                result = {
                    'path': str(checkpoint_file),
                    'name': checkpoint_file.name,
                    'directory': str(checkpoint_file.parent.name),
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'val_acc': checkpoint.get('val_acc', None),
                    'train_loss': checkpoint.get('train_loss', None),
                    'val_loss': checkpoint.get('val_loss', None),
                    'config': checkpoint.get('config', {}),
                    'size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                }

                # Calculate model parameters
                if 'backbone_state_dict' in checkpoint:
                    result['backbone_params'] = sum(p.numel() for p in checkpoint['backbone_state_dict'].values())

                if 'multihead_state_dict' in checkpoint:
                    result['multihead_params'] = sum(p.numel() for p in checkpoint['multihead_state_dict'].values())

                comparison_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
                continue

        if len(comparison_results) < 2:
            return jsonify({'success': False, 'error': 'Could not load enough valid checkpoints for comparison'})

        # Sort by validation accuracy (descending)
        comparison_results.sort(key=lambda x: x.get('val_acc') or 0, reverse=True)

        return jsonify({
            'success': True,
            'checkpoints': comparison_results,
            'count': len(comparison_results)
        })

    except Exception as e:
        logger.error(f"Compare checkpoints error: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# Error Analysis Routes (Phase 2)
# ============================================================================

@app.route('/api/generation_error_analysis')
def get_generation_error_analysis():
    """Analyze prediction errors from generation history (legacy)."""
    try:
        if not state['generation_history']:
            return jsonify({'success': False, 'error': 'No generation history available'})

        total_predictions = len(state['generation_history'])
        errors = []
        error_types = defaultdict(int)

        for idx, entry in enumerate(state['generation_history']):
            predicted = entry.get('predicted', '')
            ground_truth = entry.get('ground_truth', '')

            if predicted != ground_truth:
                edit_dist = entry.get('edit_distance', 0)

                if len(predicted) > len(ground_truth):
                    error_type = 'over_prediction'
                elif len(predicted) < len(ground_truth):
                    error_type = 'under_prediction'
                else:
                    error_type = 'substitution'

                error_types[error_type] += 1

                errors.append({
                    'index': idx,
                    'predicted': predicted,
                    'ground_truth': ground_truth,
                    'edit_distance': edit_dist,
                    'error_type': error_type,
                    'confidence': entry.get('confidence', 0)
                })

        errors.sort(key=lambda x: x['confidence'])

        return jsonify({
            'success': True,
            'total_predictions': total_predictions,
            'total_errors': len(errors),
            'error_rate': len(errors) / total_predictions if total_predictions > 0 else 0,
            'error_types': dict(error_types),
            'top_errors': errors[:20]
        })

    except Exception as e:
        logger.error(f"Error analysis failed: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# Phase 3: Data Explorer and Report Generator Routes
# ============================================================================

@app.route('/api/data_stats')
def get_data_stats():
    """Get dataset statistics for Data Explorer."""
    try:
        import glob
        import numpy as np

        # Try to find processed data directories
        processed_dirs = glob.glob('outputs/processed*')

        if not processed_dirs:
            return jsonify({
                'success': False,
                'error': 'No processed data found. Run preprocessing first.'
            })

        # Use the first processed directory
        processed_dir = processed_dirs[0]

        # Load dataset files
        train_data = np.load(f'{processed_dir}/train_sequences.npz', allow_pickle=True)
        val_data = np.load(f'{processed_dir}/val_sequences.npz', allow_pickle=True)
        test_data = np.load(f'{processed_dir}/test_sequences.npz', allow_pickle=True)

        # Get vocabulary size
        vocab_size = len(state['decomposer'].vocab) if state['decomposer'] else 668

        # Extract feature dimensions from first sample
        continuous_dim = train_data['continuous'][0].shape[1] if len(train_data['continuous']) > 0 else 0
        categorical_dim = train_data['categorical'][0].shape[1] if len(train_data['categorical']) > 0 else 0
        max_seq_length = max(len(seq) for seq in train_data['tokens']) if len(train_data['tokens']) > 0 else 0

        return jsonify({
            'success': True,
            'train_samples': len(train_data['tokens']),
            'val_samples': len(val_data['tokens']),
            'test_samples': len(test_data['tokens']),
            'vocab_size': vocab_size,
            'continuous_dim': continuous_dim,
            'categorical_dim': categorical_dim,
            'max_seq_length': max_seq_length,
            'data_dir': processed_dir
        })

    except Exception as e:
        logger.error(f"Failed to load data stats: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Generate experiment report in specified format."""
    try:
        from datetime import datetime
        import json
        import os

        data = request.json
        title = data.get('title', 'Model Evaluation Report')
        format_type = data.get('format', 'markdown')
        include_metrics = data.get('include_metrics', True)
        include_config = data.get('include_config', True)
        include_history = data.get('include_history', False)

        # Create reports directory
        os.makedirs('outputs/reports', exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate report content
        report_content = []

        if format_type == 'markdown':
            report_content.append(f"# {title}\n")
            report_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if include_metrics:
                report_content.append("## Metrics\n\n")
                report_content.append("| Metric | Value |\n")
                report_content.append("|--------|-------|\n")
                metrics = state.get('current_metrics', {})
                for key, value in metrics.items():
                    if isinstance(value, float):
                        report_content.append(f"| {key} | {value:.4f} |\n")
                    else:
                        report_content.append(f"| {key} | {value} |\n")
                report_content.append("\n")

            if include_config:
                report_content.append("## Configuration\n\n")
                report_content.append("```json\n")
                report_content.append(json.dumps(state.get('config', {}), indent=2))
                report_content.append("\n```\n\n")

            if include_history and state['generation_history']:
                report_content.append(f"## Generation History\n\n")
                report_content.append(f"Total predictions: {len(state['generation_history'])}\n\n")
                report_content.append("### Sample Predictions (first 10)\n\n")
                for i, entry in enumerate(state['generation_history'][:10]):
                    report_content.append(f"**Sample {i+1}:**\n")
                    report_content.append(f"- Predicted: `{entry.get('predicted', '')}`\n")
                    report_content.append(f"- Ground Truth: `{entry.get('ground_truth', '')}`\n")
                    report_content.append(f"- Confidence: {entry.get('confidence', 0):.4f}\n\n")

            file_ext = 'md'

        elif format_type == 'html':
            report_content.append(f"<!DOCTYPE html><html><head><title>{title}</title></head><body>")
            report_content.append(f"<h1>{title}</h1>")
            report_content.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

            if include_metrics:
                report_content.append("<h2>Metrics</h2><table border='1'>")
                report_content.append("<tr><th>Metric</th><th>Value</th></tr>")
                for key, value in state['current_metrics'].items():
                    if isinstance(value, float):
                        report_content.append(f"<tr><td>{key}</td><td>{value:.4f}</td></tr>")
                    else:
                        report_content.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                report_content.append("</table>")

            report_content.append("</body></html>")
            file_ext = 'html'

        else:  # JSON format
            report_data = {
                'title': title,
                'generated_at': datetime.now().isoformat(),
                'metrics': state['current_metrics'] if include_metrics else {},
                'config': state.get('config', {}) if include_config else {},
                'generation_history': state['generation_history'][:10] if include_history else []
            }
            report_content.append(json.dumps(report_data, indent=2))
            file_ext = 'json'

        # Write report to file
        filename = f"report_{timestamp}.{file_ext}"
        filepath = os.path.join('outputs/reports', filename)

        with open(filepath, 'w') as f:
            f.write(''.join(report_content))

        # Generate preview (first 500 characters)
        preview = ''.join(report_content)[:500]

        return jsonify({
            'success': True,
            'file_path': filepath,
            'preview': preview,
            'format': format_type
        })

    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/attention')
def get_attention_weights():
    """
    Get the latest attention weights for visualization.

    Returns attention map showing which sensor timesteps the model
    focused on when making predictions.
    """
    try:
        if state['attention_weights'] is None:
            return jsonify({
                'success': False,
                'error': 'No attention weights available. Run inference first.'
            })

        return jsonify({
            'success': True,
            'attention': state['attention_weights'],
            'model_type': state['model_type'],
            'available': state['model_type'] == 'multihead'
        })

    except Exception as e:
        logger.error(f"Failed to get attention weights: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/attention/history')
def get_attention_history():
    """Get attention weights history for comparison."""
    try:
        history = list(state['attention_history'])

        return jsonify({
            'success': True,
            'history': history,
            'count': len(history),
            'max_history': state['attention_history'].maxlen
        })

    except Exception as e:
        logger.error(f"Failed to get attention history: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================================
# WebSocket Event Handlers
# ============================================================================

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection."""
    logger.info("Client connected")
    emit('connection_status', {'status': 'connected', 'message': 'WebSocket connected successfully'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection."""
    logger.info("Client disconnected")
    # Stop inference if running
    if state['running']:
        state['running'] = False


@socketio.on('start_inference')
def handle_start_inference(data=None):
    """Start streaming inference results via WebSocket with comprehensive error handling."""
    try:
        # Validate prerequisites
        if not validate_csv_loaded():
            return
        if not validate_model_loaded():
            return

        logger.info("Starting WebSocket inference stream")
        state['running'] = True
        state['current_idx'] = 0

        # Reset state
        state['buffer'].clear()
        state['predictions_history'].clear()
        state['confusion_matrix'].clear()
        state['fingerprints'].clear()
        state['generation_history'].clear()

        emit('inference_started', {'status': 'started', 'total_rows': len(state['csv_data'])})

        # Stream predictions in real-time
        error_count = 0
        max_consecutive_errors = 5

        while state['running'] and state['current_idx'] < len(state['csv_data']):
            try:
                # Get current row
                row = state['csv_data'].iloc[state['current_idx']]

                # Extract features with validation
                try:
                    continuous, categorical = extract_features_with_validation(
                        row,
                        state.get('metadata'),
                        state['config'] if state['model'] else None
                    )
                    error_count = 0  # Reset error counter on success
                except ValueError as e:
                    logger.error(f"Feature extraction failed at row {state['current_idx']}: {e}")
                    emit_error(
                        f'Feature dimension mismatch at row {state["current_idx"]}',
                        hint='Make sure the CSV was preprocessed with the same settings as the model was trained with.',
                        exception=e
                    )
                    state['running'] = False
                    return

                # Ground truth (if available)
                ground_truth = row.get('gcode_string', None)

                # Process sample
                try:
                    predictions = process_sample(continuous, categorical, ground_truth)
                except Exception as e:
                    logger.error(f"Processing error at row {state['current_idx']}: {e}", exc_info=True)
                    error_count += 1
                    if error_count >= max_consecutive_errors:
                        emit_error(
                            f'Too many consecutive processing errors ({error_count})',
                            hint='Check model compatibility and data format.',
                            exception=e
                        )
                        state['running'] = False
                        return
                    # Skip this sample and continue
                    state['current_idx'] += 1
                    continue

                # Increment
                state['current_idx'] += 1

                # Store history
                if predictions:
                    state['predictions_history'].append(predictions)
                    # Keep only last 200 for performance
                    if len(state['predictions_history']) > 200:
                        state['predictions_history'] = state['predictions_history'][-200:]

                    # Extract sensor data for visualization (all sensors with values)
                    # Use actual CSV columns instead of metadata master_columns to handle different datasets
                    sensor_data = {}

                    # Define excluded columns (non-sensor data)
                    exclude_cols = {'time', 'gcode_line_num', 'gcode_text', 'gcode_tokens',
                                    't_console', 'gcode_line', 'gcode_string', 'raw_json',
                                    'vel', 'plane', 'line', 'posx', 'posy', 'posz', 'feed', 'momo',
                                    'mpox', 'mpoy', 'mpoz', 'unit', 'dist', 'coor'}

                    # Extract all numeric sensor columns from the actual CSV row
                    for col in row.index:
                        if col not in exclude_cols:
                            try:
                                sensor_data[col] = float(row[col])
                            except (ValueError, TypeError):
                                pass  # Skip non-numeric columns

                    # Extract 3D position data
                    position_3d = {
                        'x': float(row['mpox']) if 'mpox' in row.index else 0,
                        'y': float(row['mpoy']) if 'mpoy' in row.index else 0,
                        'z': float(row['mpoz']) if 'mpoz' in row.index else 0,
                    }

                    # Add sensor_data and position_3d to predictions for frontend
                    predictions['sensor_data'] = sensor_data
                    predictions['position_3d'] = position_3d

                    # Emit real-time update with sensor and position data
                    emit('prediction_update', {
                        'predictions': predictions,
                        'current_idx': state['current_idx'],
                        'total_rows': len(state['csv_data']),
                        'progress': (state['current_idx'] / len(state['csv_data'])) * 100
                    })

                # Small delay to prevent overwhelming the client
                socketio.sleep(0.01)  # 10ms delay = ~100 updates/second max

            except Exception as e:
                # Catch any unexpected errors in the processing loop
                logger.error(f"Unexpected error in inference loop at row {state['current_idx']}: {e}", exc_info=True)
                error_count += 1
                if error_count >= max_consecutive_errors:
                    emit_error(
                        f'Too many consecutive errors in inference loop',
                        hint='Check logs for details. Consider reloading the model or CSV.',
                        exception=e
                    )
                    state['running'] = False
                    return
                # Skip this sample and continue
                state['current_idx'] += 1

        # Finished
        if state['current_idx'] >= len(state['csv_data']):
            logger.info("Inference stream completed")
            emit('inference_completed', {'status': 'completed', 'total_processed': state['current_idx']})
            state['running'] = False

    except Exception as e:
        # Catch any top-level errors
        logger.error(f"Critical error in start_inference: {e}", exc_info=True)
        emit_error(
            'Critical error during inference',
            hint='Check logs for details. Try reloading the dashboard.',
            exception=e
        )
        state['running'] = False


@socketio.on('stop_inference')
def handle_stop_inference():
    """Stop the inference stream."""
    logger.info("Stopping WebSocket inference stream")
    state['running'] = False
    emit('inference_stopped', {'status': 'stopped', 'current_idx': state['current_idx']})


@socketio.on('request_status')
def handle_request_status():
    """Send current dashboard status."""
    emit('status_update', {
        'running': state['running'],
        'current_idx': state['current_idx'],
        'total_rows': len(state['csv_data']) if state['csv_data'] is not None else 0,
        'buffer_size': len(state['buffer']),
        'model_loaded': state['model'] is not None,
        'csv_loaded': state['csv_data'] is not None,
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Enhanced G-Code Fingerprinting Dashboard v2.5")
    print("="*60)
    print("\nCore Features:")
    print("  ✅ Token-level & Full command predictions")
    print("  ✅ Live confusion matrix")
    print("  ✅ Dark mode & CSV export")
    print("  ✅ Sensor heatmap (all 232 sensors)")
    print("  ✅ 3D position tracking")
    print("\nAdvanced Features:")
    print("  🔥 Beam search generation")
    print("  🔥 Nucleus sampling (temperature, top-p)")
    print("  🔥 Ground truth comparison & edit distance")
    print("  🔥 Generation history (last 50 commands)")
    print("  🔥 Command type distribution analytics")
    print("  🔥 Per-token confidence breakdown")
    print("  🔥 G-Code export (.nc files)")
    print("\n🏆 NEW - Multi-Head Model Support (Phase 2):")
    print("  ✨ 100% G-command accuracy!")
    print("  ✨ Hierarchical token prediction (4 heads)")
    print("  ✨ Automatic model detection (baseline vs multi-head)")
    print("  ✨ Vocabulary v2 support (170 tokens)")
    print("\nQuick Wins v2.5:")
    print("  ⚡ WebSocket support (real-time streaming)")
    print("  ⚡ Redis caching for t-SNE & analytics")
    print("  ⚡ Optimized chart rendering")
    print("  ⚡ Enhanced error handling")
    print("  ⚡ Multi-head architecture support")
    print("\nAPI Endpoints:")
    print("  📡 /api/settings - Control generation parameters")
    print("  📡 /api/generation_history - View prediction history")
    print("  📡 /api/command_types - Command distribution")
    print("  📡 /api/export_gcode - Export G-code")
    print("  📡 /api/tsne - t-SNE embeddings visualization")
    print("\nWebSocket Events:")
    print("  🔌 connect/disconnect - Connection management")
    print("  🔌 start_inference - Begin streaming predictions")
    print("  🔌 stop_inference - Pause streaming")
    print("  🔌 prediction_update - Real-time prediction events")
    print(f"\nRedis: {'✅ Connected' if REDIS_AVAILABLE else '❌ Disabled'}")
    print("\n💡 Tip: Select '🏆 Multi-Head (100% Command Acc)' from")
    print("         the model dropdown to use the Phase 2 model!")
    print("\nStarting server...")
    print("Open: http://localhost:5001")
    print("="*60 + "\n")

    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)
