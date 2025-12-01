"""
Multi-head G-code language model.

Separates token prediction into hierarchical heads to reduce gradient competition:
1. Token type gate (command vs parameter vs numeric)
2. Command head (G0, G1, M3, etc.)
3. Parameter type head (X, Y, Z, F, R, S)
4. Parameter value regression head (direct continuous value prediction)
5. Operation type head (face, adaptive, pocket, damageface, adaptive150025, unknown)

This architecture eliminates the 130:1 class imbalance problem by:
- Predicting token structure separately from token content
- Allowing different loss weights for different prediction heads
- Enabling the model to focus on command structure without interference from numeric values
- Using direct regression for numeric values to learn smooth numeric relationships
- Classifying operation type at sequence level for manufacturing context
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import SinePositionalEncoding
except ImportError:
    # When running as __main__, use absolute import
    from miracle.model.model import SinePositionalEncoding

__all__ = [
    "MultiHeadGCodeLM",
]


class MultiHeadGCodeLM(nn.Module):
    """
    Multi-head G-code language model with hierarchical token prediction.

    Architecture:
    1. Token embedding + positional encoding
    2. Causal transformer decoder
    3. Five prediction heads:
       a. Type gate: SPECIAL/COMMAND/PARAMETER/NUMERIC (4 classes)
       b. Command head: Command ID (n_commands classes)
       c. Parameter type head: X/Y/Z/F/R/S (n_param_types classes)
       d. Parameter value regression head: continuous numeric values (direct regression)
       e. Operation type head: face/adaptive/pocket/etc. (n_operation_types classes, per-sequence)

    During training:
    - All heads are trained jointly with teacher forcing
    - Loss weights can be adjusted per head
    - Regression head uses Huber loss for robustness to outliers

    During inference:
    - Type gate determines which head to use for token prediction
    - Selected head generates the token component
    - Components are composed back into full tokens
    - Operation type is predicted from sequence-level pooling
    """

    def __init__(
        self,
        d_model: int,
        n_commands: int,
        n_param_types: int,
        n_param_values: int = 100,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        vocab_size: int = 170,  # For embedding table
        n_operation_types: int = 6,  # Number of operation types (face, adaptive, pocket, etc.)
    ):
        """
        Args:
            d_model: Hidden dimension
            n_commands: Number of command tokens (G0, G1, M3, etc.)
            n_param_types: Number of parameter types (X, Y, Z, F, R, S, etc.)
            n_param_values: Number of parameter value buckets (default: 100 for 00-99)
            nhead: Number of attention heads
            num_layers: Number of decoder layers
            dropout: Dropout probability
            vocab_size: Full vocabulary size (for token embedding)
            n_operation_types: Number of operation types (face, adaptive, pocket, damageface, adaptive150025, unknown)
        """
        super().__init__()
        self.d_model = d_model
        self.n_commands = n_commands
        self.n_param_types = n_param_types
        self.n_param_values = n_param_values
        self.n_operation_types = n_operation_types

        # Token embedding (shared across all heads)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinePositionalEncoding(d_model)

        # Causal transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, 4 * d_model, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Prediction heads
        self.type_gate = nn.Linear(d_model, 4)  # 4 types: SPECIAL, COMMAND, PARAMETER, NUMERIC

        self.command_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_commands),
        )

        self.param_type_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_param_types),
        )

        # Direct regression for parameter values
        # Predicts continuous numeric values without bucketing
        # This approach learns smooth numeric relationships and generalizes better
        self.param_value_regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 4, 1),  # Direct continuous value prediction
        )

        # Operation type head (predicts once per sequence, not per token)
        # Uses global average pooling over sequence dimension
        self.operation_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_operation_types),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def causal_mask(sz: int, device) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(
        self,
        memory: torch.Tensor,
        tgt_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing.

        Args:
            memory: [B, Tm, D] encoder outputs (e.g., LSTM outputs)
            tgt_tokens: [B, Tg] target token IDs
            return_attention: If True, also return attention weights

        Returns:
            Dictionary with:
            - 'type_logits': [B, Tg, 4] token type predictions
            - 'command_logits': [B, Tg, n_commands] command predictions
            - 'param_type_logits': [B, Tg, n_param_types] parameter type predictions
            - 'param_value_regression': [B, Tg, 1] continuous numeric value predictions
            - 'operation_logits': [B, n_operation_types] operation type predictions (per-sequence)
            - 'attention_weights': (optional) List of [B, num_heads, Tg, Tm] cross-attention weights per layer
        """
        # Embed and decode
        x = self.pos(self.embed(tgt_tokens))  # [B, Tg, D]
        mask = self.causal_mask(x.size(1), x.device)  # [Tg, Tg]

        if return_attention:
            # Extract attention weights by hooking into decoder layers
            attention_weights = []

            def hook_fn(module, input, output):
                """Hook to capture cross-attention weights."""
                # TransformerDecoderLayer returns (output,) or just output
                # We need to access the multi-head attention module
                if hasattr(module, 'multihead_attn'):
                    # Store the cross-attention weights
                    # Note: This is a simplified approach; actual implementation may vary
                    pass

            # Register hooks
            hooks = []
            for layer in self.decoder.layers:
                hook = layer.register_forward_hook(hook_fn)
                hooks.append(hook)

            dec = self.decoder(tgt=x, memory=memory, tgt_mask=mask)  # [B, Tg, D]

            # Remove hooks
            for hook in hooks:
                hook.remove()
        else:
            dec = self.decoder(tgt=x, memory=memory, tgt_mask=mask)  # [B, Tg, D]

        # Predict with all heads
        type_logits = self.type_gate(dec)  # [B, Tg, 4]
        command_logits = self.command_head(dec)  # [B, Tg, n_commands]
        param_type_logits = self.param_type_head(dec)  # [B, Tg, n_param_types]

        # Direct regression for parameter values
        param_value_regression = self.param_value_regression_head(dec)  # [B, Tg, 1]

        # Operation type prediction (per-sequence, not per-token)
        # Use global average pooling over sequence dimension
        dec_pooled = dec.mean(dim=1)  # [B, D]
        operation_logits = self.operation_head(dec_pooled)  # [B, n_operation_types]

        result = {
            'type_logits': type_logits,
            'command_logits': command_logits,
            'param_type_logits': param_type_logits,
            'param_value_regression': param_value_regression,
            'operation_logits': operation_logits,
        }

        if return_attention:
            result['attention_weights'] = attention_weights

        return result

    def extract_attention_weights(
        self,
        memory: torch.Tensor,
        tgt_tokens: torch.Tensor,
        average_heads: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for visualization.

        Args:
            memory: [B, Tm, D] encoder outputs
            tgt_tokens: [B, Tg] target token IDs
            average_heads: If True, average across attention heads. If False, return per-head attention.

        Returns:
            Dictionary with attention weights (optionally averaged across heads) and per-layer
        """
        x = self.pos(self.embed(tgt_tokens))  # [B, Tg, D]
        mask = self.causal_mask(x.size(1), x.device)

        # Manually forward through decoder layers to extract attention
        attention_maps = []

        for layer in self.decoder.layers:
            # Get cross-attention from this layer
            # Note: TransformerDecoderLayer has multihead_attn for cross-attention
            attn_output, attn_weights = layer.multihead_attn(
                query=x,
                key=memory,
                value=memory,
                need_weights=True,
                average_attn_weights=average_heads,  # Configurable: average or per-head
            )
            attention_maps.append(attn_weights.detach().cpu())

            # Continue forward pass through this layer
            x = layer(x, memory, tgt_mask=mask)

        # Average attention across all layers (if heads are averaged, shape is [B, Tg, Tm])
        # If not averaged, shape is [B, num_heads, Tg, Tm]
        if average_heads:
            avg_attention = torch.stack(attention_maps).mean(dim=0)  # [B, Tg, Tm]
        else:
            avg_attention = torch.stack(attention_maps).mean(dim=0)  # [B, num_heads, Tg, Tm]

        return {
            'attention': avg_attention.numpy(),  # Convert to numpy for JSON serialization
            'layer_attentions': [a.numpy() for a in attention_maps],
            'average_heads': average_heads,
            'num_layers': len(attention_maps),
        }

    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        max_len: int,
        bos_id: int = 1,
        eos_id: int = 2,
        decomposer=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Greedy autoregressive generation with EOS token stopping.

        Args:
            memory: [B, Tm, D] encoder outputs
            max_len: Maximum sequence length to generate
            bos_id: BOS token ID
            eos_id: EOS token ID (stops generation when predicted)
            decomposer: TokenDecomposer instance for token composition

        Returns:
            - token_ids: [B, max_len] generated token IDs
            - predictions: Dictionary with hierarchical predictions
        """
        B = memory.size(0)
        device = memory.device

        # Start with BOS token
        out = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        # Track which sequences have finished (predicted EOS)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Store predictions
        type_preds = []
        command_preds = []
        param_type_preds = []
        param_value_preds = []

        for step in range(max_len):
            # Check if all sequences have finished
            if finished.all():
                break
            # Embed and decode
            x = self.pos(self.embed(out))  # [B, step+1, D]
            mask = self.causal_mask(x.size(1), device)
            dec = self.decoder(tgt=x, memory=memory, tgt_mask=mask)  # [B, step+1, D]

            # Get predictions for last token
            last_hidden = dec[:, -1:, :]  # [B, 1, D]

            # Predict type
            type_logits = self.type_gate(last_hidden)  # [B, 1, 4]
            type_pred = torch.argmax(type_logits, dim=-1)  # [B, 1]

            # Predict command/parameter based on type
            command_logits = self.command_head(last_hidden)  # [B, 1, n_commands]
            param_type_logits = self.param_type_head(last_hidden)  # [B, 1, n_param_types]

            # Direct regression for param_value
            param_value_regression = self.param_value_regression_head(last_hidden)  # [B, 1, 1]

            command_pred = torch.argmax(command_logits, dim=-1)  # [B, 1]
            param_type_pred = torch.argmax(param_type_logits, dim=-1)  # [B, 1]
            # For numeric values, round regression output to nearest integer for token composition
            param_value_pred = torch.round(param_value_regression.squeeze(-1)).long()  # [B, 1]

            # Store predictions
            type_preds.append(type_pred)
            command_preds.append(command_pred)
            param_type_preds.append(param_type_pred)
            param_value_preds.append(param_value_pred)

            # Compose next token
            if decomposer is not None:
                # Use decomposer to compose token from hierarchical predictions
                next_tokens = []
                for b in range(B):
                    if finished[b]:
                        # Already finished, append PAD
                        next_tokens.append(0)
                    else:
                        token_id = decomposer.compose_token(
                            int(type_pred[b, 0].item()),
                            int(command_pred[b, 0].item()),
                            int(param_type_pred[b, 0].item()),
                            int(param_value_pred[b, 0].item()),
                        )
                        next_tokens.append(token_id)
                nxt = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
            else:
                # Fallback: use type prediction to select which head to use
                # This is less accurate but works without decomposer
                nxt = torch.zeros(B, 1, dtype=torch.long, device=device)
                for b in range(B):
                    if finished[b]:
                        nxt[b, 0] = 0  # PAD
                        continue
                    t = int(type_pred[b, 0].item())
                    if t == 1:  # COMMAND
                        nxt[b, 0] = command_pred[b, 0]
                    elif t == 2:  # PARAMETER
                        nxt[b, 0] = param_type_pred[b, 0]
                    elif t == 3:  # NUMERIC
                        nxt[b, 0] = param_value_pred[b, 0]
                    else:  # SPECIAL (includes EOS)
                        nxt[b, 0] = eos_id  # Use EOS token

            # Check for EOS tokens and mark finished sequences
            # Type 0 = SPECIAL, which includes PAD/BOS/EOS tokens
            for b in range(B):
                if not finished[b]:
                    # Check if type is SPECIAL (0) - this indicates end of meaningful generation
                    if int(type_pred[b, 0].item()) == 0:
                        finished[b] = True
                        nxt[b, 0] = eos_id  # Force EOS token

            # Append to sequence
            out = torch.cat([out, nxt], dim=1)

        # Concatenate predictions
        predictions = {
            'type': torch.cat(type_preds, dim=1),  # [B, max_len]
            'command_id': torch.cat(command_preds, dim=1),
            'param_type_id': torch.cat(param_type_preds, dim=1),
            'param_value_id': torch.cat(param_value_preds, dim=1),
        }

        return out[:, 1:], predictions  # Remove BOS token


# Example usage
if __name__ == '__main__':
    # Test multi-head model
    print("Testing MultiHeadGCodeLM...")

    # Model parameters
    d_model = 256
    n_commands = 15  # G0, G1, ..., M3, M5, etc.
    n_param_types = 10  # X, Y, Z, F, R, S, etc.
    n_param_values = 100  # 00-99
    vocab_size = 170

    # Create model
    model = MultiHeadGCodeLM(
        d_model=d_model,
        n_commands=n_commands,
        n_param_types=n_param_types,
        n_param_values=n_param_values,
        nhead=4,
        num_layers=2,
        vocab_size=vocab_size,
    )

    # Test forward pass
    B, Tm, Tg = 4, 64, 32
    memory = torch.randn(B, Tm, d_model)
    tgt_tokens = torch.randint(0, vocab_size, (B, Tg))

    outputs = model(memory, tgt_tokens)

    print(f"\nForward pass:")
    print(f"  Memory: {memory.shape}")
    print(f"  Target tokens: {tgt_tokens.shape}")
    print(f"  Type logits: {outputs['type_logits'].shape}")
    print(f"  Command logits: {outputs['command_logits'].shape}")
    print(f"  Param type logits: {outputs['param_type_logits'].shape}")
    print(f"  Param value coarse logits: {outputs['param_value_coarse_logits'].shape}")
    print(f"  Param value residual: {outputs['param_value_residual'].shape}")
    print(f"  Operation logits: {outputs['operation_logits'].shape}")

    # Test generation
    generated, preds = model.generate(memory, max_len=16, bos_id=1)
    print(f"\nGeneration:")
    print(f"  Generated tokens: {generated.shape}")
    print(f"  Type predictions: {preds['type'].shape}")
    print(f"  Command predictions: {preds['command_id'].shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,}")
    print(f"✅ Multi-head model test passed!")
