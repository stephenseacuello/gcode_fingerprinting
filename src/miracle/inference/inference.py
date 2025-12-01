"""
G-code Inference Module

This module provides comprehensive inference with all model outputs:
- G-code token prediction (with probabilities and sampling)
- Classification predictions
- Sensor reconstruction and error analysis
- Anomaly detection scores
- Fingerprint embeddings for machine identification

Supports multiple sampling strategies (greedy, temperature, top-p nucleus sampling).
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.dataset.dataset import GCodeDataset, collate_fn
from miracle.utilities.gcode_tokenizer import GCodeTokenizer
from miracle.model.generation import sample_with_temperature
from miracle.utilities.device import get_device, print_device_info


@dataclass
class InferencePrediction:
    """Container for all model outputs for a single sample"""

    # Original data
    original_gcode: str
    sample_idx: int

    # G-code prediction
    gcode_generated: List[str]
    gcode_generated_ids: List[int]
    gcode_top5_tokens: List[Tuple[str, float]]  # Top-5 predictions at each position
    gcode_entropy: float  # Prediction confidence (lower = more confident)

    # Classification
    classification_predicted: int
    classification_probs: List[float]
    classification_confidence: float

    # Reconstruction
    reconstruction_error: float
    reconstruction_mae: float
    reconstruction_mse: float

    # Fingerprint
    fingerprint: List[float]  # 128-D embedding
    fingerprint_norm: float

    # Anomaly detection
    anomaly_score: float
    is_anomalous: bool  # True if anomaly_score > threshold

    # Additional metrics
    sequence_length: int
    actual_length: int


class GCodeInference:
    """
    Comprehensive inference engine that extracts all outputs from the model.

    Provides complete model analysis including:
    - Token-level predictions with probabilities
    - Classification results with confidence scores
    - Reconstruction quality metrics
    - Anomaly detection scores
    - Fingerprint embeddings for machine identification
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_dir: Path,
        device: str = None,  # Auto-detects GPU if available
        anomaly_threshold: float = 0.1,
        temperature: float = 1.0,
        top_p: float = 1.0
    ):
        """
        Initialize enhanced inference engine.

        Args:
            checkpoint_path: Path to model checkpoint
            data_dir: Directory with preprocessing metadata (for tokenizer)
            device: Device to run inference on
            anomaly_threshold: Threshold for anomaly detection
            temperature: Temperature for token generation (1.0 = no change)
            top_p: Nucleus sampling threshold (1.0 = disabled)
        """
        # Auto-detect device if not specified
        self.device = get_device(device)
        self.anomaly_threshold = anomaly_threshold
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Create model
        self.config = ModelConfig(**checkpoint['config'])
        self.model = MM_DTAE_LSTM(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer from vocabulary JSON
        vocab_path = Path('data/gcode_vocab.json')
        if vocab_path.exists():
            self.tokenizer = GCodeTokenizer.load(vocab_path)
        else:
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

        print(f"✅ Model loaded: {self.model.count_params(self.model):,} parameters")
        print(f"   Vocabulary size: {len(self.tokenizer.vocab)}")
        print(f"   Anomaly threshold: {anomaly_threshold}")
        print(f"   Temperature: {temperature}")
        print(f"   Top-p: {top_p}")

    @torch.no_grad()
    def predict_all_outputs(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        lengths: torch.Tensor,
        original_gcode: str = "",
        sample_idx: int = 0
    ) -> InferencePrediction:
        """
        Run inference and extract ALL model outputs.

        Args:
            continuous: [1, T, 115] continuous features
            categorical: [1, T, 6] categorical features
            lengths: [1] sequence lengths
            original_gcode: Original G-code text (for comparison)
            sample_idx: Sample index

        Returns:
            InferencePrediction with all model outputs
        """
        # Ensure correct shapes and types
        continuous = continuous.to(self.device)
        categorical = categorical.float().to(self.device)  # CRITICAL: Convert to float
        lengths = lengths.to(self.device)

        # Forward pass through full model
        # Model expects mods as a list of [B, T, D] tensors
        mods = [continuous, categorical]
        output = self.model(mods=mods, lengths=lengths, gcode_in=None)

        # Extract components
        memory = output['memory']  # LSTM outputs [B, T, d_model]
        dtae_output = output['recon']  # Reconstruction [B, T, d_model]

        # 1. G-CODE PREDICTION
        gcode_data = self._predict_gcode(memory, lengths)

        # 2. CLASSIFICATION (already computed by model)
        cls_logits = output['cls']  # [B, num_classes]
        cls_probs = torch.softmax(cls_logits, dim=-1)
        cls_predicted = torch.argmax(cls_probs, dim=-1)
        cls_data = {
            'predicted': cls_predicted.item(),
            'probs': cls_probs[0].cpu().tolist(),
            'confidence': cls_probs[0, cls_predicted].item()
        }

        # 3. RECONSTRUCTION (already computed by model)
        recon_data = self._compute_reconstruction(dtae_output, continuous, lengths)

        # 4. FINGERPRINT (already computed by model)
        fingerprint = output['fingerprint']  # [B, T, 128]
        # Average over time for a single embedding
        fp_pooled = fingerprint.mean(dim=1)  # [B, 128]
        fp_norm = fp_pooled.norm(dim=-1).item()
        fp_data = {
            'embedding': fp_pooled[0].cpu().tolist(),
            'norm': fp_norm
        }

        # 5. ANOMALY DETECTION (use anomaly head from model)
        anom_logits = output['anom']  # [B, 1] - single logit
        anomaly_score = torch.sigmoid(anom_logits[0, 0]).item()  # Convert to probability
        is_anomalous = anomaly_score > self.anomaly_threshold

        # Package into dataclass
        return InferencePrediction(
            original_gcode=original_gcode,
            sample_idx=sample_idx,
            gcode_generated=gcode_data['tokens'],
            gcode_generated_ids=gcode_data['token_ids'],
            gcode_top5_tokens=gcode_data['top5'],
            gcode_entropy=gcode_data['entropy'],
            classification_predicted=cls_data['predicted'],
            classification_probs=cls_data['probs'],
            classification_confidence=cls_data['confidence'],
            reconstruction_error=recon_data['mae'],
            reconstruction_mae=recon_data['mae'],
            reconstruction_mse=recon_data['mse'],
            fingerprint=fp_data['embedding'],
            fingerprint_norm=fp_data['norm'],
            anomaly_score=anomaly_score,
            is_anomalous=is_anomalous,
            sequence_length=continuous.size(1),
            actual_length=lengths.item()
        )

    def _predict_gcode(self, memory: torch.Tensor, lengths: torch.Tensor) -> Dict:
        """Generate G-code tokens with temperature and nucleus sampling"""
        max_len = 20
        bos_id = self.tokenizer.vocab.get('<SOS>', 1)
        eos_id = self.tokenizer.vocab.get('<EOS>', 2)

        # Create reverse mapping for decoding
        id2token = {v: k for k, v in self.tokenizer.vocab.items()}

        batch_size = memory.size(0)
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)

        entropies = []
        all_top5 = []

        for step in range(max_len):
            # Get current sequence embeddings
            tgt = self.model.gcode_head.pos(self.model.gcode_head.embed(generated))

            # Create causal mask
            mask = self.model.gcode_head.causal_mask(tgt.size(1), tgt.device)

            # Decode
            dec = self.model.gcode_head.decoder(tgt=tgt, memory=memory, tgt_mask=mask)

            # Project to vocabulary
            logits = self.model.gcode_head.proj(dec[:, -1, :])  # [B, vocab_size]

            # Apply temperature
            logits = logits / self.temperature

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Calculate entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropies.append(entropy[0].item())

            # Get top-5 predictions
            top5_probs, top5_ids = torch.topk(probs[0], k=min(5, probs.size(-1)))
            top5_tokens = [
                (id2token.get(idx.item(), '<UNK>'), prob.item())
                for idx, prob in zip(top5_ids, top5_probs)
            ]
            all_top5.append(top5_tokens)

            # Sample next token with temperature and nucleus sampling
            next_token = sample_with_temperature(
                logits[0],
                temperature=self.temperature,
                top_p=self.top_p
            )  # [1]

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Stop if EOS generated
            if next_token.item() == eos_id:
                break

        # Decode tokens
        token_ids = generated[0].cpu().tolist()[1:]  # Skip <SOS>
        tokens = [id2token.get(tid, '<UNK>') for tid in token_ids]

        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']]

        avg_entropy = np.mean(entropies) if entropies else 0.0

        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'top5': all_top5,
            'entropy': avg_entropy
        }

    def _predict_classification(self, memory: torch.Tensor, lengths: torch.Tensor) -> Dict:
        """Extract classification predictions"""
        # Pool memory over time dimension
        pooled = memory.mean(dim=1)  # [B, d_model]

        # Classification head
        logits = self.model.classification_head(pooled)  # [B, num_classes]
        probs = F.softmax(logits, dim=-1)

        predicted = torch.argmax(probs, dim=-1)
        confidence = probs[0, predicted].item()

        return {
            'predicted': predicted.item(),
            'probs': probs[0].cpu().tolist(),
            'confidence': confidence
        }

    def _compute_reconstruction(
        self,
        dtae_output: torch.Tensor,
        continuous: torch.Tensor,
        lengths: torch.Tensor
    ) -> Dict:
        """Compute reconstruction error"""
        # DTAE output is [B, T, d_model=64]
        # Need to project to [B, T, 115] for comparison

        # Use a linear projection (in real implementation, this should be a proper head)
        # For now, compute error on the hidden states directly
        seq_len = lengths[0].item()

        # Simple MSE and MAE
        # Note: This is simplified - in production you'd want a proper reconstruction head
        # For now, compare magnitudes
        target_norm = continuous[0, :seq_len].norm(dim=-1).mean()
        output_norm = dtae_output[0, :seq_len].norm(dim=-1).mean()

        mae = abs(target_norm - output_norm).item()
        mse = ((target_norm - output_norm) ** 2).item()

        return {
            'mae': mae,
            'mse': mse
        }

    def _extract_fingerprint(self, memory: torch.Tensor, lengths: torch.Tensor) -> Dict:
        """Extract fingerprint embedding"""
        # Pool memory
        pooled = memory.mean(dim=1)  # [B, d_model]

        # Fingerprint head
        fingerprint = self.model.fingerprint_head(pooled)  # [B, 128]

        # Normalize
        fp_norm = fingerprint.norm(dim=-1).item()

        return {
            'embedding': fingerprint[0].cpu().tolist(),
            'norm': fp_norm
        }

    def predict_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ) -> List[InferencePrediction]:
        """
        Run enhanced inference on a batch of samples.

        Args:
            dataloader: DataLoader with test samples
            max_samples: Maximum number of samples to process

        Returns:
            List of InferencePrediction objects
        """
        predictions = []
        sample_idx = 0

        print(f"\n🔍 Running enhanced inference...")
        print(f"   Device: {self.device}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top-p: {self.top_p}")
        print()

        for batch in dataloader:
            continuous = batch['continuous']
            categorical = batch['categorical']
            lengths = batch['lengths']
            tokens = batch['tokens']

            batch_size = continuous.size(0)

            for i in range(batch_size):
                if max_samples and sample_idx >= max_samples:
                    return predictions

                # Decode original G-code
                original_ids = tokens[i].cpu().tolist()
                id2token = {v: k for k, v in self.tokenizer.vocab.items()}
                original_tokens = [id2token.get(tid, '<UNK>') for tid in original_ids]
                original_tokens = [t for t in original_tokens if t not in ['<PAD>', '<SOS>', '<EOS>']]
                original_gcode = ' '.join(original_tokens)

                # Get prediction
                pred = self.predict_all_outputs(
                    continuous=continuous[i:i+1],
                    categorical=categorical[i:i+1],
                    lengths=lengths[i:i+1],
                    original_gcode=original_gcode,
                    sample_idx=sample_idx
                )

                predictions.append(pred)
                sample_idx += 1

                if sample_idx % 10 == 0:
                    print(f"   Processed {sample_idx} samples...")

        print(f"✅ Processed {len(predictions)} samples\n")
        return predictions


def print_prediction(pred: InferencePrediction, show_fingerprint: bool = False):
    """Pretty-print an enhanced prediction"""
    print(f"\n{'='*80}")
    print(f"SAMPLE {pred.sample_idx}")
    print(f"{'='*80}")

    print(f"\n📝 ORIGINAL G-CODE:")
    print(f"   {pred.original_gcode}")

    print(f"\n🤖 GENERATED G-CODE:")
    generated_str = ' '.join(pred.gcode_generated) if pred.gcode_generated else "(empty)"
    print(f"   {generated_str}")
    print(f"   Entropy: {pred.gcode_entropy:.3f} (lower = more confident)")

    print(f"\n🎯 CLASSIFICATION:")
    print(f"   Predicted Class: {pred.classification_predicted}")
    print(f"   Confidence: {pred.classification_confidence:.1%}")
    print(f"   All Probabilities: {[f'{p:.3f}' for p in pred.classification_probs]}")

    print(f"\n🔧 RECONSTRUCTION:")
    print(f"   MAE: {pred.reconstruction_mae:.4f}")
    print(f"   MSE: {pred.reconstruction_mse:.4f}")

    print(f"\n⚠️  ANOMALY DETECTION:")
    status = "🚨 ANOMALOUS" if pred.is_anomalous else "✅ NORMAL"
    print(f"   Status: {status}")
    print(f"   Score: {pred.anomaly_score:.4f}")

    print(f"\n🔍 FINGERPRINT:")
    print(f"   Embedding Dimension: 128")
    print(f"   L2 Norm: {pred.fingerprint_norm:.3f}")
    if show_fingerprint:
        print(f"   Values (first 10): {[f'{x:.3f}' for x in pred.fingerprint[:10]]}")

    print(f"\n📊 METADATA:")
    print(f"   Sequence Length: {pred.sequence_length}")
    print(f"   Actual Length: {pred.actual_length}")


def save_results(predictions: List[InferencePrediction], output_path: Path):
    """Save enhanced predictions to JSON"""
    results = {
        'num_samples': len(predictions),
        'predictions': [asdict(p) for p in predictions],
        'summary': {
            'avg_entropy': np.mean([p.gcode_entropy for p in predictions]),
            'avg_reconstruction_error': np.mean([p.reconstruction_mae for p in predictions]),
            'avg_anomaly_score': np.mean([p.anomaly_score for p in predictions]),
            'num_anomalies': sum([p.is_anomalous for p in predictions]),
            'classification_distribution': {}
        }
    }

    # Count classification distribution
    for p in predictions:
        cls = p.classification_predicted
        results['summary']['classification_distribution'][cls] = \
            results['summary']['classification_distribution'].get(cls, 0) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='G-code Comprehensive Inference Engine')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory with preprocessed data')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detects GPU if available, or specify: cpu/cuda/mps)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=10, help='Max samples to show')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=1.0, help='Nucleus sampling threshold')
    parser.add_argument('--anomaly-threshold', type=float, default=0.1, help='Anomaly detection threshold')

    args = parser.parse_args()

    # Create inference engine
    inference = GCodeInference(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        anomaly_threshold=args.anomaly_threshold,
        temperature=args.temperature,
        top_p=args.top_p
    )

    # Load test data
    test_path = args.data_dir / 'test_sequences.npz'
    test_dataset = GCodeDataset(test_path)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"\n📊 Test Dataset: {len(test_dataset)} samples")

    # Run inference
    predictions = inference.predict_batch(test_loader, max_samples=args.max_samples)

    # Print first few predictions
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)

    for pred in predictions[:5]:  # Show first 5
        print_prediction(pred, show_fingerprint=False)

    # Save all results
    output_path = args.output_dir / 'inference_results.json'
    save_results(predictions, output_path)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\n📊 Overall Statistics (n={len(predictions)}):")
    print(f"   Avg Entropy: {np.mean([p.gcode_entropy for p in predictions]):.3f}")
    print(f"   Avg Reconstruction Error: {np.mean([p.reconstruction_mae for p in predictions]):.4f}")
    print(f"   Avg Anomaly Score: {np.mean([p.anomaly_score for p in predictions]):.4f}")
    print(f"   Anomalies Detected: {sum([p.is_anomalous for p in predictions])} ({sum([p.is_anomalous for p in predictions])/len(predictions)*100:.1f}%)")

    print(f"\n🎯 Classification Distribution:")
    cls_dist = {}
    for p in predictions:
        cls_dist[p.classification_predicted] = cls_dist.get(p.classification_predicted, 0) + 1
    for cls, count in sorted(cls_dist.items()):
        print(f"   Class {cls}: {count} samples ({count/len(predictions)*100:.1f}%)")

    print(f"\n✅ Complete! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
