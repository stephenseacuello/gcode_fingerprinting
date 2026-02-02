#!/usr/bin/env python3
"""
Comprehensive Nested Ablation Study for G-code Fingerprinting.

Runs 2,394 (full) or 492 (conference) training runs covering:
- Per-sensor modality ablations (grouped + individual)
- Global baselines (all-sensors, machine-only, sensor-only)
- Sensor-level leave-one-out
- Cross-sensor modality ablations
- Machine feature leakage isolation
- All with 3 seeds for ANOVA statistical testing

Four phases:
  manifest  - Generate run configs (no GPU)
  validate  - Dry-run mask verification (no GPU)
  train     - Execute training runs (GPU)
  analyze   - ANOVA + importance rankings (no GPU)

Usage:
    # Generate and validate
    python scripts/experiments/run_comprehensive_ablation.py \
        --phase manifest --mode conference \
        --config configs/decoder_config.json \
        --data-dir outputs/jan30/encoder_pipeline/data \
        --output-dir outputs/comprehensive_ablation

    # Train on Bizon
    python scripts/experiments/run_comprehensive_ablation.py \
        --phase train \
        --config configs/decoder_config.json \
        --data-dir outputs/jan30/encoder_pipeline/data \
        --encoder-path outputs/jan30/encoder_pipeline/encoder_checkpoint/best_model.pt \
        --vocab-path data/vocabulary_4digit_full.json \
        --output-dir outputs/comprehensive_ablation \
        --num-workers 32 --no-save-weights

    # Analyze
    python scripts/experiments/run_comprehensive_ablation.py \
        --phase analyze \
        --data-dir outputs/jan30/encoder_pipeline/data \
        --output-dir outputs/comprehensive_ablation

Author: Claude Code
Date: January 2026
"""

import os
import sys
import json
import argparse
import platform
import time
import traceback
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from miracle.model.model import MM_DTAE_LSTM, ModelConfig
from miracle.model.sensor_multihead_decoder import SensorMultiHeadDecoder
from miracle.dataset.decoder_dataset import DecoderDatasetFromSplits, decoder_collate_fn
from miracle.training.metrics import compute_comprehensive_metrics

# =============================================================================
# CONSTANTS (verified against metadata.json)
# =============================================================================

OPERATIONAL_SENSORS = [
    'xa_motor', 'frame_r1', 'frame_r2', 'frame_b2',
    'frame_l3', 'frame_l2', 'spindle1', 'spindle2',
    'y_bed__1', 'y_bed__2', 'y_bed__3', 'y_bed__4',
]

NON_FUNCTIONAL_SENSORS = ['frame_b1', 'frame_l1', 'z_gant_1', 'z_gant_2']

ALL_SENSORS = OPERATIONAL_SENSORS + NON_FUNCTIONAL_SENSORS

# Top 6 for conference mode (covers all physical locations + performance range)
CONFERENCE_SENSORS = [
    'xa_motor',   # rank 1, motor-mounted
    'frame_r1',   # rank 2, frame-mounted (best)
    'y_bed__1',   # rank 2 tie, bed-mounted
    'frame_r2',   # rank 9, frame-mounted (worst -- contrast)
    'spindle1',   # rank 7, spindle-mounted
    'frame_l3',   # rank 4, frame-mounted (mid)
]

LEAKAGE_TOP3_SENSORS = ['xa_motor', 'frame_r1', 'y_bed__1']

# 17 sensor modalities per sensor (sensor_id.modality format)
SENSOR_MODALITIES = [
    'Ax', 'Ay', 'Az',
    'Gx', 'Gy', 'Gz',
    'Mx', 'My', 'Mz',
    'Pressure', 'Temperature', 'Proximity',
    'ColorR', 'ColorG', 'ColorB', 'ColorA',
    'RMS',
]

# 6 sensor modality groups
SENSOR_MODALITY_GROUPS = {
    'accelerometer': ['Ax', 'Ay', 'Az'],
    'gyroscope': ['Gx', 'Gy', 'Gz'],
    'magnetometer': ['Mx', 'My', 'Mz'],
    'environmental': ['Pressure', 'Temperature', 'Proximity'],
    'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
    'rms': ['RMS'],
}

# 24 machine features in 4 groups (exact column names from metadata.json)
MACHINE_GROUPS = {
    'motor_electrical': [
        'spindle', 'spindle_A',
        'x_motor', 'x_motor_A',
        'y_motor', 'y_motor_A',
        'z_motor', 'z_motor_A',
    ],
    'machine_positions': ['mpox', 'mpoy', 'mpoz'],
    'controller_state': ['coor', 'dist', 'gcode_line', 'stat', 'unit'],
    'leakage_risk': ['posx', 'posy', 'posz', 'feed', 'momo', 'line', 'vel', 'plane'],
}

# Legitimate machine groups (excluding leakage risk)
LEGITIMATE_MACHINE_GROUPS = ['motor_electrical', 'machine_positions', 'controller_state']
ALL_MACHINE_GROUPS = list(MACHINE_GROUPS.keys())

# Combined groups for ablation: 6 sensor + 3 legitimate machine = 9
ABLATION_GROUPS = {**SENSOR_MODALITY_GROUPS}
for g in LEGITIMATE_MACHINE_GROUPS:
    ABLATION_GROUPS[g] = MACHINE_GROUPS[g]

# Individual modalities for ablation: 17 sensor + 3 machine = 20
INDIVIDUAL_MODALITIES = SENSOR_MODALITIES + LEGITIMATE_MACHINE_GROUPS

DEFAULT_SEEDS = [42, 123, 456]


# =============================================================================
# MASK BUILDER
# =============================================================================

def get_machine_group(col: str) -> Optional[str]:
    """Return which machine group a non-sensor column belongs to."""
    for group_name, features in MACHINE_GROUPS.items():
        if col in features:
            return group_name
    return None


def build_mask(
    master_columns: List[str],
    sensors: Optional[List[str]] = None,
    modalities: Optional[List[str]] = None,
    machine_groups: Optional[List[str]] = None,
    exclude_sensors: Optional[List[str]] = None,
    exclude_modalities: Optional[List[str]] = None,
    exclude_machine_groups: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Build a binary feature mask.

    Inclusion (AND logic -- all provided filters must match):
        sensors: which sensors to include (None = all 16)
        modalities: which modalities to include (None = all 17)
        machine_groups: which machine groups to include (None = all 4, [] = none)

    Exclusion (override inclusions):
        exclude_sensors, exclude_modalities, exclude_machine_groups
    """
    mask = np.zeros(len(master_columns), dtype=np.float32)

    for i, col in enumerate(master_columns):
        if '.' in col:
            # Sensor column: sensor_id.modality
            sensor_id, modality = col.split('.', 1)

            # Inclusion checks
            sensor_ok = (sensors is None or sensor_id in sensors)
            mod_ok = (modalities is None or modality in modalities)

            # Exclusion checks
            excl_sensor = (exclude_sensors is not None and sensor_id in exclude_sensors)
            excl_mod = (exclude_modalities is not None and modality in exclude_modalities)

            if sensor_ok and mod_ok and not excl_sensor and not excl_mod:
                mask[i] = 1.0
        else:
            # Machine feature column
            group = get_machine_group(col)
            if group is None:
                continue  # Unknown column, skip

            # Inclusion
            grp_ok = (machine_groups is None or group in machine_groups)

            # Exclusion
            excl_grp = (exclude_machine_groups is not None and group in exclude_machine_groups)

            if grp_ok and not excl_grp:
                mask[i] = 1.0

    return mask


def build_mask_from_spec(master_columns: List[str], spec: Dict) -> np.ndarray:
    """Build mask from a manifest mask_spec dict."""
    return build_mask(
        master_columns,
        sensors=spec.get('sensors'),
        modalities=spec.get('modalities'),
        machine_groups=spec.get('machine_groups'),
        exclude_sensors=spec.get('exclude_sensors'),
        exclude_modalities=spec.get('exclude_modalities'),
        exclude_machine_groups=spec.get('exclude_machine_groups'),
    )


# =============================================================================
# MANIFEST GENERATION
# =============================================================================

def _count_active(master_columns, spec):
    """Helper to compute expected active features for a mask spec."""
    mask = build_mask_from_spec(master_columns, spec)
    return int(mask.sum())


def generate_manifest(
    mode: str,
    master_columns: List[str],
    seeds: List[int],
    components: List[str],
) -> List[Dict]:
    """Generate all run configs for the study."""
    manifest = []

    per_sensor_list = OPERATIONAL_SENSORS if mode == 'full' else CONFERENCE_SENSORS
    all_machine = ALL_MACHINE_GROUPS
    legit_machine = LEGITIMATE_MACHINE_GROUPS

    for seed in seeds:

        # =====================================================================
        # Component A: Per-sensor nested modality ablation
        # =====================================================================
        if 'A' in components:
            for sensor in per_sensor_list:
                # Baseline: this sensor + all machine (including leakage)
                spec = {'sensors': [sensor], 'machine_groups': all_machine}
                manifest.append({
                    'run_id': f'A_{sensor}_baseline_seed{seed}',
                    'component': 'A',
                    'sensor': sensor,
                    'ablation_type': 'baseline',
                    'target': 'all',
                    'seed': seed,
                    'mask_spec': spec,
                    'expected_active_features': _count_active(master_columns, spec),
                    'tags': ['component_A', 'per_sensor', sensor, 'baseline'],
                })

                # Grouped LOO: remove one group at a time
                for group_name, group_modalities in ABLATION_GROUPS.items():
                    if group_name in SENSOR_MODALITY_GROUPS:
                        spec = {
                            'sensors': [sensor],
                            'exclude_modalities': group_modalities,
                            'machine_groups': all_machine,
                        }
                    else:
                        # Machine group LOO
                        excl = [group_name]
                        spec = {
                            'sensors': [sensor],
                            'machine_groups': [g for g in all_machine if g not in excl],
                        }
                    manifest.append({
                        'run_id': f'A_{sensor}_grouped_loo_{group_name}_seed{seed}',
                        'component': 'A',
                        'sensor': sensor,
                        'ablation_type': 'grouped_loo',
                        'target': group_name,
                        'seed': seed,
                        'mask_spec': spec,
                        'expected_active_features': _count_active(master_columns, spec),
                        'tags': ['component_A', 'per_sensor', sensor, 'grouped_loo', group_name],
                    })

                # Grouped LOI: keep only one group
                for group_name, group_modalities in ABLATION_GROUPS.items():
                    if group_name in SENSOR_MODALITY_GROUPS:
                        spec = {
                            'sensors': [sensor],
                            'modalities': group_modalities,
                            'machine_groups': all_machine,
                        }
                    else:
                        # Machine group LOI: only this machine group + sensor
                        spec = {
                            'sensors': [sensor],
                            'machine_groups': [group_name],
                        }
                    manifest.append({
                        'run_id': f'A_{sensor}_grouped_loi_{group_name}_seed{seed}',
                        'component': 'A',
                        'sensor': sensor,
                        'ablation_type': 'grouped_loi',
                        'target': group_name,
                        'seed': seed,
                        'mask_spec': spec,
                        'expected_active_features': _count_active(master_columns, spec),
                        'tags': ['component_A', 'per_sensor', sensor, 'grouped_loi', group_name],
                    })

                # Individual LOO/LOI (full mode only)
                if mode == 'full':
                    for mod in INDIVIDUAL_MODALITIES:
                        if mod in SENSOR_MODALITIES:
                            # Individual sensor modality LOO
                            spec_loo = {
                                'sensors': [sensor],
                                'exclude_modalities': [mod],
                                'machine_groups': all_machine,
                            }
                            # Individual sensor modality LOI
                            spec_loi = {
                                'sensors': [sensor],
                                'modalities': [mod],
                                'machine_groups': all_machine,
                            }
                        else:
                            # Machine group as individual LOO
                            spec_loo = {
                                'sensors': [sensor],
                                'machine_groups': [g for g in all_machine if g != mod],
                            }
                            # Machine group as individual LOI
                            spec_loi = {
                                'sensors': [sensor],
                                'machine_groups': [mod],
                            }

                        manifest.append({
                            'run_id': f'A_{sensor}_individual_loo_{mod}_seed{seed}',
                            'component': 'A',
                            'sensor': sensor,
                            'ablation_type': 'individual_loo',
                            'target': mod,
                            'seed': seed,
                            'mask_spec': spec_loo,
                            'expected_active_features': _count_active(master_columns, spec_loo),
                            'tags': ['component_A', 'per_sensor', sensor, 'individual_loo', mod],
                        })
                        manifest.append({
                            'run_id': f'A_{sensor}_individual_loi_{mod}_seed{seed}',
                            'component': 'A',
                            'sensor': sensor,
                            'ablation_type': 'individual_loi',
                            'target': mod,
                            'seed': seed,
                            'mask_spec': spec_loi,
                            'expected_active_features': _count_active(master_columns, spec_loi),
                            'tags': ['component_A', 'per_sensor', sensor, 'individual_loi', mod],
                        })

        # =====================================================================
        # Component B: Global baselines
        # =====================================================================
        if 'B' in components:
            # All 296 features
            spec = {}  # None = include everything
            manifest.append({
                'run_id': f'B_all_features_seed{seed}',
                'component': 'B',
                'sensor': 'all',
                'ablation_type': 'baseline',
                'target': 'all_features',
                'seed': seed,
                'mask_spec': spec,
                'expected_active_features': _count_active(master_columns, spec),
                'tags': ['component_B', 'global', 'baseline', 'all_features'],
            })

            # Machine features only (no sensors)
            spec = {'sensors': [], 'machine_groups': all_machine}
            manifest.append({
                'run_id': f'B_machine_only_seed{seed}',
                'component': 'B',
                'sensor': 'none',
                'ablation_type': 'baseline',
                'target': 'machine_only',
                'seed': seed,
                'mask_spec': spec,
                'expected_active_features': _count_active(master_columns, spec),
                'tags': ['component_B', 'global', 'baseline', 'machine_only'],
            })

            # Per-sensor: sensor-only (no machine features)
            for sensor in CONFERENCE_SENSORS:
                spec = {'sensors': [sensor], 'machine_groups': []}
                manifest.append({
                    'run_id': f'B_{sensor}_sensor_only_seed{seed}',
                    'component': 'B',
                    'sensor': sensor,
                    'ablation_type': 'baseline',
                    'target': 'sensor_only',
                    'seed': seed,
                    'mask_spec': spec,
                    'expected_active_features': _count_active(master_columns, spec),
                    'tags': ['component_B', 'global', sensor, 'sensor_only'],
                })

            # Per-sensor: sensor + legitimate machine only
            for sensor in CONFERENCE_SENSORS:
                spec = {'sensors': [sensor], 'machine_groups': legit_machine}
                manifest.append({
                    'run_id': f'B_{sensor}_legit_machine_seed{seed}',
                    'component': 'B',
                    'sensor': sensor,
                    'ablation_type': 'baseline',
                    'target': 'sensor_legit_machine',
                    'seed': seed,
                    'mask_spec': spec,
                    'expected_active_features': _count_active(master_columns, spec),
                    'tags': ['component_B', 'global', sensor, 'legit_machine'],
                })

        # =====================================================================
        # Component C: Sensor-level leave-one-out (all 12 sensors)
        # =====================================================================
        if 'C' in components:
            for sensor in OPERATIONAL_SENSORS:
                spec = {'exclude_sensors': [sensor]}
                manifest.append({
                    'run_id': f'C_without_{sensor}_seed{seed}',
                    'component': 'C',
                    'sensor': sensor,
                    'ablation_type': 'sensor_loo',
                    'target': sensor,
                    'seed': seed,
                    'mask_spec': spec,
                    'expected_active_features': _count_active(master_columns, spec),
                    'tags': ['component_C', 'sensor_loo', sensor],
                })

        # =====================================================================
        # Component D: Cross-sensor grouped modality ablation
        # =====================================================================
        if 'D' in components:
            for group_name, group_modalities in ABLATION_GROUPS.items():
                if group_name in SENSOR_MODALITY_GROUPS:
                    # Remove modality from ALL sensors
                    spec_loo = {'exclude_modalities': group_modalities}
                    # Keep only this modality from all sensors + all machine
                    spec_loi = {'modalities': group_modalities}
                else:
                    # Machine group LOO
                    spec_loo = {
                        'machine_groups': [g for g in all_machine if g != group_name],
                    }
                    # Machine group LOI
                    spec_loi = {
                        'sensors': [],
                        'machine_groups': [group_name],
                    }

                manifest.append({
                    'run_id': f'D_cross_grouped_loo_{group_name}_seed{seed}',
                    'component': 'D',
                    'sensor': 'cross',
                    'ablation_type': 'cross_grouped_loo',
                    'target': group_name,
                    'seed': seed,
                    'mask_spec': spec_loo,
                    'expected_active_features': _count_active(master_columns, spec_loo),
                    'tags': ['component_D', 'cross_sensor', 'grouped_loo', group_name],
                })
                manifest.append({
                    'run_id': f'D_cross_grouped_loi_{group_name}_seed{seed}',
                    'component': 'D',
                    'sensor': 'cross',
                    'ablation_type': 'cross_grouped_loi',
                    'target': group_name,
                    'seed': seed,
                    'mask_spec': spec_loi,
                    'expected_active_features': _count_active(master_columns, spec_loi),
                    'tags': ['component_D', 'cross_sensor', 'grouped_loi', group_name],
                })

        # =====================================================================
        # Component E: Cross-sensor individual modality ablation (full only)
        # =====================================================================
        if 'E' in components and mode == 'full':
            for mod in INDIVIDUAL_MODALITIES:
                if mod in SENSOR_MODALITIES:
                    spec_loo = {'exclude_modalities': [mod]}
                    spec_loi = {'modalities': [mod]}
                else:
                    spec_loo = {
                        'machine_groups': [g for g in all_machine if g != mod],
                    }
                    spec_loi = {
                        'sensors': [],
                        'machine_groups': [mod],
                    }

                manifest.append({
                    'run_id': f'E_cross_individual_loo_{mod}_seed{seed}',
                    'component': 'E',
                    'sensor': 'cross',
                    'ablation_type': 'cross_individual_loo',
                    'target': mod,
                    'seed': seed,
                    'mask_spec': spec_loo,
                    'expected_active_features': _count_active(master_columns, spec_loo),
                    'tags': ['component_E', 'cross_sensor', 'individual_loo', mod],
                })
                manifest.append({
                    'run_id': f'E_cross_individual_loi_{mod}_seed{seed}',
                    'component': 'E',
                    'sensor': 'cross',
                    'ablation_type': 'cross_individual_loi',
                    'target': mod,
                    'seed': seed,
                    'mask_spec': spec_loi,
                    'expected_active_features': _count_active(master_columns, spec_loi),
                    'tags': ['component_E', 'cross_sensor', 'individual_loi', mod],
                })

        # =====================================================================
        # Component F: Leakage isolation
        # =====================================================================
        if 'F' in components:
            for sensor in LEAKAGE_TOP3_SENSORS:
                # sensor + all machine (may overlap with A baseline)
                spec_all = {'sensors': [sensor], 'machine_groups': all_machine}
                manifest.append({
                    'run_id': f'F_{sensor}_all_machine_seed{seed}',
                    'component': 'F',
                    'sensor': sensor,
                    'ablation_type': 'leakage',
                    'target': 'all_machine',
                    'seed': seed,
                    'mask_spec': spec_all,
                    'expected_active_features': _count_active(master_columns, spec_all),
                    'tags': ['component_F', 'leakage', sensor, 'all_machine'],
                })

                # sensor + legitimate machine only (no leakage_risk)
                spec_legit = {'sensors': [sensor], 'machine_groups': legit_machine}
                manifest.append({
                    'run_id': f'F_{sensor}_legit_machine_seed{seed}',
                    'component': 'F',
                    'sensor': sensor,
                    'ablation_type': 'leakage',
                    'target': 'legit_machine',
                    'seed': seed,
                    'mask_spec': spec_legit,
                    'expected_active_features': _count_active(master_columns, spec_legit),
                    'tags': ['component_F', 'leakage', sensor, 'legit_machine'],
                })

                # sensor only (no machine at all)
                spec_none = {'sensors': [sensor], 'machine_groups': []}
                manifest.append({
                    'run_id': f'F_{sensor}_no_machine_seed{seed}',
                    'component': 'F',
                    'sensor': sensor,
                    'ablation_type': 'leakage',
                    'target': 'no_machine',
                    'seed': seed,
                    'mask_spec': spec_none,
                    'expected_active_features': _count_active(master_columns, spec_none),
                    'tags': ['component_F', 'leakage', sensor, 'no_machine'],
                })

    return manifest


# =============================================================================
# VALIDATION
# =============================================================================

def validate_manifest(manifest: List[Dict], master_columns: List[str]):
    """Validate all masks in the manifest."""
    errors = []
    component_counts = {}

    for run in manifest:
        mask = build_mask_from_spec(master_columns, run['mask_spec'])
        active = int(mask.sum())
        expected = run['expected_active_features']

        comp = run['component']
        component_counts[comp] = component_counts.get(comp, 0) + 1

        if active != expected:
            errors.append(
                f"  {run['run_id']}: expected {expected}, got {active}"
            )
        if active == 0:
            errors.append(
                f"  {run['run_id']}: ZERO active features!"
            )

    print(f"\nManifest validation: {len(manifest)} runs")
    print("Component breakdown:")
    for comp in sorted(component_counts):
        print(f"  {comp}: {component_counts[comp]} runs")
    print(f"  Total: {sum(component_counts.values())}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(e)
        raise ValueError(f"Validation failed with {len(errors)} errors")
    else:
        print("\nAll masks validated successfully.")


# =============================================================================
# TRAINING
# =============================================================================

class MaskedDataset:
    """Wrapper that applies a binary feature mask to sensor data."""

    def __init__(self, base_dataset, mask: np.ndarray):
        self.base_dataset = base_dataset
        self.mask = torch.from_numpy(mask).float()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item['sensor_features'] = item['sensor_features'] * self.mask
        return item


def split_modalities(sensor_data, group_indices):
    """Split [B, T, D] tensor into list of [B, T, Cm] per modality."""
    return [sensor_data[:, :, indices] for indices in group_indices]


def evaluate_model(
    encoder: nn.Module,
    decoder: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda',
    group_indices: List = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset with comprehensive metrics."""
    encoder.eval()
    decoder.eval()

    total_correct = 0
    total_tokens = 0
    total_sequences = 0
    correct_sequences = 0
    all_preds = []
    all_targets = []
    all_pred_seqs = []
    all_target_seqs = []

    with torch.no_grad():
        for batch in data_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            mods = split_modalities(sensor_data, group_indices)
            sensor_lengths = torch.full((sensor_data.size(0),), sensor_data.size(1),
                                        dtype=torch.long, device=device)
            encoder_out = encoder(mods, sensor_lengths)
            sensor_memory = encoder_out['memory']

            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            preds = logits.argmax(dim=-1)

            pad_mask = target_tokens != 0
            total_correct += ((preds == target_tokens) & pad_mask).sum().item()
            total_tokens += pad_mask.sum().item()

            all_preds.append(preds)
            all_targets.append(target_tokens)

            for i in range(preds.size(0)):
                seq_mask = pad_mask[i]
                if seq_mask.sum() > 0:
                    all_pred_seqs.append(preds[i][seq_mask].cpu().tolist())
                    all_target_seqs.append(target_tokens[i][seq_mask].cpu().tolist())
                if (preds[i][seq_mask] == target_tokens[i][seq_mask]).all():
                    correct_sequences += 1
                total_sequences += 1

    result = {
        'token_accuracy': total_correct / total_tokens if total_tokens > 0 else 0,
        'sequence_accuracy': correct_sequences / total_sequences if total_sequences > 0 else 0,
    }

    # Comprehensive metrics
    if all_preds:
        all_preds_t = torch.cat(all_preds, dim=0)
        all_targets_t = torch.cat(all_targets, dim=0)
        comp = compute_comprehensive_metrics(
            all_preds_t, all_targets_t,
            all_pred_seqs, all_target_seqs,
            pad_token=0,
        )
        result.update(comp)

    return result


def train_single_run(
    run_config: Dict,
    master_columns: List[str],
    config: Dict,
    encoder: nn.Module,
    base_datasets: Dict,
    vocab: Dict,
    output_dir: Path,
    device: str = 'cuda',
    max_epochs: int = 100,
    patience: int = 20,
    num_workers: int = 0,
    no_save_weights: bool = False,
    use_wandb: bool = False,
    group_indices: List = None,
    encoder_d_model: int = 256,
) -> Dict:
    """Train a single ablation run."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    run_id = run_config['run_id']
    seed = run_config['seed']
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build mask
    mask = build_mask_from_spec(master_columns, run_config['mask_spec'])
    active = int(mask.sum())

    print(f"\n{'='*70}")
    print(f"[{run_id}] seed={seed}, features={active}/{len(master_columns)}")
    print(f"{'='*70}")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project='gcode-comprehensive-ablation',
                name=run_id,
                tags=run_config.get('tags', []),
                config={
                    'component': run_config['component'],
                    'sensor': run_config['sensor'],
                    'ablation_type': run_config['ablation_type'],
                    'target': run_config['target'],
                    'seed': seed,
                    'features_active': active,
                    'max_epochs': max_epochs,
                    'patience': patience,
                    **{k: v for k, v in config.items() if not isinstance(v, dict)},
                },
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}, continuing without wandb")
            wandb_run = None

    # Create masked datasets
    train_dataset = MaskedDataset(base_datasets['train'], mask)
    val_dataset = MaskedDataset(base_datasets['val'], mask)
    test_dataset = MaskedDataset(base_datasets['test'], mask)

    batch_size = config.get('batch_size', 16)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=decoder_collate_fn, num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=decoder_collate_fn, num_workers=num_workers,
    )

    # Create decoder
    input_dim = len(master_columns)
    d_model = config['d_model']
    n_heads = config['n_heads']
    if d_model % n_heads != 0:
        for nh in [32, 24, 16, 8]:
            if d_model % nh == 0:
                n_heads = nh
                break

    decoder = SensorMultiHeadDecoder(
        sensor_dim=encoder_d_model,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=config['n_layers'],
        vocab_size=len(vocab),
        n_operations=9,
        n_types=4,
        n_commands=6,
        n_param_types=10,
        d_ff=d_model * config.get('ffn_multiplier', 4),
        dropout=config['dropout'],
        max_seq_len=config.get('max_seq_len', 32),
        n_decimal_digits=4,
        max_int_digits=2,
        embed_dropout=config.get('embed_dropout', 0.1),
        drop_path_rate=config.get('drop_path_rate', 0.1),
        use_sensor_prior=config.get('use_sensor_prior', True),
        sensor_prior_weight=config.get('sensor_prior_weight', 0.5),
    )
    decoder.to(device)

    optimizer = AdamW(
        decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=config.get('restart_period', 20),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=config.get('label_smoothing', 0.1),
    )

    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        # Train
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            sensor_data = batch['sensor_features'].to(device)
            operations = batch['operation_type'].to(device)
            input_tokens = batch['input_tokens'].to(device)
            target_tokens = batch['target_tokens'].to(device)

            with torch.no_grad():
                mods = split_modalities(sensor_data, group_indices)
                sensor_lengths = torch.full((sensor_data.size(0),), sensor_data.size(1),
                                            dtype=torch.long, device=device)
                encoder_out = encoder(mods, sensor_lengths)
                sensor_memory = encoder_out['memory']

            optimizer.zero_grad()
            outputs = decoder(
                tokens=input_tokens,
                sensor_embeddings=sensor_memory,
                operation_type=operations,
            )

            logits = outputs['legacy_logits']
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(), config.get('grad_clip', 1.0)
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_metrics = evaluate_model(encoder, decoder, val_loader, device, group_indices)
        val_acc = val_metrics['token_accuracy']

        if wandb_run:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_loss,
                'val_token_acc': val_acc,
                'val_seq_acc': val_metrics['sequence_accuracy'],
            })

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}:  Loss: {avg_loss:.4f}  Token Acc: val={val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'val_acc': val_acc,
                'run_id': run_id,
            }, run_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test evaluation
    ckpt_path = run_dir / 'best_model.pt'
    if ckpt_path.exists():
        decoder.load_state_dict(
            torch.load(ckpt_path, weights_only=False)['model_state_dict']
        )
    test_metrics = evaluate_model(encoder, decoder, test_loader, device, group_indices)

    elapsed = time.time() - start_time
    results = {
        'run_id': run_id,
        'component': run_config['component'],
        'sensor': run_config['sensor'],
        'ablation_type': run_config['ablation_type'],
        'target': run_config['target'],
        'seed': seed,
        'best_epoch': best_epoch,
        'best_val_token_acc': best_val_acc,
        'test_token_acc': test_metrics['token_accuracy'],
        'test_sequence_acc': test_metrics['sequence_accuracy'],
        'test_precision_macro': test_metrics.get('precision_macro', 0),
        'test_recall_macro': test_metrics.get('recall_macro', 0),
        'test_f1_macro': test_metrics.get('f1_macro', 0),
        'test_bleu_1': test_metrics.get('bleu_1', 0),
        'test_bleu_4': test_metrics.get('bleu_4', 0),
        'test_edit_distance': test_metrics.get('edit_distance', 0),
        'test_exact_match': test_metrics.get('exact_match', 0),
        'features_active': active,
        'features_total': len(master_columns),
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
    }

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if wandb_run:
        import wandb
        wandb_metrics = {
            'test_token_acc': test_metrics['token_accuracy'],
            'test_seq_acc': test_metrics['sequence_accuracy'],
            'best_epoch': best_epoch,
        }
        for k in ['precision_macro', 'recall_macro', 'f1_macro', 'bleu_1', 'bleu_4',
                   'edit_distance', 'exact_match']:
            if k in test_metrics:
                wandb_metrics[f'test_{k}'] = test_metrics[k]
        wandb.log(wandb_metrics)
        wandb.finish()

    # Optionally delete weights
    if no_save_weights and ckpt_path.exists():
        ckpt_path.unlink()

    print(f"  Done: Token Acc: val={best_val_acc:.2%}  test={test_metrics['token_accuracy']:.2%}  "
          f"elapsed={elapsed:.0f}s")

    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def run_analysis(output_dir: Path, manifest: List[Dict]):
    """Run ANOVA and importance analysis on completed results."""
    from scipy import stats

    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    all_results = {}
    missing = []
    for run in manifest:
        results_path = output_dir / run['run_id'] / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                all_results[run['run_id']] = json.load(f)
        else:
            missing.append(run['run_id'])

    print(f"\nLoaded {len(all_results)}/{len(manifest)} results ({len(missing)} missing)")
    if missing and len(missing) <= 20:
        for m in missing:
            print(f"  Missing: {m}")

    # =========================================================================
    # Per-component analysis
    # =========================================================================

    # Helper: group results by (component, sensor, ablation_type, target)
    # across seeds to get lists of test_token_acc for ANOVA
    def group_by_condition(component_filter):
        groups = {}
        for run_id, result in all_results.items():
            if result['component'] != component_filter:
                continue
            key = (result['sensor'], result['ablation_type'], result['target'])
            if key not in groups:
                groups[key] = []
            groups[key].append(result['test_token_acc'])
        return groups

    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_runs': len(manifest),
            'completed_runs': len(all_results),
            'missing_runs': len(missing),
        },
        'components': {},
    }

    # --- Component A: Per-sensor modality importance ---
    comp_a = group_by_condition('A')
    if comp_a:
        sensor_rankings = {}
        for (sensor, atype, target), accs in comp_a.items():
            if sensor not in sensor_rankings:
                sensor_rankings[sensor] = {}
            key = f'{atype}_{target}'
            sensor_rankings[sensor][key] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'n': len(accs),
                'values': accs,
            }

        # Compute importance for each sensor
        for sensor, results in sensor_rankings.items():
            baseline_key = 'baseline_all'
            if baseline_key in results:
                baseline_mean = results[baseline_key]['mean']
                for key, data in results.items():
                    if key.startswith('grouped_loo_') or key.startswith('individual_loo_'):
                        data['importance'] = baseline_mean - data['mean']

        report['components']['A'] = {
            'description': 'Per-sensor modality ablation',
            'sensor_rankings': sensor_rankings,
        }

    # --- Component B: Global baselines ---
    comp_b = group_by_condition('B')
    if comp_b:
        baselines = {}
        for (sensor, atype, target), accs in comp_b.items():
            key = f'{sensor}_{target}'
            baselines[key] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'n': len(accs),
            }
        report['components']['B'] = {
            'description': 'Global baselines',
            'baselines': baselines,
        }

    # --- Component C: Sensor-level LOO ---
    comp_c = group_by_condition('C')
    if comp_c:
        sensor_loo = {}
        for (sensor, atype, target), accs in comp_c.items():
            sensor_loo[sensor] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'n': len(accs),
            }

        # ANOVA across sensors
        groups_for_anova = [v for v in comp_c.values() if len(v) >= 2]
        anova_result = None
        if len(groups_for_anova) >= 2:
            try:
                f_stat, p_value = stats.f_oneway(*groups_for_anova)
                anova_result = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant_005': bool(p_value < 0.05),
                    'significant_001': bool(p_value < 0.01),
                }
                # Tukey HSD
                if p_value < 0.05:
                    try:
                        tukey = stats.tukey_hsd(*groups_for_anova)
                        sensor_names = list(comp_c.keys())
                        pairwise = []
                        for i in range(len(sensor_names)):
                            for j in range(i + 1, len(sensor_names)):
                                si = sensor_names[i][0]  # sensor name from tuple
                                sj = sensor_names[j][0]
                                p = float(tukey.pvalue[i][j])
                                pairwise.append({
                                    'sensor_a': si,
                                    'sensor_b': sj,
                                    'p_value': p,
                                    'significant': bool(p < 0.05),
                                })
                        anova_result['tukey_pairwise'] = pairwise
                    except Exception as e:
                        anova_result['tukey_error'] = str(e)
            except Exception as e:
                anova_result = {'error': str(e)}

        report['components']['C'] = {
            'description': 'Sensor-level leave-one-out',
            'sensor_loo': sensor_loo,
            'anova': anova_result,
        }

    # --- Component D: Cross-sensor grouped ---
    comp_d = group_by_condition('D')
    if comp_d:
        cross_grouped = {}
        for (sensor, atype, target), accs in comp_d.items():
            key = f'{atype}_{target}'
            cross_grouped[key] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'n': len(accs),
            }
        report['components']['D'] = {
            'description': 'Cross-sensor grouped modality ablation',
            'results': cross_grouped,
        }

    # --- Component F: Leakage ---
    comp_f = group_by_condition('F')
    if comp_f:
        leakage = {}
        for (sensor, atype, target), accs in comp_f.items():
            key = f'{sensor}_{target}'
            leakage[key] = {
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
                'n': len(accs),
            }
        report['components']['F'] = {
            'description': 'Leakage isolation analysis',
            'results': leakage,
        }

    # Save report
    with open(analysis_dir / 'comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ABLATION ANALYSIS SUMMARY")
    print("=" * 80)

    if 'B' in report['components']:
        print("\n--- GLOBAL BASELINES ---")
        for key, data in sorted(report['components']['B']['baselines'].items()):
            print(f"  {key:40s}: {data['mean']*100:.2f}% (+/-{data['std']*100:.2f})")

    if 'C' in report['components']:
        print("\n--- SENSOR LOO (accuracy when sensor removed) ---")
        sorted_loo = sorted(
            report['components']['C']['sensor_loo'].items(),
            key=lambda x: x[1]['mean']
        )
        for sensor, data in sorted_loo:
            print(f"  without {sensor:15s}: {data['mean']*100:.2f}%")
        if report['components']['C'].get('anova'):
            a = report['components']['C']['anova']
            print(f"  ANOVA: F={a.get('f_statistic', 'N/A'):.4f}, "
                  f"p={a.get('p_value', 'N/A'):.6f}")

    if 'F' in report['components']:
        print("\n--- LEAKAGE ANALYSIS ---")
        for key, data in sorted(report['components']['F']['results'].items()):
            print(f"  {key:40s}: {data['mean']*100:.2f}%")

    print(f"\nFull report: {analysis_dir / 'comprehensive_report.json'}")


# =============================================================================
# RAY PARALLEL TRAINING
# =============================================================================

def ray_worker(
    run_config: Dict,
    master_columns: List[str],
    config: Dict,
    encoder_path: str,
    data_dir: str,
    vocab_path: str,
    output_dir: str,
    max_epochs: int,
    patience: int,
    num_workers: int,
    no_save_weights: bool,
    use_wandb: bool,
    gpu_id: int,
) -> Dict:
    """
    Ray remote function: each worker loads its own encoder + datasets
    on a specific GPU and trains one run.
    """
    import torch
    # Ray sets CUDA_VISIBLE_DEVICES to a single GPU, so always use cuda:0
    device = 'cuda:0'
    output_dir = Path(output_dir)

    run_id = run_config['run_id']
    results_path = output_dir / run_id / 'results.json'
    if results_path.exists():
        return {'run_id': run_id, 'status': 'skipped'}

    try:
        # Load resources (each worker independently)
        with open(vocab_path) as f:
            vocab = json.load(f)['vocab']

        # Build modality indices
        metadata_path = Path(data_dir) / 'metadata.json'
        with open(metadata_path) as f:
            metadata = json.load(f)
        columns = metadata['continuous_columns']
        sensor_patterns = {
            'accelerometer': ['Ax', 'Ay', 'Az'],
            'gyroscope': ['Gx', 'Gy', 'Gz'],
            'magnetometer': ['Mx', 'My', 'Mz'],
            'environmental': ['Pressure', 'Temperature', 'Proximity'],
            'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
            'rms': ['RMS'],
        }
        groups = {name: [] for name in sensor_patterns}
        groups['machine'] = []
        for idx, col in enumerate(columns):
            matched = False
            if '.' in col:
                _, feat = col.rsplit('.', 1)
                for group_name, patterns in sensor_patterns.items():
                    if feat in patterns:
                        groups[group_name].append(idx)
                        matched = True
                        break
            if not matched:
                groups['machine'].append(idx)
        group_names = list(groups.keys())
        group_indices = [groups[name] for name in group_names]
        sensor_dims = [len(groups[name]) for name in group_names]

        # Load MM_DTAE_LSTM encoder
        ckpt = torch.load(encoder_path, map_location='cpu', weights_only=False)
        enc_config = ckpt['config']
        if not isinstance(enc_config, ModelConfig):
            enc_config = ModelConfig(**{k: v for k, v in enc_config.items()
                                        if k in ModelConfig.__dataclass_fields__})
        encoder = MM_DTAE_LSTM(enc_config)
        sd = ckpt['model_state_dict']
        if 'head_cls.weight' in sd:
            n_cls = sd['head_cls.weight'].shape[0]
            if encoder.head_cls.out_features != n_cls:
                encoder.head_cls = nn.Linear(enc_config.d_model, n_cls)
        encoder.load_state_dict(sd, strict=False)
        encoder_d_model = enc_config.d_model
        encoder.to(device)
        encoder.eval()

        max_seq_len = config.get('max_seq_len', 32)
        base_datasets = {
            'train': DecoderDatasetFromSplits(
                split_dir=data_dir, split='train', max_token_len=max_seq_len,
            ),
            'val': DecoderDatasetFromSplits(
                split_dir=data_dir, split='val', max_token_len=max_seq_len,
            ),
            'test': DecoderDatasetFromSplits(
                split_dir=data_dir, split='test', max_token_len=max_seq_len,
            ),
        }

        result = train_single_run(
            run_config=run_config,
            master_columns=master_columns,
            config=config,
            encoder=encoder,
            base_datasets=base_datasets,
            vocab=vocab,
            output_dir=output_dir,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            num_workers=num_workers,
            no_save_weights=no_save_weights,
            use_wandb=use_wandb,
            group_indices=group_indices,
            encoder_d_model=encoder_d_model,
        )
        return {'run_id': run_id, 'status': 'completed', 'result': result}

    except Exception as e:
        traceback.print_exc()
        return {'run_id': run_id, 'status': 'failed', 'error': str(e)}


def train_with_ray(
    manifest: List[Dict],
    master_columns: List[str],
    config: Dict,
    args,
    output_dir: Path,
):
    """Distribute training across GPUs using Ray."""
    import ray

    num_gpus = args.num_gpus
    ray.init(ignore_reinit_error=True)
    print(f"\nRay initialized: {ray.cluster_resources()}")
    print(f"Using {num_gpus} GPUs for parallel training")

    # Each worker needs num_gpus=1 so Ray makes CUDA visible to the process.
    # We still manually assign gpu_id for device placement.
    remote_worker = ray.remote(num_cpus=1, num_gpus=1)(ray_worker)

    subset = manifest[args.start_idx:args.end_idx]

    # Filter out already-completed runs
    pending = []
    skipped = 0
    for run_config in subset:
        results_path = output_dir / run_config['run_id'] / 'results.json'
        if results_path.exists():
            skipped += 1
        else:
            pending.append(run_config)

    print(f"Total: {len(subset)}, Pending: {len(pending)}, Already done: {skipped}")

    if not pending:
        print("Nothing to do.")
        ray.shutdown()
        return

    # Submit tasks round-robin across GPUs
    # Keep up to num_gpus tasks in flight at a time
    in_flight = {}  # ray_ref -> run_id
    completed = 0
    failed = 0
    failures = []
    failures_path = output_dir / 'failures.json'
    if failures_path.exists():
        with open(failures_path) as f:
            failures = json.load(f)

    task_idx = 0

    def submit_next():
        nonlocal task_idx
        if task_idx >= len(pending):
            return
        run_config = pending[task_idx]
        gpu_id = task_idx % num_gpus
        ref = remote_worker.remote(
            run_config=run_config,
            master_columns=master_columns,
            config=config,
            encoder_path=args.encoder_path,
            data_dir=args.data_dir,
            vocab_path=args.vocab_path,
            output_dir=str(output_dir),
            max_epochs=args.max_epochs,
            patience=args.patience,
            num_workers=args.num_workers,
            no_save_weights=args.no_save_weights,
            use_wandb=not args.no_wandb,
            gpu_id=gpu_id,
        )
        in_flight[ref] = run_config['run_id']
        task_idx += 1

    # Seed initial batch
    for _ in range(min(num_gpus, len(pending))):
        submit_next()

    # Process as they complete
    while in_flight:
        done_refs, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        for ref in done_refs:
            run_id = in_flight.pop(ref)
            try:
                result = ray.get(ref)
                if result['status'] == 'completed':
                    completed += 1
                    print(f"  [{completed + failed}/{len(pending)}] "
                          f"{run_id}: DONE")
                elif result['status'] == 'failed':
                    failed += 1
                    failures.append({
                        'run_id': run_id,
                        'error': result.get('error', 'unknown'),
                        'timestamp': datetime.now().isoformat(),
                    })
                    print(f"  [{completed + failed}/{len(pending)}] "
                          f"{run_id}: FAILED - {result.get('error', '')[:80]}")
                else:
                    # skipped (shouldn't happen since we pre-filtered)
                    pass
            except Exception as e:
                failed += 1
                failures.append({
                    'run_id': run_id,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })
                print(f"  [{completed + failed}/{len(pending)}] "
                      f"{run_id}: EXCEPTION - {str(e)[:80]}")

            # Save failures incrementally
            with open(failures_path, 'w') as f:
                json.dump(failures, f, indent=2)

            # Submit next task
            submit_next()

    ray.shutdown()
    print(f"\n{'='*70}")
    print(f"RAY TRAINING COMPLETE: {completed} done, {skipped} skipped, {failed} failed")
    print(f"{'='*70}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive nested ablation study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--phase', choices=['manifest', 'validate', 'train', 'analyze', 'all'],
                        default='all', help='Which phase to run')
    parser.add_argument('--mode', choices=['full', 'conference'], default='conference',
                        help='full=2394 runs, conference=492 runs')
    parser.add_argument('--config', type=str, help='Model config JSON')
    parser.add_argument('--data-dir', type=str, required=True, help='Data splits directory')
    parser.add_argument('--encoder-path', type=str, help='Encoder checkpoint')
    parser.add_argument('--vocab-path', type=str, help='Vocabulary JSON')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seeds', type=str, default='42,123,456',
                        help='Comma-separated seeds')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='First manifest index to process')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='Last manifest index to process (exclusive)')
    parser.add_argument('--components', type=str, default='A,B,C,D,E,F',
                        help='Comma-separated components to include')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers (default: 32 on Linux, 0 on macOS)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no-save-weights', action='store_true',
                        help='Delete model checkpoints after evaluation')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--manifest-path', type=str, default=None,
                        help='Path to pre-generated manifest')
    parser.add_argument('--use-ray', action='store_true',
                        help='Use Ray for parallel training across GPUs')
    parser.add_argument('--num-gpus', type=int, default=2,
                        help='Number of GPUs for Ray (default: 2)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(',')]
    components = [c.strip() for c in args.components.split(',')]

    if args.num_workers is None:
        args.num_workers = 0 if platform.system() in ('Windows', 'Darwin') else 32

    # Load master columns
    metadata_path = Path(args.data_dir) / 'metadata.json'
    with open(metadata_path) as f:
        metadata = json.load(f)
    master_columns = metadata['continuous_columns']
    print(f"Loaded {len(master_columns)} columns from metadata")

    # ---- Phase: Manifest ----
    if args.phase in ('manifest', 'all'):
        manifest = generate_manifest(args.mode, master_columns, seeds, components)
        manifest_path = output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"\nGenerated manifest: {len(manifest)} runs ({args.mode} mode)")

        # Summary
        comp_counts = {}
        for run in manifest:
            c = run['component']
            comp_counts[c] = comp_counts.get(c, 0) + 1
        for c in sorted(comp_counts):
            print(f"  Component {c}: {comp_counts[c]} runs")

    # ---- Phase: Validate ----
    if args.phase in ('validate', 'all'):
        manifest_path = args.manifest_path or (output_dir / 'manifest.json')
        with open(manifest_path) as f:
            manifest = json.load(f)
        validate_manifest(manifest, master_columns)

    # ---- Phase: Train ----
    if args.phase in ('train', 'all'):
        assert args.config, "--config required for training"
        assert args.encoder_path, "--encoder-path required for training"
        assert args.vocab_path, "--vocab-path required for training"

        # Load config (needed for both Ray and sequential paths)
        with open(args.config) as f:
            config = json.load(f)

        # Load manifest
        manifest_path = args.manifest_path or (output_dir / 'manifest.json')
        with open(manifest_path) as f:
            manifest = json.load(f)

        if args.use_ray:
            # ---- Ray parallel path ----
            print(f"Using Ray with {args.num_gpus} GPUs")
            train_with_ray(
                manifest=manifest,
                master_columns=master_columns,
                config=config,
                args=args,
                output_dir=output_dir,
            )
        else:
            # ---- Sequential path ----
            # Auto-detect device
            if args.device is None:
                if torch.cuda.is_available():
                    args.device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    args.device = 'mps'
                else:
                    args.device = 'cpu'
            print(f"Device: {args.device}")

            with open(args.vocab_path) as f:
                vocab_data = json.load(f)
            vocab = vocab_data['vocab']

            # Build modality indices
            _sensor_patterns = {
                'accelerometer': ['Ax', 'Ay', 'Az'],
                'gyroscope': ['Gx', 'Gy', 'Gz'],
                'magnetometer': ['Mx', 'My', 'Mz'],
                'environmental': ['Pressure', 'Temperature', 'Proximity'],
                'color': ['ColorR', 'ColorG', 'ColorB', 'ColorA'],
                'rms': ['RMS'],
            }
            _groups = {name: [] for name in _sensor_patterns}
            _groups['machine'] = []
            for idx, col in enumerate(master_columns):
                matched = False
                if '.' in col:
                    _, feat = col.rsplit('.', 1)
                    for gn, pats in _sensor_patterns.items():
                        if feat in pats:
                            _groups[gn].append(idx)
                            matched = True
                            break
                if not matched:
                    _groups['machine'].append(idx)
            _group_names = list(_groups.keys())
            group_indices = [_groups[name] for name in _group_names]

            # Load MM_DTAE_LSTM encoder (frozen)
            ckpt = torch.load(args.encoder_path, map_location='cpu', weights_only=False)
            enc_config = ckpt['config']
            if not isinstance(enc_config, ModelConfig):
                enc_config = ModelConfig(**{k: v for k, v in enc_config.items()
                                            if k in ModelConfig.__dataclass_fields__})
            encoder = MM_DTAE_LSTM(enc_config)
            sd = ckpt['model_state_dict']
            if 'head_cls.weight' in sd:
                n_cls = sd['head_cls.weight'].shape[0]
                if encoder.head_cls.out_features != n_cls:
                    encoder.head_cls = nn.Linear(enc_config.d_model, n_cls)
            encoder.load_state_dict(sd, strict=False)
            encoder_d_model = enc_config.d_model
            encoder.to(args.device)
            encoder.eval()
            print(f"MM_DTAE_LSTM encoder loaded and frozen (d_model={encoder_d_model})")

            # Load base datasets
            max_seq_len = config.get('max_seq_len', 32)
            base_datasets = {
                'train': DecoderDatasetFromSplits(
                    split_dir=args.data_dir, split='train', max_token_len=max_seq_len,
                ),
                'val': DecoderDatasetFromSplits(
                    split_dir=args.data_dir, split='val', max_token_len=max_seq_len,
                ),
                'test': DecoderDatasetFromSplits(
                    split_dir=args.data_dir, split='test', max_token_len=max_seq_len,
                ),
            }
            print(f"Datasets loaded: train={len(base_datasets['train'])}, "
                  f"val={len(base_datasets['val'])}, test={len(base_datasets['test'])}")

            subset = manifest[args.start_idx:args.end_idx]
            print(f"\nProcessing runs {args.start_idx} to "
                  f"{args.start_idx + len(subset) - 1} of {len(manifest)}")

            # Failures log
            failures_path = output_dir / 'failures.json'
            failures = []
            if failures_path.exists():
                with open(failures_path) as f:
                    failures = json.load(f)

            completed = 0
            skipped = 0
            failed = 0

            for i, run_config in enumerate(subset):
                run_dir = output_dir / run_config['run_id']
                results_path = run_dir / 'results.json'

                if results_path.exists():
                    skipped += 1
                    continue

                global_idx = args.start_idx + i
                print(f"\n[{global_idx}/{len(manifest)}] ({completed} done, "
                      f"{skipped} skipped, {failed} failed)")

                try:
                    train_single_run(
                        run_config=run_config,
                        master_columns=master_columns,
                        config=config,
                        encoder=encoder,
                        base_datasets=base_datasets,
                        vocab=vocab,
                        output_dir=output_dir,
                        device=args.device,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        num_workers=args.num_workers,
                        no_save_weights=args.no_save_weights,
                        use_wandb=not args.no_wandb,
                        group_indices=group_indices,
                        encoder_d_model=encoder_d_model,
                    )
                    completed += 1
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e) or 'out of memory' in str(e):
                        print(f"  OOM ERROR: {e}")
                        failures.append({
                            'run_id': run_config['run_id'],
                            'error': 'OOM',
                            'timestamp': datetime.now().isoformat(),
                        })
                        failed += 1
                        torch.cuda.empty_cache()
                    else:
                        raise
                except Exception as e:
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                    failures.append({
                        'run_id': run_config['run_id'],
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                    })
                    failed += 1

                # Save failures incrementally
                with open(failures_path, 'w') as f:
                    json.dump(failures, f, indent=2)

            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETE: {completed} done, {skipped} skipped, {failed} failed")
            print(f"{'='*70}")

    # ---- Phase: Analyze ----
    if args.phase in ('analyze', 'all'):
        manifest_path = args.manifest_path or (output_dir / 'manifest.json')
        with open(manifest_path) as f:
            manifest = json.load(f)
        run_analysis(output_dir, manifest)


if __name__ == '__main__':
    main()
