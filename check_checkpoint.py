#!/usr/bin/env python3
"""Quick script to inspect checkpoint keys."""
import torch

checkpoint = torch.load('outputs/direct_regression/checkpoint_best.pt', map_location='cpu', weights_only=False)
state_dict = checkpoint.get('multihead_state_dict', checkpoint.get('model_state_dict', {}))

# Find all param_value related keys
param_value_keys = [k for k in state_dict.keys() if 'param_value' in k]
print('Param value keys:')
for k in sorted(param_value_keys):
    print(f'  {k}: {state_dict[k].shape}')
