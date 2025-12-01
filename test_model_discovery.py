#!/usr/bin/env python3
"""
Quick test to verify the 5 new models will be discovered by flask_dashboard
"""
from pathlib import Path

def discover_models():
    """Simulate the model discovery logic from flask_dashboard.py"""
    models = []
    outputs_dir = Path('outputs')

    # Scan for checkpoint_best.pt files
    for checkpoint in outputs_dir.rglob('checkpoint_best.pt'):
        model_dir = checkpoint.parent

        models.append({
            'path': str(checkpoint),
            'name': model_dir.name,
            'size_mb': checkpoint.stat().st_size / (1024 * 1024)
        })

    return sorted(models, key=lambda x: x['name'])

if __name__ == '__main__':
    print("=" * 80)
    print("MODEL DISCOVERY TEST")
    print("=" * 80)

    models = discover_models()

    print(f"\nFound {len(models)} models:\n")

    # Highlight the 5 new models
    target_models = [
        'Best_ParamValue_ugkjmojf_Nov27',
        'Best_ParamType_4hufje7i_Nov29',
        'Best_ParamValue2_b0dmn0l_Nov27',
        'Run_s9z6u0cz_Nov26',
        'Run_op4m69pw_Nov29'
    ]

    found_targets = []

    for model in models:
        is_target = model['name'] in target_models
        marker = "✅ NEW" if is_target else "   "
        print(f"{marker}  {model['name']:<50} {model['size_mb']:>8.1f} MB")

        if is_target:
            found_targets.append(model['name'])

    print("\n" + "=" * 80)
    print(f"TARGET MODELS FOUND: {len(found_targets)}/5")
    print("=" * 80)

    if len(found_targets) == 5:
        print("\n✅ SUCCESS! All 5 models will appear in the flask_dashboard dropdown!")
    else:
        print(f"\n⚠️  Only found {len(found_targets)} of 5 target models:")
        for name in found_targets:
            print(f"   ✅ {name}")

        missing = set(target_models) - set(found_targets)
        if missing:
            print(f"\n   Missing:")
            for name in missing:
                print(f"   ❌ {name}")
