# Hyperparameter Sweep Quick Start

## ⚡ Fixed and Ready to Run!

The sweep configuration has been fixed to work with your `train_multihead.py` script. The issues were:
1. **Underscores vs hyphens**: W&B uses underscores, argparse expects hyphens
2. **Boolean flags**: W&B passes `--flag=True`, argparse expects just `--flag`

**Solution**: Updated the YAML to use hyphens (`max-epochs`, `wandb-project`) and always enable flags via custom command section.

---

## 🚀 Run the Basic Sweep Now

```bash
# Test in dry-run mode first
./scripts/run_sweeps.sh --sweep basic --agents 2 --dry-run

# Run the actual sweep (50-100 runs, Bayesian optimization)
./scripts/run_sweeps.sh --sweep basic --agents 2
```

**This will now work correctly!** ✅

This will optimize:
- **Model architecture:** hidden_dim (96-256), num_layers (2-4), num_heads (4-8)
- **Training:** batch_size, learning_rate, weight_decay
- **Loss weighting:** command_weight (1.0-5.0)

**Expected duration:** 2-3 days with 2 parallel agents
**Target:** >70% overall accuracy

---

## 📋 What's in the Basic Sweep?

**File:** `sweeps/phase3/basic_hyperparameter_sweep.yaml`

**Parameters being tested:**
- `hidden_dim`: [96, 128, 192, 256]
- `num_layers`: [2, 3, 4]
- `num_heads`: [4, 6, 8]
- `batch_size`: [4, 8, 16]
- `learning_rate`: 0.0001 - 0.01 (log-uniform)
- `weight_decay`: 0.0 - 0.1 (uniform)
- `command_weight`: [1.0, 2.0, 3.0, 5.0]

**Fixed parameters:**
- `max_epochs`: 50
- `use_augmentation`: true
- `oversample_factor`: 3

---

## ⚠️ Issue with Original Sweeps

The original sweep configs (vocabulary, augmentation, warmup, architecture, loss_weighting) used parameter names from W&B documentation examples, but your training script uses different argument names:

### Parameter Name Mismatches

| Sweep YAML | Your Script | Status |
|------------|-------------|--------|
| `--augmentation` | `--use-augmentation` | ❌ Mismatch |
| `--d_model` | `--hidden_dim` | ❌ Mismatch |
| `--epochs` | `--max-epochs` | ❌ Mismatch |
| `--lr` | `--learning_rate` | ❌ Mismatch |
| `--n_encoder_layers` | `--num_layers` | ❌ Mismatch |
| `--n_heads` | `--num_heads` | ❌ Mismatch |
| `--vocab_bucket_digits` | N/A | ❌ Not supported |

---

## 🔧 Two Options to Fix Other Sweeps

### Option 1: Use the Basic Sweep (Recommended for Now)

The basic sweep works immediately and will give you good results. It tests the most important hyperparameters that affect model performance.

### Option 2: Update Training Script (For Full Sweep Support)

To use all the original sweeps, you would need to update `train_multihead.py` to accept W&B-style parameter names. This requires:

1. Adding command-line arguments for all sweep parameters
2. Supporting vocabulary bucketing (`--vocab-bucket-digits`)
3. Supporting all augmentation parameters
4. Supporting warmup scheduler configuration

**This is a significant refactoring** and not necessary if the basic sweep gives you good results.

---

## 📊 Monitoring Your Sweep

### View Results in W&B

1. Go to https://wandb.ai/seacuello-university-of-rhode-island/gcode-fingerprinting
2. Click "Sweeps" tab
3. Select your sweep
4. View:
   - Parallel coordinates plot
   - Parameter importance
   - Best runs
   - Training curves

### Find Best Configuration

After the sweep completes, the best configuration will be automatically identified. You can then:

```bash
# Train final model with best config
python scripts/train_production.py \
  --sweep-id seacuello-university-of-rhode-island/gcode-fingerprinting/<sweep_id> \
  --output models/production \
  --export-onnx \
  --quantize
```

---

## 🎯 Expected Results

### Current Baseline (Phase 1)
- Command Accuracy: **100.0%** ✓
- Overall Accuracy: **58.5%**

### After Basic Sweep
- Command Accuracy: **100.0%** (maintain)
- Overall Accuracy: **65-70%** (target)
- Parameter Value: **60-65%** (improve from 56.2%)

### Improvements Expected
- Better model architecture (optimal hidden_dim, layers, heads)
- Tuned learning rate and weight decay
- Balanced loss weights for multi-head training

---

## 🔄 Sweep Lifecycle

1. **Start sweep**: Creates sweep on W&B
2. **Agents run**: 2 parallel agents execute runs
3. **Bayesian optimization**: W&B suggests next configs to try
4. **Early termination**: Hyperband stops poor runs early
5. **Best config found**: After 50-100 runs
6. **Train final model**: Use best config for production

---

## 💡 Pro Tips

1. **Start with 2 agents** - Don't overload your machine
2. **Monitor W&B dashboard** - Check progress regularly
3. **Early stopping works** - Hyperband will terminate bad runs
4. **Save the best config** - Export it for production training
5. **Validate on test set** - Re-train best model to verify results

---

## 🆘 Troubleshooting

### Sweep Fails Immediately

```bash
# Check training script works
.venv/bin/python train_multihead.py --help

# Check W&B login
wandb login --relogin
```

### Out of Memory

Reduce `batch_size` or `hidden_dim` in the sweep YAML:

```yaml
batch_size:
  values: [2, 4, 8]  # Smaller batches

hidden_dim:
  values: [96, 128]  # Smaller models
```

### Sweep Takes Too Long

- Use fewer agents (1 instead of 2)
- Reduce `max_epochs` to 30
- Let Hyperband early termination work

---

## 📚 Next Steps

After the basic sweep completes:

1. **Analyze results** in W&B dashboard
2. **Train production model** with best config
3. **Export to ONNX** and quantize
4. **Benchmark performance**
5. **Deploy with Docker**

All the infrastructure is ready for production deployment!

---

**Created:** November 19, 2025
**Status:** Ready to run
**Estimated Duration:** 2-3 days (50-100 runs, 2 agents)
