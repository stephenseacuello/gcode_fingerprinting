# Testing Guide

This document describes the testing infrastructure for the G-code fingerprinting project.

## Overview

The project uses `pytest` for all testing with the following structure:

```
tests/
├── conftest.py                 # Shared fixtures
├── pytest.ini                  # Pytest configuration
├── unit/                       # Unit tests
│   ├── test_data_augmentation.py
│   ├── test_gcode_tokenizer.py
│   └── test_target_utils.py
├── integration/                # Integration tests
│   ├── test_train_pipeline.py
│   └── test_inference_pipeline.py
└── fixtures/                   # Test data fixtures
```

## Running Tests

### Quick Start

```bash
# Run all tests
./scripts/run_tests.sh

# Run without coverage
./scripts/run_tests.sh --no-cov

# Run only fast tests
./scripts/run_tests.sh -m "not slow"

# Run tests in parallel
./scripts/run_tests.sh -n
```

### Using pytest directly

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_data_augmentation.py

# Run tests matching pattern
pytest tests/ -k "augment"

# Run with verbose output
pytest tests/ -vv

# Run with coverage
pytest tests/ --cov=src/miracle --cov-report=html
```

## Test Categories

Tests are organized by markers:

- `@pytest.mark.unit` - Fast unit tests for individual modules
- `@pytest.mark.integration` - Integration tests requiring full pipeline
- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.gpu` - Tests requiring GPU/MPS

### Running specific categories

```bash
# Run only unit tests
pytest tests/ -m unit

# Run all except slow tests
pytest tests/ -m "not slow"

# Run integration tests
pytest tests/ -m integration
```

## Current Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| data_augmentation.py | 24 | 86% |
| gcode_tokenizer.py | 37 | ~80% |
| target_utils.py | 27 | ~85% |
| **Total** | **88** | **~80%** |

## Writing New Tests

### Unit Test Example

```python
import pytest
from miracle.dataset.data_augmentation import DataAugmenter

@pytest.mark.unit
class TestDataAugmenter:
    def test_noise_preserves_shape(self):
        """Test that noise injection preserves tensor shape."""
        augmenter = DataAugmenter(noise_level=0.02)
        continuous = torch.randn(64, 135)

        augmented = augmenter.add_sensor_noise(continuous)

        assert augmented.shape == continuous.shape
```

### Using Fixtures

Shared fixtures are defined in `tests/conftest.py`:

```python
def test_with_sample_data(sample_continuous_data, sample_categorical_data):
    """Test using shared fixtures."""
    assert sample_continuous_data.shape == (5, 64, 135)
    assert sample_categorical_data.shape == (5, 64, 4)
```

### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.slow
def test_full_training_pipeline(temp_dir):
    """Test complete training pipeline on tiny dataset."""
    # Setup
    config = create_test_config()
    model = create_model(config)

    # Train for 1 epoch
    train_one_epoch(model, tiny_dataset)

    # Verify
    assert model_checkpoint_exists()
```

## Pre-commit Hooks

Install pre-commit hooks to run tests automatically:

```bash
pip install pre-commit
pre-commit install
```

Hooks will run on `git commit`:
- Code formatting (black, isort)
- Linting (flake8)
- Fast tests (unit tests only)

## Continuous Integration

When pushing to GitHub, tests run automatically via GitHub Actions (TODO: setup).

## Code Coverage

### Viewing Coverage Report

After running tests with coverage:

```bash
# Generate HTML report
pytest tests/ --cov=src/miracle --cov-report=html

# Open in browser
open htmlcov/index.html
```

### Coverage Requirements

- **Minimum coverage:** 70% (currently disabled during test build-up)
- **Target coverage:** 80%+
- New code should include tests

## Troubleshooting

### Tests fail with import errors

```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Install test dependencies
pip install -r requirements.txt
```

### Tests are slow

```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run tests in parallel
pytest tests/ -n auto
```

### Coverage report not generated

```bash
# Explicitly enable coverage
pytest tests/ --cov=src/miracle --cov-report=html
```

## Best Practices

1. **Test naming:** `test_<function>_<scenario>` (e.g., `test_augment_sample_preserves_labels`)
2. **One assertion per test:** Keep tests focused
3. **Use fixtures:** Share setup code via conftest.py
4. **Mark appropriately:** Use `@pytest.mark.unit`, `@pytest.mark.slow`, etc.
5. **Test edge cases:** Empty inputs, invalid values, boundary conditions
6. **Mock external dependencies:** Use `pytest-mock` for external services
7. **Fast tests:** Unit tests should run in <100ms

## Next Steps

- [ ] Add model tests (test_multihead_lm.py)
- [ ] Add comprehensive integration tests
- [ ] Set up GitHub Actions CI/CD
- [ ] Increase coverage to 80%+
- [ ] Add performance regression tests

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
