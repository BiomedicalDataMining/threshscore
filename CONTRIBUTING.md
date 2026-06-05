# Contributing to threshscore

Thank you for your interest in contributing! This document describes the development workflow.

## Development Setup

```bash
git clone https://github.com/BiomedicalDataMining/threshscore.git
cd threshscore
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

Tests require >= 90% coverage. The suite currently covers 188 tests across gates, metrics, scoring, thresholds, validation, and plotting.

## Linting and Type Checking

```bash
ruff check src tests
mypy src/threshscore
```

## Adding a Gate Function

1. Create a new module in `src/threshscore/gates/` subclassing `BaseGate`.
2. Implement `__call__(self, value: float, threshold: float, **params) -> float`.
3. Register it in `src/threshscore/gates/registry.py`.
4. Export it from `src/threshscore/gates/__init__.py`.
5. Add tests in `tests/test_gates.py`.

## Submitting Changes

- Open an issue first for non-trivial changes.
- Fork the repo and create a feature branch.
- Keep commits focused; write meaningful commit messages.
- Ensure all tests pass and coverage stays above 90% before opening a PR.
- Sign off on the [Developer Certificate of Origin](https://developercertificate.org/) by using `git commit -s`.

## Code Style

- Follow PEP 8, enforced via `ruff`.
- Type annotations are required on all public functions; `mypy --strict` must pass.
- Line length: 100 characters.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
