# Contributing to RuleChef

## Development Setup

```bash
git clone https://github.com/KRLabsOrg/rulechef.git
cd rulechef
pip install -e ".[dev,grex]"
```

## Running Tests

```bash
# All tests (excluding live API tests)
pytest --ignore=tests/test_incremental_live.py -q

# With coverage
pytest --ignore=tests/test_incremental_live.py --cov=rulechef

# Live API tests (requires OPENAI_API_KEY)
pytest tests/test_incremental_live.py -m integration
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy-lang.org/) for type checking.

```bash
# Lint
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Type check
mypy rulechef/ --ignore-missing-imports
```

CI runs these checks on every pull request. Make sure they pass before submitting.

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests as needed
4. Ensure `ruff check .`, `mypy rulechef/`, and `pytest` all pass
5. Open a pull request against `main`

## Developer Certificate of Origin

By contributing you certify the [Developer Certificate of Origin](https://developercertificate.org/):
that you wrote the contribution (or have the right to submit it) and that it may
be distributed under this repository's Apache-2.0 license. You confirm this by
signing off each commit:

```bash
git commit -s -m "your message"
```

which adds a `Signed-off-by: Your Name <you@example.com>` line. PRs with
unsigned commits will be asked to rebase with sign-off.

## Contributor tasks

Issues labeled [`good first issue`](https://github.com/KRLabsOrg/rulechef/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
are scoped for newcomers: each includes the relevant file/line pointers,
acceptance criteria, and a "Start here" command. Comment on the issue to claim it.
