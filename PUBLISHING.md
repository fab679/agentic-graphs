# Publishing to PyPI

This guide explains how to publish `agentic-graphs` to PyPI using GitHub Actions.

## Prerequisites

1. PyPI Account: Create an account at [pypi.org](https://pypi.org/account/register/)
2. GitHub Repository Secrets: Set up trusted publishing

## Setup (One-time)

### 1. Configure PyPI Trusted Publishing

Follow the [PyPI documentation](https://docs.pypi.org/trusted-publishers/) to allow GitHub Actions to publish without API tokens:

1. Go to your PyPI account settings
2. Add a trusted publisher (GitHub)
3. Select your repository and workflow

Alternatively, you can use an API token:

1. Create an API token at https://pypi.org/manage/account/tokens/
2. Add it as a GitHub Secret named `PYPI_API_TOKEN`

### 2. Update Package Metadata

Ensure `pyproject.toml` has:
- `license = {text = "MIT"}`
- `authors` field
- `project.urls` with repository link (update the URL to your actual GitHub repo)

Update this in `pyproject.toml`:
```toml
[project.urls]
Repository = "https://github.com/YOUR_USERNAME/agentic-graphs"
Documentation = "https://github.com/YOUR_USERNAME/agentic-graphs#readme"
BugTracker = "https://github.com/YOUR_USERNAME/agentic-graphs/issues"
```

## Publishing

### Option 1: Publish via Release (Recommended)

1. Update `version` in `pyproject.toml`
2. Commit and create a git tag:
   ```bash
   git tag v0.2.1
   git push origin v0.2.1
   ```
3. Create a release on GitHub (e.g., https://github.com/YOUR_USERNAME/agentic-graphs/releases/new)
4. The `publish.yml` workflow will automatically build and publish to PyPI

### Option 2: Manual Workflow Dispatch

1. Go to Actions > Publish to PyPI
2. Click "Run workflow"
3. Enter the version number (e.g., `0.2.1`)
4. The workflow will build, publish, and create a release

## Verification

After publishing:

1. Check [pypi.org/project/agentic-graphs](https://pypi.org/project/agentic-graphs/)
2. Install from PyPI:
   ```bash
   pip install agentic-graphs==0.2.1
   ```
3. Verify imports work:
   ```python
   from agentic_graphs import Agent, Graph, Node
   print("✓ Installation successful")
   ```

## Troubleshooting

- **"Unauthorized" error**: Check PyPI trusted publishing is configured or API token is set
- **Build fails**: Ensure `pyproject.toml` is valid and `src/agentic_graphs/__init__.py` exists
- **Version already exists**: Update version in `pyproject.toml` before publishing

## CI/CD Details

Two workflows are configured:

### `tests.yml`
- Runs on every push and PR
- Tests Python 3.10, 3.11, 3.12
- Checks linting and imports

### `publish.yml`
- Triggers on release creation or manual workflow dispatch
- Builds wheel and sdist
- Publishes to PyPI using trusted publishing or API token
- Creates release notes automatically (for manual dispatch)
