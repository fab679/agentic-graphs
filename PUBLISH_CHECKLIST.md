# PyPI Publishing Checklist

Follow this checklist to publish `agentic-graphs` to PyPI:

## Pre-Publication

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` or release notes (if exists)
- [ ] Ensure all tests pass: `uv run pytest`
- [ ] Verify package builds: `uv build`
- [ ] Check wheel and sdist are created in `dist/`

## GitHub Setup

- [ ] Repository is public on GitHub
- [ ] README.md is complete and accurate
- [ ] LICENSE file exists (✓ Already created)
- [ ] GitHub Actions workflows are in `.github/workflows/` (✓ Already created)
- [ ] Update repository URLs in `pyproject.toml`:
  ```toml
  [project.urls]
  Repository = "https://github.com/YOUR_USERNAME/agentic-graphs"
  Documentation = "https://github.com/YOUR_USERNAME/agentic-graphs#readme"
  BugTracker = "https://github.com/YOUR_USERNAME/agentic-graphs/issues"
  ```

## PyPI Configuration (One-time)

### Option A: Trusted Publishing (Recommended)

1. Create PyPI account at https://pypi.org/account/register/
2. Verify email
3. Go to Account Settings > Publishing
4. Add trusted publisher:
   - PyPI Project Name: `agentic-graphs`
   - GitHub Repository Owner: `YOUR_USERNAME`
   - Repository Name: `agentic-graphs`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

### Option B: API Token

1. Create API token at https://pypi.org/manage/account/tokens/
2. Add GitHub Secret:
   - Name: `PYPI_API_TOKEN`
   - Value: (paste token)
3. Update `.github/workflows/publish.yml` to use it

## Publication Methods

### Method 1: GitHub Release (Recommended)

```bash
# Update version
sed -i 's/version = ".*/version = "0.2.1"/' pyproject.toml

# Commit
git add pyproject.toml
git commit -m "Bump version to 0.2.1"

# Create tag
git tag v0.2.1
git push origin v0.2.1

# Go to GitHub and create release from tag
# https://github.com/YOUR_USERNAME/agentic-graphs/releases/new
```

### Method 2: Workflow Dispatch

1. Go to GitHub Actions
2. Select "Publish to PyPI"
3. Click "Run workflow"
4. Enter version (e.g., `0.2.1`)
5. Workflow will publish and create release

## Post-Publication

- [ ] Verify on PyPI: https://pypi.org/project/agentic-graphs/
- [ ] Test installation:
  ```bash
  pip install --upgrade agentic-graphs
  python -c "from agentic_graphs import Agent; print('✓ OK')"
  ```
- [ ] Check GitHub release was created
- [ ] Announce release in docs/social media if desired

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Unauthorized" on publish | Check PyPI trusted publishing config or API token |
| Build fails | Run `uv build` locally to debug |
| Version already exists | Update version in `pyproject.toml` |
| Workflow doesn't run | Check `.github/workflows/publish.yml` syntax |
| Tests fail in CI | Ensure `tests/` directory exists or remove from workflow |

## Useful Commands

```bash
# Build locally
uv build

# Clean build artifacts
rm -rf dist/ build/ *.egg-info

# Check package metadata
python -m build --sdist --wheel --outdir dist/
twine check dist/*

# Publish manually (requires twine)
twine upload dist/*
```

## References

- [PyPI Help](https://pypi.org/help/)
- [hatchling Documentation](https://hatch.pypa.io/)
- [GitHub Actions Publishing](https://docs.github.com/en/actions/publishing-packages)
