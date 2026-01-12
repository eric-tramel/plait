# Release Process

This document describes the release process for plait, including versioning conventions, the release checklist, and recovery procedures.

## Versioning Scheme

plait uses **Calendar Versioning (CalVer)** with the format:

```
YYYY.MM.MICRO
```

### Version Components

| Component | Description | Example |
|-----------|-------------|---------|
| `YYYY` | Four-digit year of the release | `2025` |
| `MM` | Month of the initial feature release (1-12, no leading zero) | `1` |
| `MICRO` | Patch number within the month, starting at 0 | `0`, `1`, `2` |

### Version Examples

- `2025.1.0` - First release of January 2025
- `2025.1.1` - First patch to the January 2025 release
- `2025.2.0` - First release of February 2025
- `2025.12.0` - First release of December 2025

### Version Semantics

1. **New feature releases** increment the month: `2025.1.0` -> `2025.2.0`
2. **Bug fixes and patches** increment the micro version: `2025.1.0` -> `2025.1.1`
3. **Year changes** are automatic when releasing in a new calendar year

This format is [PEP 440](https://peps.python.org/pep-0440/) compliant and provides clear age indication for packages.

## Release Checklist

Follow these steps to create a new release:

### 1. Prepare the Release

- [ ] Ensure all changes are merged to `main`
- [ ] Verify CI passes on `main`: `gh run list --branch main`
- [ ] Review unreleased changes in `CHANGELOG.md`

### 2. Update Version and Changelog

```bash
# Update version in pyproject.toml
# Change: version = "YYYY.MM.MICRO"

# Update CHANGELOG.md:
# - Change [Unreleased] to [YYYY.MM.MICRO] - YYYY-MM-DD
# - Add new empty [Unreleased] section at top
```

### 3. Commit Version Bump

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v2025.1.0"
```

### 4. Create and Push Tag

```bash
# Create annotated tag
git tag -a v2025.1.0 -m "Release v2025.1.0"

# Push commit and tag
git push origin main
git push origin v2025.1.0
```

### 5. Verify Release Pipeline

1. **Check GitHub Actions**: Navigate to Actions tab, verify the release workflow starts
2. **Verify TestPyPI**: Check [TestPyPI](https://test.pypi.org/project/plait/) for the new version
3. **Verify PyPI**: Check [PyPI](https://pypi.org/project/plait/) for the new version

### 6. Verify Installation

```bash
# Test installation from PyPI
pip install plait==2025.1.0

# Verify version
python -c "import plait; print(plait.__version__)"
```

### 7. Create GitHub Release

```bash
gh release create v2025.1.0 \
  --title "v2025.1.0" \
  --notes "See [CHANGELOG.md](CHANGELOG.md) for details."
```

## Automated Publishing

The release workflow (`.github/workflows/release.yml`) automatically:

1. Runs all CI checks (lint, types, tests)
2. Builds source distribution and wheel using `hatchling`
3. Publishes to TestPyPI for validation
4. Publishes to PyPI using trusted publishing (OIDC)

### Trusted Publishing

plait uses [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/) for secure, tokenless publishing:

- No API tokens or secrets to manage
- Uses GitHub's OIDC identity for authentication
- Automatic digital attestations for published packages

## PyPI Trusted Publisher Setup

This is a one-time setup required before the first release.

### TestPyPI Setup

1. Go to [TestPyPI](https://test.pypi.org/) and sign in
2. Navigate to "Your projects" > "Publishing"
3. Add a new pending publisher:
   - **PyPI Project Name**: `plait`
   - **Owner**: `eric-tramel`
   - **Repository name**: `plait`
   - **Workflow name**: `release.yml`
   - **Environment name**: `testpypi`

### PyPI Setup

1. Go to [PyPI](https://pypi.org/) and sign in
2. Navigate to "Your projects" > "Publishing"
3. Add a new pending publisher:
   - **PyPI Project Name**: `plait`
   - **Owner**: `eric-tramel`
   - **Repository name**: `plait`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`

> **Note**: After the first successful publish, the "pending publisher" becomes a regular trusted publisher tied to your project.

## Rollback Procedures

### Failed TestPyPI Publish

If TestPyPI publishing fails:

1. **Check logs**: Review the GitHub Actions workflow logs for errors
2. **Fix the issue**: Common issues include:
   - Version already exists (bump the micro version)
   - Build failures (fix and re-tag)
3. **Delete the tag and re-release**:
   ```bash
   git tag -d v2025.1.0
   git push origin :refs/tags/v2025.1.0
   # Fix issues, then re-tag and push
   ```

### Failed PyPI Publish

If PyPI publishing fails after TestPyPI succeeds:

1. **Do NOT delete the tag** (TestPyPI already has this version)
2. **Check PyPI trusted publisher configuration**
3. **If the version was partially uploaded**: Contact PyPI support

### Yanking a Release

If a release has critical issues and was already published:

```bash
# Yank from PyPI (prevents new installs, existing installs unaffected)
pip index yank plait --version 2025.1.0

# Create a patch release with the fix
# Use the next micro version: 2025.1.1
```

### Reverting Changes

If you need to revert the release commit:

```bash
# Revert the version bump commit (creates new commit)
git revert HEAD

# Do NOT force-push or rewrite history on main
```

## Release Schedule

plait follows a flexible release schedule:

- **Feature releases**: When significant new functionality is ready
- **Patch releases**: As needed for bug fixes and security updates
- **No fixed cadence**: Quality over schedule

## Version in Code

The version is defined in `pyproject.toml`:

```toml
[project]
name = "plait"
version = "2025.1.0"
```

To access the version programmatically:

```python
from importlib.metadata import version
print(version("plait"))
```

## References

- [Python Packaging Versioning Guide](https://packaging.python.org/en/latest/discussions/versioning/)
- [PEP 440 - Version Identification](https://peps.python.org/pep-0440/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions PyPI Publish](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [CalVer](https://calver.org/)
