# TASKS.md

A line-by-line breakdown of PRs to implement plait, in order.

Each PR represents a single, tested, reviewable increment of functionality.

---

## PR Requirements

Every PR must:
- [ ] Create a feature branch from `main`
- [ ] Include implementation code
- [ ] Include unit tests (100% coverage of new code)
- [ ] Include integration tests where applicable
- [ ] Update `CHANGELOG.md`
- [ ] Pass `make ci`
- [ ] Include usage examples in docstrings or tests

---

## Workflow

For each PR:

```bash
# 1. Create branch
git checkout main
git pull
git checkout -b feat/feature-name

# 2. Read referenced design docs
# Review the Design Docs sections listed in the PR

# 3. Implement
# ... write code ...

# 4. Test
make ci

# 5. Update CHANGELOG
# Add entry under [Unreleased]

# 6. Open new issues for discovered work
# If you discover bugs, TODOs, or follow-up work during implementation,
# use /new-issue to create tracking issues

# 7. Commit
git add .
git commit -m "feat: description"

# 8. Push and create PR
git push -u origin feat/feature-name
gh pr create --title "PR-XXX: Description" --body "..."

# 9. After review and merge
git checkout main
git pull

# 10. Close the issue
# If this PR was fixing an issue, suggest /close-issue to close it
```
