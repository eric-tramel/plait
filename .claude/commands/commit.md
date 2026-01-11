# Commit Changes

Create a commit for the current changes following project conventions.

## Instructions

1. **Check status and changes** - Run in parallel:
   - `git status` to see all untracked files
   - `git diff --stat` to see staged and unstaged changes
   - `git log --oneline -5` to see recent commit message style

2. **Analyze changes** and draft a commit message:
   - Summarize the nature of changes (new feature, enhancement, bug fix, refactoring, test, docs)
   - Do not commit files that likely contain secrets (.env, credentials.json, etc.)
   - Draft a concise commit message focusing on the "why" rather than the "what"

3. **Stage and commit**:
   - Add relevant files to staging
   - Create the commit with PR-style message format (see below)
   - Run `git status` after commit to verify success

## Commit Message Format

```
feat(scope): short description

## Summary
1-2 sentences explaining what this change does and why.

## Changes
- Bullet points of specific changes made
- Include new files, modified files, removed files

## Testing
- List test functions/classes added or updated
- Note any manual testing performed

## References
- Design: design_docs/xxx.md
- Task: PR-XXX in TASKS.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Commit Prefixes

- `feat(scope):` - New feature
- `fix(scope):` - Bug fix
- `refactor(scope):` - Code restructuring without behavior change
- `test(scope):` - Adding or updating tests
- `docs(scope):` - Documentation only
- `chore(scope):` - Build, tooling, or maintenance

## Important Notes

- Always run `make ci` before committing to ensure tests pass
- Use HEREDOC for multi-line commit messages to preserve formatting
- Never commit .env files, credentials, or secrets
- Squash multiple WIP commits before final commit
