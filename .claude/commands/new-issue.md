---
description: Create a new GitHub issue using the gh CLI
allowed-tools: Bash(gh issue:*), AskUserQuestion, Task
argument-hint: <issue title or description>
---

## Task

Create a new GitHub issue on the repository using the `gh` CLI based on the user's request.

## Instructions

1. **Parse the user's request**: Extract the issue title and any description details from their input.

2. **Search for related issues and PRs** (REQUIRED): Use the Task tool to invoke the `issue-and-pr-search` agent to find existing issues or PRs related to this topic. This step is mandatory - never skip it.
   - Search using keywords from the user's request
   - Note any issues this new issue might depend on, block, or relate to
   - Note any PRs that addressed similar concerns
   - These will be linked in the new issue body

3. **Interview the user**: Before creating the issue, ask clarifying questions to gather necessary details:
   - What problem does this solve or what value does it add?
   - Are there specific implementation constraints or preferences?
   - What are the key acceptance criteria from the user's perspective?
   - Present the related issues/PRs found and ask if any should be linked

   Use the AskUserQuestion tool to efficiently gather this information. Skip questions where the user has already provided clear answers.

4. **Create the issue using gh CLI**: Use `gh issue create` with appropriate flags:
   - `--title` for the issue title
   - `--body` for the issue description (use a HEREDOC for multi-line content)
   - Add `--label` flags if the user specifies labels
   - Add `--assignee` if the user specifies an assignee

5. **Issue body template**: Structure the body with:

```markdown
## Description

<Expand on the user's request into a clear problem statement or feature description>

## Related

- Depends on: #X, #Y (or None)
- Related to: #Z, #W (or None)
- See also: #PR-number (if relevant PRs exist)

## Acceptance Criteria

- [ ] <Specific, testable criterion 1>
- [ ] <Specific, testable criterion 2>
- [ ] <Add more as needed>

## Notes

<Any additional context, constraints, or implementation hints>
```

6. **Example command**:

```bash
gh issue create --title "Add retry logic to LLM client" --body "$(cat <<'EOF'
## Description

The LLM client should automatically retry failed requests with exponential backoff to handle transient API errors gracefully.

## Related

- Depends on: None
- Related to: #3 (rate limiting implementation)
- See also: #12 (previous retry discussion)

## Acceptance Criteria

- [ ] Implement exponential backoff with configurable max retries
- [ ] Retry on 429 (rate limit) and 5xx errors only
- [ ] Add tests for retry behavior

## Notes

Consider integrating with the existing ResourceManager rate limiting.
EOF
)"
```

7. **Confirm creation**: The `gh` command will output the issue URL. Share this with the user.

## User Request

$ARGUMENTS
