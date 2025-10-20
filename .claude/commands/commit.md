---
description: Analyze code changes and create a git commit with conventional commit message
---

You are about to create a git commit. Follow these steps carefully:

## Step 1: Analyze Changes

Run the following commands in parallel to understand the current state:
- `git status` - See all changed files
- `git diff --staged` - See staged changes (if any)
- `git diff` - See unstaged changes
- `git log -5 --oneline` - See recent commit messages for style reference

## Step 2: Review Changes

Carefully review all changes:
- Identify what was added, modified, or deleted
- Understand the purpose of each change
- Note any breaking changes
- Check for files that should NOT be committed (secrets, temporary files, etc.)

## Step 3: Categorize Changes

Determine the commit type based on Conventional Commits:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code formatting (no logic change)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Build process, dependencies, tooling
- `perf`: Performance improvement

Determine the scope (optional):
- `frontend`, `backend`, `training`, `mvp`
- Module names like `llm`, `websocket`, `trainer`
- Feature areas like `chat`, `monitoring`, `api`

## Step 4: Draft Commit Message

Create a commit message following this format:

```
<type>(<scope>): <subject>

<body>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Guidelines:**
- Subject: Clear, concise, imperative mood (max 50 chars)
- Body: Explain WHAT and WHY (not HOW)
- Focus on the user impact
- Mention issue numbers if applicable

**Examples:**

```
feat(mvp): add LLM-based intent parsing

Implement natural language to training config conversion using Claude API.
Supports classification tasks with ResNet50 model.

Closes #123
```

```
fix(backend): resolve training process not terminating

Training subprocess wasn't being cleaned up properly, causing zombie
processes. Added proper signal handling and timeout management.

Fixes #456
```

```
docs(mvp): add MVP implementation plan

Create detailed 2-week implementation roadmap with daily tasks and
deliverables. Includes simplified architecture and scope definition.
```

## Step 5: Stage Files

Stage the appropriate files. DO NOT stage:
- `.env` files
- Secrets or credentials
- Binary files unless intended
- `__pycache__`, `node_modules`
- Temporary or generated files

Ask the user which files to stage if unclear, or stage all relevant changes if obvious.

## Step 6: Create Commit

Use the following format for the git commit command:

```bash
git add <files>

git commit -m "$(cat <<'EOF'
<commit message here with proper formatting>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

## Step 7: Verify

After committing:
- Run `git log -1` to show the commit
- Run `git status` to confirm clean state
- Report success to the user

## Important Notes

- **NEVER commit sensitive data** (API keys, passwords, tokens)
- **DO NOT use `--amend`** unless explicitly requested
- **DO NOT push** unless the user explicitly asks
- If there are no changes, inform the user
- If pre-commit hooks fail, show the error and ask for guidance

## Example Workflow

```bash
# 1. Check changes
git status
git diff

# 2. Stage files
git add mvp/backend/app/main.py mvp/backend/app/config.py

# 3. Commit
git commit -m "$(cat <<'EOF'
feat(backend): add FastAPI application structure

Create main FastAPI app with health check endpoint and CORS middleware.
Add configuration management with pydantic-settings.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# 4. Verify
git log -1
git status
```

Now proceed with analyzing the current changes and creating the commit.
