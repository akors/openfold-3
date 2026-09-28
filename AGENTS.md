# OpenFold-3 Agent Guidelines

## Before You Begin

When a task requires locating or understanding OpenFold3 code (e.g. "where does X happen", "what owns Y"), consult [CODEBASE_MAP.md](CODEBASE_MAP.md) instead of scanning the repository. It documents the end-to-end flow, key modification points, and ownership boundaries in the codebase. Skip it for tasks that don't need that context.

## Code Changes Policy

**Avoid changing code.** When code changes are required:

1. Minimize changes first—identify the smallest diff that solves the problem
2. Explain what will change and why
3. Ask for explicit permission before making modifications

This ensures code changes are intentional, reviewed, and aligned with project needs.

## Missing Tools

If a required tool (e.g. `pixi`) is not installed or not on `PATH`, ask the user to install it manually instead of
searching for it or installing it yourself.

## Code Style

- NumPy-style docstrings
- 120-character line width preferred (allow lines over 120 when it improves readability)
- Concise code: as simple as possible, but no simpler
- Follow Zen of Python principles

## Commits

When committing on the user's behalf, identify yourself with a commit trailer:

```bash
git commit --trailer "Co-Authored-By: <agent name> <model name> <noreply email>" ...
```

For example `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. If the model name is unavailable, give the agent name alone
(e.g. `Co-Authored-By: Copilot <noreply email>`). Use your vendor's noreply address.

## Build and Test

See [docs/BUILD_TESTING.md](docs/BUILD_TESTING.md) for test conventions and environment setup using Pixi.
