---
name: Always use uv run
description: User requires uv run for all python commands, never bare python/python3
type: feedback
---

Always use `uv run python` instead of bare `python` or `python3` for any command execution.

**Why:** Project uses uv for dependency management and the user explicitly corrected this.
**How to apply:** Every time you run a python script or one-liner via Bash, prefix with `uv run`.
