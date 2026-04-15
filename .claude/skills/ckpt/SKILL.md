---
name: ckpt
description: Checkpoint the project — sync docs and commit all changes. Use when the user runs /ckpt or asks to checkpoint/commit/sync docs.
---

# ckpt

Checkpoint the project: sync docs and commit everything.

1. Run `git status` and `git diff` to see all pending changes.
2. Update `CLAUDE.md` if the architecture overview, file layout, API, or current status is out of date.
3. Update any affected files in `docs/` as relevant to the changes.
4. Stage all modified tracked source files (never use `git add -A` — follow CLAUDE.md git rules: no `CoACD/`, `compare_output/`, `octocat_output/`, `tmpcompare/`, `*.ncu-rep`, or build artifacts). Stage by explicitly naming files from `git status`.
5. Commit with a concise message describing what changed, then push.
