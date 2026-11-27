# Agent Rules and Workflows

This directory contains GoTorch-specific engineering rules, reusable workflows and templates.

- `rules/`: mandatory engineering constraints by domain.
- `workflows/`: repeatable multi-step procedures for issue-to-implementation and release readiness.
- `templates/`: implementation-plan and review templates.

Portable reusable agent skills live in the standard project location `.agents/skills/<skill>/SKILL.md`. The root `AGENTS.md` is the common entry point and `docs/TASK.md` is the canonical phase checklist.

Do not keep duplicate skill copies here. If a skill changes, update its one canonical file under `.agents/skills/`.
