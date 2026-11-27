# Connecting Coding Agents to GoTorch

This repository uses a **shared instruction contract** and **portable project skills**. The goal is for different coding-agent runtimes to understand the same project scope, task IDs, source layout, branch rules and verification commands.

## Shared source of truth

Every agent working in this repository should use:

1. `AGENTS.md` — root-level repository instructions.
2. `docs/TASK.md` — canonical P0–P13 descriptions, dependencies, Task IDs, checklists and acceptance criteria.
3. `docs/CORE_CONTRACTS.md` and `docs/ARCHITECTURE.md` — implementation boundaries.
4. `.agent/rules/` — domain-specific engineering constraints.
5. `.agents/skills/<skill>/SKILL.md` — reusable, on-demand implementation/review skills.
6. `CONTRIBUTING.md` — contributor and PR workflow.

The instruction files do **not** grant repository access or GitHub write permissions. The agent runtime must have the repository checked out/opened as its workspace, and its user/operator must separately authorize tools and credentials used to create branches, commits or pull requests.

## Agent compatibility

| Agent runtime | Project instruction discovery | Project skills | Setup |
|---|---|---|---|
| OpenAI Codex | Reads root and nested `AGENTS.md` files | `.agents/skills/` | Open the repository root as the Codex workspace. Start a new task/session from the repo so it loads current guidance. |
| OpenClaw | Uses project/workspace guidance; `AGENTS.md` is the common repo entry point | `.agents/skills/` is a documented project-agent skill root | Set the GoTorch checkout as the active workspace. Check that project skills are enabled in the OpenClaw skill configuration. |
| OpenCode | Reads root/nested `AGENTS.md` | `.agents/skills/` is a project compatibility skill source | Open the repository root as the workspace. Skill permission rules can hide or ask before loading skills. |
| Hermes Agent | Supports project `AGENTS.md` generation/instructions via `/init` and reads project context | Hermes defaults to profile-local `~/.hermes/skills/`; the repo skills directory can be added as an external skill directory | Configure `skills.external_dirs` as shown below. |
| GitHub Copilot | `.github/copilot-instructions.md` | Agent Skills support varies by host/product surface | The committed Copilot instruction file points back to the same `AGENTS.md` and `docs/TASK.md`. |

Tool behavior and discovery options can change between releases. Verify with the runtime's current documentation and a small discovery test rather than assuming every tool loads every file automatically.

## Hermes setup

Hermes' default skill library is profile-local. To let Hermes discover the project skills from a GoTorch checkout, add the checkout's absolute `.agents/skills` path under `skills.external_dirs` in `~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - /absolute/path/to/gotorch/.agents/skills
```

Replace the example path with the actual clone path on that machine. Keep existing Hermes configuration keys; merge this entry rather than overwriting the entire file. On Windows, use a valid absolute path accepted by the installed Hermes version, or configure the repository under a WSL/Linux workspace and use its Linux path.

Hermes can create or update a repository `AGENTS.md` via `/init`; this project already provides one. Do not replace it with generated content without reviewing and preserving its GoTorch-specific rules.

## OpenCode and OpenClaw

Both current runtimes document project skills under `.agents/skills/`. The directory contains six skills with required `name` and `description` frontmatter:

- `gotorch-add-cpu-kernel`
- `gotorch-change-public-api`
- `gotorch-distributed-feature`
- `gotorch-implement-phase`
- `gotorch-repository-maintenance`
- `gotorch-review-change`

The folder name is a stable skill ID. Avoid copying these into another repository directory with different content. If a runtime needs a different location, configure it to read this canonical directory where supported.

## First-run verification checklist

- [ ] Open the GoTorch repository root in the selected coding agent.
- [ ] Ask the agent to summarize its active project instructions and confirm it references `AGENTS.md`.
- [ ] Ask it to identify `docs/TASK.md` as the canonical task plan.
- [ ] Ask it to list the skills it discovered; verify the six GoTorch skill IDs.
- [ ] Ask it to locate `src/tensor/metadata.go` and confirm the import path rule `github.com/adhitiad/gotorch/src/...`.
- [ ] Ask it to identify the active development branch `adhitiad_gpt` and stable branch `main`.
- [ ] Confirm repository write tools and GitHub credentials are available before asking it to commit or push.
- [ ] Confirm it does not edit `main` without explicit authorization.

## Non-goals

- No agent-specific duplicate task plans.
- No agent can skip dependency or acceptance gates by choosing a different model/provider.
- No skill file grants shell, filesystem, network, GitHub or deployment permissions.
- No agent may treat an interface or TODO as a completed framework feature.
