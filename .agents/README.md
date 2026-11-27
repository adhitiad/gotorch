# Portable Agent Skills

Reusable GoTorch task skills live in `.agents/skills/<skill-name>/SKILL.md`.

Each skill has YAML frontmatter with a stable `name` and short `description`, followed by a tool-agnostic Markdown procedure. Skills are designed to be read by different coding-agent runtimes; they must not assume a specific IDE, model provider or tool API.

The repository-wide instruction entry point is `AGENTS.md`. The shared roadmap and task state is `docs/TASK.md`. Skill files explain how to perform a type of task; they do not own project status or override the repository instructions.

When a skill is changed, update only its canonical file under this directory. Do not create divergent vendor-specific copies.
