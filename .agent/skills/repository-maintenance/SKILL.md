# Repository Maintenance Skill

## Purpose
Maintain GoTorch repository metadata, documentation, governance, and contributor infrastructure without changing framework runtime code unless explicitly requested.

## Before editing
1. Inspect the current branch, latest commit, and target files.
2. Read README.md, AGENTS.md, CONTRIBUTING.md, and relevant docs.
3. Identify existing conventions and avoid duplicate or destructive changes.
4. Confirm the requested target branch and whether source code is in scope.

## Workflow
1. Define a minimal change plan and acceptance criteria.
2. Make focused edits with stable relative links.
3. Validate Markdown structure, YAML syntax, paths, and cross-document consistency.
4. Verify all links to files exist where possible.
5. Commit using Conventional Commits.
6. Report exact paths, commit SHA, branch, and checks performed.

## Safety
- Never rewrite main history or force-push.
- Never overwrite existing content without reading and preserving relevant material.
- Do not invent maintainers, project status, releases, benchmark results, or CI outcomes.
- Do not claim repository settings or GitHub Project resources were changed unless the API confirms it.
- Do not modify Go source files when the task is documentation/governance-only.
