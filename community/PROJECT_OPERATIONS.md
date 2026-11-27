# Project Operations

## Repository administration
- Keep the default branch stable and protected.
- Require pull requests and successful CI before merging.
- Disable force pushes and branch deletion on protected branches.
- Prefer squash merges for focused contribution history.
- Enable Issues and Discussions; use repository documentation as the source of truth.
- Enable private vulnerability reporting and dependency/security alerts where available.

## Labels and milestones
Use a small, consistent label taxonomy. Avoid creating overlapping labels with different meanings. Milestones should represent deliverable releases or roadmap phase checkpoints with explicit exit criteria.

## GitHub Projects
Recommended views: Roadmap by phase, Kanban by status, Contributor queue, Quality & performance, Security (restricted), and Release readiness.

## Automation
Automate low-risk validation and stale metadata only when maintainers can understand and override it. Do not auto-close valid reports solely due to inactivity. Keep workflow permissions read-only by default.

## Repository settings limitation
This file documents desired configuration; it does not itself change GitHub repository settings, branch protection, labels, milestones, or Projects. Verify each setting in GitHub before claiming it is active.
