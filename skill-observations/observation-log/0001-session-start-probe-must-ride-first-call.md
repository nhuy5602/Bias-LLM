---
id: 1
title: "Session-start storage probe must ride the first skill-load call"
status: open
type: open-source
skill: [task-observer]
proposes_skill: []
target_file: []
siblings_checked: "No registry exists; installed user, project, agent, and plugin skill names were searched and no task-observer sibling or family was found."
area: "Session Start Protocol and activation enforcement"
date: 2026-11-16
session_context: "An interactive research assessment session that loaded Task Observer before probing its pinned workspace."
parked_until:
resolved:
resolution:
reference:
commands_verified: none
---

**Issue:** The session loaded Task Observer in its first tool call, then probed the pinned observation workspace in a later call. This violated Session Start Protocol step 1 even though the rule was present and subsequently read. The protection in play was written guidance only; by definition, the agent cannot know the guidance before the first call that loads it.

**Suggested improvement:** Add a host-level activation hook or a one-shot session-start entrypoint that batches the skill load and pinned-workspace probe before ordinary task tooling becomes available. Keep the prose rule, but make the correct first call the default path rather than a condition discovered after that call.

**Principle:** Any obligation that must hold on the first tool call needs enforcement outside the instruction that the first call is loading; otherwise compliance depends on knowledge that becomes available too late.
