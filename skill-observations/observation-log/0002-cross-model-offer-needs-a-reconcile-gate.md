---
id: 2
title: "Cross-model offer needs a hard gate before reconciliation"
status: open
type: open-source
skill: [doubt-driven-development]
proposes_skill: []
target_file: []
siblings_checked: "No registry exists; installed review-related skills were searched. Code review was excluded because doubt-driven-development/SKILL.md, Interaction with Other Skills, defines it as complementary post-hoc review rather than this in-flight adversarial cycle."
area: "Cross-model escalation sequence"
date: 2026-11-16
session_context: "An interactive feasibility assessment with a fresh-context single-model reviewer."
parked_until:
resolved:
resolution:
reference:
commands_verified: none
---

**Issue:** After the fresh-context reviewer returned, the orchestrator acknowledged and began classifying several findings before making the skill's mandatory cross-model second-opinion offer. The protection in play was written sequencing guidance only; no workflow state prevented reconciliation from starting early.

**Suggested improvement:** Make the reviewer-return handler enter an explicit cross-model-choice state that blocks reconciliation output until the user chooses a CLI, manual review, or skip. The transition should be visible and mechanically required, not an item remembered while reading a long review.

**Principle:** A user-choice checkpoint that must precede analysis needs a hard workflow transition; prose ordering alone is easy to cross while processing a dense result.
