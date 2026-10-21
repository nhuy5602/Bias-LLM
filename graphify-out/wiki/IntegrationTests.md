# IntegrationTests

> 13 nodes · cohesion 0.23

## Key Concepts

- **IntegrationTests** (32 connections) — `tests/test_collab.py`
- **invoke_cli()** (15 connections) — `scripts/collab.py`
- **validate_budget_extension()** (5 connections) — `scripts/collab.py`
- **.test_budget_extension_allows_raising_global_launch_ceiling_only()** (2 connections) — `tests/test_collab.py`
- **.test_budget_extension_can_remove_all_quota_caps()** (2 connections) — `tests/test_collab.py`
- **.test_budget_extension_only_raises_limits_without_resetting_accounting()** (2 connections) — `tests/test_collab.py`
- **.test_claude_budget_defaults_to_two_dollars()** (2 connections) — `tests/test_collab.py`
- **.test_claude_request_budget_floors_remaining_total()** (2 connections) — `tests/test_collab.py`
- **.test_claude_request_budget_uses_remaining_member_cap()** (2 connections) — `tests/test_collab.py`
- **.test_claude_unlimited_mode_omits_native_usd_flag()** (2 connections) — `tests/test_collab.py`
- **.test_codex_adapter_preserves_provider_and_does_not_quote_keys()** (2 connections) — `tests/test_collab.py`
- **.test_runner_operates_without_launch_or_elapsed_time_caps()** (2 connections) — `tests/test_collab.py`
- **.test_duo_refuses_to_take_over_three_agent_ledger()** (1 connections) — `tests/test_collab.py`

## Relationships

- [invoke](invoke.md) (7 shared connections)
- [.test_duo_fresh_run_uses_only_codex_and_gemini](test_duo_fresh_run_uses_only_codex_and_gemini.md) (6 shared connections)
- [main](main.md) (6 shared connections)
- [collab.py](collab.py.md) (5 shared connections)
- [run_process](run_process.md) (3 shared connections)
- [semantic_label_pilot.py](semantic_label_pilot.py.md) (1 shared connections)
- [SchedulerTests](SchedulerTests.md) (1 shared connections)

## Source Files

- `scripts/collab.py`
- `tests/test_collab.py`

## Audit Trail

- EXTRACTED: 37 (79%)
- INFERRED: 10 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*