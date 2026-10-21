# SchedulerTests

> 17 nodes · cohesion 0.16

## Key Concepts

- **SchedulerTests** (17 connections) — `tests/test_collab.py`
- **parse_response()** (8 connections) — `scripts/collab.py`
- **.make()** (7 connections) — `tests/test_collab.py`
- **.test_auth_error_does_not_retry()** (3 connections) — `tests/test_collab.py`
- **.test_budget_and_workspace_trust_errors_do_not_retry()** (3 connections) — `tests/test_collab.py`
- **.test_changed_prompt_or_report_is_not_silently_reused()** (2 connections) — `tests/test_collab.py`
- **.test_cli_error_envelopes_fail_even_with_zero_exit()** (2 connections) — `tests/test_collab.py`
- **.test_completed_task_is_not_paid_twice()** (2 connections) — `tests/test_collab.py`
- **.test_global_budget_persists_across_resume()** (2 connections) — `tests/test_collab.py`
- **.test_max_three_retries_and_circuit_breaker()** (2 connections) — `tests/test_collab.py`
- **.test_nested_error_is_clear()** (2 connections) — `tests/test_collab.py`
- **.test_real_subprocess_stdin_and_timeout()** (2 connections) — `tests/test_collab.py`
- **.setUp()** (1 connections) — `tests/test_collab.py`
- **.success()** (1 connections) — `tests/test_collab.py`
- **auth()** (1 connections) — `tests/test_collab.py`
- **fail()** (1 connections) — `tests/test_collab.py`
- **.timeout_invoke()** (1 connections) — `tests/test_collab.py`

## Relationships

- [main](main.md) (3 shared connections)
- [run_process](run_process.md) (2 shared connections)
- [semantic_label_pilot.py](semantic_label_pilot.py.md) (1 shared connections)
- [logo_duo_pilot.py](logo_duo_pilot.py.md) (1 shared connections)
- [invoke](invoke.md) (1 shared connections)

## Source Files

- `scripts/collab.py`
- `tests/test_collab.py`

## Audit Trail

- EXTRACTED: 22 (73%)
- INFERRED: 8 (27%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*