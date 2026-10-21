# run_process

> 12 nodes · cohesion 0.21

## Key Concepts

- **run_process()** (14 connections) — `scripts/collab.py`
- **executable()** (11 connections) — `scripts/collab.py`
- **usd_floor_cents()** (5 connections) — `scripts/collab.py`
- **claude_command()** (5 connections) — `scripts/semantic_label_pilot.py`
- **kill_tree()** (4 connections) — `scripts/collab.py`
- **invoke()** (4 connections) — `scripts/logo_duo_pilot.py`
- **invoke()** (4 connections) — `scripts/logo_robustness_pilot.py`
- **subprocess_version()** (4 connections) — `scripts/semantic_label_pilot.py`
- **subprocess** (4 connections)
- **os** (3 connections)
- **.test_timeout_preserves_diagnostics()** (2 connections) — `tests/test_collab.py`
- **.test_claude_command_is_read_only_and_enforces_budget()** (2 connections) — `tests/test_semantic_label_pilot.py`

## Relationships

- [semantic_label_pilot.py](semantic_label_pilot.py.md) (3 shared connections)
- [logo_duo_pilot.py](logo_duo_pilot.py.md) (1 shared connections)

## Source Files

- `scripts/collab.py`
- `scripts/logo_duo_pilot.py`
- `scripts/logo_robustness_pilot.py`
- `scripts/semantic_label_pilot.py`
- `tests/test_collab.py`
- `tests/test_semantic_label_pilot.py`

## Audit Trail

- EXTRACTED: 16 (89%)
- INFERRED: 2 (11%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [index](index.md) to navigate.*