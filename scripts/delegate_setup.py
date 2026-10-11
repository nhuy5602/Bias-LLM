"""One bounded setup delegation: each external colleague owns one code review."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collab import ROOT, Runner, invoke_cli

folder = ROOT / 'collab/setup-review'
source = (ROOT / 'scripts/collab.py').read_text(encoding='utf-8')
tests = (ROOT / 'tests/test_collab.py').read_text(encoding='utf-8')
common = '''The user explicitly delegates framework implementation work to you. Review and propose a concrete unified diff; do not call models or edit files yourself. Your stdout becomes your deliverable and the integrator applies it. Be concise, at most 1800 words. No approval questions. Cite exact bugs, fix them, and give meaningful regression tests. You may use only supplied source. No need to invoke skills/tools. No secrets are supplied.\n'''
tasks = {
 'claude': '''You OWN the scheduler/budget/resume portion. Audit Runner, time accounting, global and per-task attempts, concurrent failure handling, process timeout and state consistency. Return a minimal unified diff fixing material issues and regression tests. Do not modify CLI command adapters.\n''',
 'gemini': '''You OWN CLI adapters and output parsing. Windows subprocess shell=False; quoted TOML dotted key overrides produce invalid transport in mcp_servers."node_repl" (quotes became part of the key). Solve this without changing global config/model/provider/authentication and without unsafe sandbox bypass. Claude uses tools disabled due to cost; Gemini is in plan mode with --skip-trust. Check stdout error envelopes and log recording. Propose a minimal unified diff and regression tests, no changes to scheduler.\n'''
}
r = Runner(folder, invoke_cli, max_calls=4, retries=1, timeout=180, max_seconds=240, backoff=2)
with ThreadPoolExecutor(max_workers=2) as pool:
    fs = {m:pool.submit(r.task, 'setup-'+m, m, common+t+'\nSOURCE:\n'+source+'\nTESTS:\n'+tests) for m,t in tasks.items()}
    for member, future in fs.items():
        print(member, 'DONE' if future.result() else 'FAILED', flush=True)

