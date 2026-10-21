import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collab import Runner, parse_response, run_process

class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
    def make(self, fn, **kw):
        return Runner(self.root, fn, max_calls=kw.get('max_calls', 12),
                      retries=3, timeout=1, max_seconds=30, backoff=0)
    def timeout_invoke(self, member, prompt, folder, timeout):
        self.calls.append(member)
        raise TimeoutError('test timeout')
    def success(self, member, prompt, folder, timeout):
        self.calls.append(member)
        return 'A research proposal with hypotheses and testable predictions. ' * 4
    def test_max_three_retries_and_circuit_breaker(self):
        r = self.make(self.timeout_invoke)
        self.assertIsNone(r.task('r1-codex', 'codex', 'prompt'))
        self.assertEqual(len(self.calls), 4)
        r.task('r2-codex', 'codex', 'prompt')
        self.assertEqual(len(self.calls), 4)
    def test_global_budget_persists_across_resume(self):
        self.make(self.timeout_invoke, max_calls=2).task('r1', 'codex', 'prompt')
        self.make(self.success, max_calls=2).task('r2', 'claude', 'prompt')
        self.assertEqual(len(self.calls), 2)
    def test_completed_task_is_not_paid_twice(self):
        self.make(self.success).task('r1', 'codex', 'prompt')
        self.make(self.timeout_invoke).task('r1', 'codex', 'prompt')
        self.assertEqual(len(self.calls), 1)
    def test_changed_prompt_or_report_is_not_silently_reused(self):
        self.make(self.success).task('r1', 'codex', 'prompt')
        with self.assertRaises(ValueError):
            self.make(self.success).task('r1', 'codex', 'changed prompt')
        (self.root / 'reports/r1.md').write_text('tampered')
        with self.assertRaises(ValueError):
            self.make(self.success).task('r1', 'codex', 'prompt')
    def test_auth_error_does_not_retry(self):
        def auth(*args):
            self.calls.append('auth')
            raise RuntimeError('401 invalid API key')
        self.make(auth).task('r1', 'gemini', 'prompt')
        self.assertEqual(len(self.calls), 1)

    def test_budget_and_workspace_trust_errors_do_not_retry(self):
        for message in ('Reached maximum budget ($1)', 'folder is not trusted'):
            with tempfile.TemporaryDirectory() as d:
                calls = []
                def fail(*args):
                    calls.append(1)
                    raise RuntimeError(message)
                Runner(Path(d), fail, backoff=0).task('x', 'claude', 'p')
                self.assertEqual(len(calls), 1)
    def test_real_subprocess_stdin_and_timeout(self):
        out, err, code = run_process(
            [sys.executable, '-c', 'import sys; print(sys.stdin.read())'],
            'Unicode: tiếng Việt', self.root, 5)
        self.assertIn('tiếng Việt', out)
        self.assertEqual(code, 0)
        with self.assertRaises(TimeoutError):
            run_process([sys.executable, '-c', 'import time; time.sleep(10)'],
                        '', self.root, 0.2)
    def test_cli_error_envelopes_fail_even_with_zero_exit(self):
        with self.assertRaises(RuntimeError):
            parse_response('claude', '{"is_error":true,"result":"quota exceeded"}')
        with self.assertRaises(RuntimeError):
            parse_response('gemini', '{"error":{"message":"invalid API key"}}')
        self.assertEqual(parse_response('gemini', '{"response":"hello"}'), 'hello')

    def test_nested_error_is_clear(self):
        with self.assertRaisesRegex(RuntimeError, '^quota exceeded$'):
            parse_response('gemini', '{"error":{"message":"quota exceeded"}}')

    def test_invalid_invoke_output_and_config_fail_once(self):
        for value in (None, ValueError('invalid transport in config')):
            with tempfile.TemporaryDirectory() as d:
                calls = []
                def invoke(*a):
                    calls.append(1)
                    if isinstance(value, Exception):
                        raise value
                    return value
                r = Runner(Path(d), invoke, max_calls=1, backoff=0)
                self.assertIsNone(r.task('x','codex','p'))
                self.assertEqual(len(calls), 1)

    def test_timeout_preserves_diagnostics(self):
        with self.assertRaises(TimeoutError) as cm:
            run_process([sys.executable, '-c', 'import time; print("partial", flush=True); time.sleep(10)'],
                        '', self.root, 0.3)
        self.assertIn('partial', cm.exception.stdout)


class IntegrationTests(unittest.TestCase):

    def test_budget_extension_only_raises_limits_without_resetting_accounting(self):
        from collab import validate_budget_extension
        old = {'rounds': 1, 'max_calls': 12, 'max_seconds': 500, 'member_max_calls': 4,
               'member_max_seconds': 480, 'member_max_usd': 2.0, 'sources': {'paper.pdf': 'abc'},
               'controller': 'old'}
        new = dict(old, max_seconds=1200, member_max_calls=6, member_max_seconds=960, controller='new')
        self.assertTrue(validate_budget_extension(old, new))
        self.assertFalse(validate_budget_extension(new, new))
        with self.assertRaises(ValueError):
            validate_budget_extension(old, dict(new, max_seconds=400))
        self.assertTrue(validate_budget_extension(old, dict(new, max_calls=13)))
        with self.assertRaises(ValueError):
            validate_budget_extension(old, dict(new, max_calls=11))
        with self.assertRaises(ValueError):
            validate_budget_extension(old, dict(new, member_max_usd=4.0))
        with self.assertRaises(ValueError):
            validate_budget_extension(old, dict(new, sources={'paper.pdf':'changed'}))

    def test_budget_extension_allows_raising_global_launch_ceiling_only(self):
        from collab import validate_budget_extension
        old = {'sources': {'p.pdf':'hash'}, 'max_calls':12, 'max_seconds':500, 'timeout':180,
               'member_max_calls':4, 'member_max_seconds':480, 'member_max_usd':2}
        raised = dict(old, max_calls=16, max_seconds=1200, timeout=600, member_max_calls=8, member_max_seconds=1200)
        self.assertTrue(validate_budget_extension(old, raised))
        with self.assertRaises(ValueError):
            validate_budget_extension(old, dict(raised, max_calls=11))

    def test_budget_extension_can_remove_all_quota_caps(self):
        from collab import validate_budget_extension
        old = {'sources': {'p.pdf':'hash'}, 'max_calls':16, 'max_seconds':1800,
               'timeout':180, 'member_max_calls':8, 'member_max_seconds':1600,
               'member_max_usd':2.0, 'unlimited_budget':False}
        unlimited = dict(old, max_calls=None, max_seconds=None, member_max_calls=None,
                         member_max_seconds=None, member_max_usd=None, unlimited_budget=True)
        self.assertTrue(validate_budget_extension(old, unlimited))
        with self.assertRaises(ValueError):
            validate_budget_extension(unlimited, dict(unlimited, member_max_calls=1000))

    def test_runner_operates_without_launch_or_elapsed_time_caps(self):
        with tempfile.TemporaryDirectory() as d:
            r = Runner(Path(d), lambda *a: 'A completed report with evidence and a falsifier. ' * 4,
                       max_calls=None, timeout=1, max_seconds=None,
                       member_max_calls=None, member_max_seconds=None, member_max_usd=None)
            self.assertIsNotNone(r.task('one-task', 'codex', 'prompt'))
            self.assertEqual(r.state['calls'], 1)
            self.assertTrue(r.remaining() == float('inf'))

    def test_claude_unlimited_mode_omits_native_usd_flag(self):
        from unittest.mock import patch
        from collab import invoke_cli
        with tempfile.TemporaryDirectory() as d:
            with patch('collab.executable', return_value=['claude']), \
                 patch('collab.run_process', return_value=('{"result":"ok"}', '', 0)) as proc:
                invoke_cli('claude', 'prompt', Path(d), 10, unlimited_budget=True)
                cmd = proc.call_args.args[0]
                self.assertNotIn('--max-budget-usd', cmd)

    def test_changed_synthesis_prompt_uses_a_fresh_reconciliation_task(self):
        from collab import task_name_for_prompt, digest
        tasks = {'synthesis-reconciled-codex': {'prompt_hash': digest('old peers')}}
        self.assertEqual(task_name_for_prompt(tasks, 'synthesis-reconciled-codex', 'new peers'),
                         'synthesis-reconciled-codex-v2')
        self.assertEqual(task_name_for_prompt(tasks, 'synthesis-reconciled-codex', 'old peers'),
                         'synthesis-reconciled-codex')

    def test_timeout_repair_task_preserves_exhausted_attempt_history(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'Problem/source.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'repair source')
            source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            context = '===== paper.pdf PAGE 5 =====\ncontext'
            role_map = collab.get_owners(0)
            parent_id='20261012-050000-000001'
            parent=root/'collab/runs'/parent_id
            (parent/'reports').mkdir(parents=True)
            parent_tasks={}
            parent_reports={}
            for m in collab.MEMBERS:
                name=f'round-04-{m}'
                report=f'Parent proposal from {m}. '*6
                parent_reports[m]=report
                (parent/'reports'/f'{name}.md').write_text(report,encoding='utf-8')
                parent_tasks[name]={'status':'DONE','output_hash':collab.digest(report)}
            (parent/'state.json').write_text(json.dumps({'status':'COMPLETED','tasks':parent_tasks}),encoding='utf-8')
            rid='20261012-050001-000001'
            run=root/'collab/runs'/rid
            (run/'inputs').mkdir(parents=True)
            (run/'reports').mkdir()
            (run/'inputs/context.txt').write_text(context,encoding='utf-8')
            peers='\n\n'.join(f'### {m} proposal (round 4)\n{parent_reports[m][:collab.MAX_PEER_CHARS]}'
                                for m in collab.MEMBERS)
            old_prompt=collab.make_prompt('gemini',5,context,peers,role_name='novelty',role_map=role_map)
            codex_prompt=collab.make_prompt('codex',5,context,peers,role_name='pilot',role_map=role_map)
            claude_prompt=collab.make_prompt('claude',5,context,peers,role_name='evidence_gate',role_map=role_map)
            reports={}; tasks={}
            for m,prompt in [('codex',codex_prompt),('claude',claude_prompt)]:
                name=f'round-05-{m}'; report=f'Round five report by {m}. '*8
                reports[m]=report
                (run/'reports'/f'{name}.md').write_text(report,encoding='utf-8')
                tasks[name]={'status':'DONE','attempts':1,'prompt_hash':collab.digest(prompt),
                             'output_hash':collab.digest(report)}
            tasks['round-05-gemini']={'status':'FAILED','attempts':4,'prompt_hash':collab.digest(old_prompt),
                                      'error':'CLI timeout after 180s'}
            original_attempt = run/'logs'/'round-05-gemini'/'4'
            original_attempt.mkdir(parents=True)
            (original_attempt/'prompt.txt').write_text(old_prompt,encoding='utf-8')
            signature={'sources':{pdf.name:source_hash},'rounds':1,'max_calls':12,'timeout':180,
                'max_seconds':1200,'member_max_calls':8,'member_max_seconds':1200,'member_max_usd':2,
                'rotate_roles':False,'section_failover':False,'controller':'old',
                'continue_from':parent_id,'continuation_round':5,'role_map':role_map}
            (run/'manifest.json').write_text(json.dumps({'signature':signature,'context_hash':collab.digest(context)}),encoding='utf-8')
            (run/'state.json').write_text(json.dumps({'calls':6,'elapsed':500,'status':'DEGRADED',
                'scientific_status':'NEEDS_EVIDENCE','tasks':tasks,'member_stats':{
                    'codex':{'calls':1,'attempts':1,'seconds':20,'cost_usd':0,'tasks':1},
                    'claude':{'calls':1,'attempts':1,'seconds':20,'cost_usd':1.84,'tasks':1},
                    'gemini':{'calls':4,'attempts':4,'seconds':480,'cost_usd':0,'tasks':0}},
                'unavailable':{'gemini':'CLI timeout after 180s'},'role_map':role_map}),encoding='utf-8')
            (root/'collab/latest.json').write_text(json.dumps({'id':rid}),encoding='utf-8')
            seen=[]
            def invoke(member,prompt,folder,timeout,**kwargs):
                seen.append((member,timeout,kwargs))
                if member=='gemini':
                    return 'Gemini recovered its failed report with counterexample and falsifier. '*6
                return 'New synthesis with audit and unresolved evidence. '*6
            with patch.object(collab,'ROOT',root), patch.object(collab,'invoke_cli',invoke), \
                 patch.object(collab.signal,'signal'), patch.object(sys,'argv',[
                    'collab.py','--repair-gemini','--extend-budget','--max-rounds','1',
                    '--max-calls','16','--member-max-calls','8','--member-max-seconds','1200',
                    '--member-max-usd','2','--timeout','600','--max-seconds','1800']):
                self.assertEqual(collab.main(),2)
            self.assertEqual(seen[0][0],'gemini')
            self.assertEqual(seen[0][1],600)
            final=json.loads((run/'state.json').read_text(encoding='utf-8'))
            self.assertEqual(final['tasks']['round-05-gemini']['attempts'],4)
            self.assertEqual(final['tasks']['round-05-gemini']['status'],'DONE')
            self.assertIn('round-05-gemini-repair-01',final['tasks'])
            self.assertEqual(final['member_stats']['claude']['calls'],1)
            self.assertEqual(final['member_stats']['claude']['cost_usd'],1.84)
            self.assertEqual(final['status'],'DEGRADED')

    def test_targeted_gemini_repair_and_claude_retry_keep_each_budget(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); pdf=root/'Problem/source.pdf'; pdf.parent.mkdir(); pdf.write_bytes(b'repair source')
            source_hash=hashlib.sha256(pdf.read_bytes()).hexdigest()
            context='===== paper.pdf PAGE 5 =====\ncontext'
            role_map=collab.get_owners(0)
            parent_id='20261012-050000-000003'; parent=root/'collab/runs'/parent_id
            (parent/'reports').mkdir(parents=True); parent_tasks={}; parent_reports={}
            for m in collab.MEMBERS:
                name=f'round-04-{m}'; report=f'Round four proposal by {m}. '*8
                parent_reports[m]=report
                (parent/'reports'/f'{name}.md').write_text(report,encoding='utf-8')
                parent_tasks[name]={'status':'DONE','output_hash':collab.digest(report)}
            (parent/'state.json').write_text(json.dumps({'status':'COMPLETED','tasks':parent_tasks}),encoding='utf-8')
            rid='20261012-050001-000003'; run=root/'collab/runs'/rid
            (run/'inputs').mkdir(parents=True); (run/'reports').mkdir()
            (run/'inputs/context.txt').write_text(context,encoding='utf-8')
            peer_text='\n\n'.join(f'### {m} proposal (round 4)\n{parent_reports[m][:collab.MAX_PEER_CHARS]}'
                                    for m in collab.MEMBERS)
            tasks={}; stats={}
            for m in collab.MEMBERS:
                stats[m]={'calls':1,'attempts':1,'seconds':20.0,'cost_usd':0.0,'tasks':1}
                name=f'round-05-{m}'
                if m=='gemini':
                    prompt=collab.make_prompt(m,5,context,peer_text,role_name=role_map[m],role_map=role_map)
                    (run/'logs'/name/'4').mkdir(parents=True)
                    (run/'logs'/name/'4'/'prompt.txt').write_text(prompt,encoding='utf-8')
                    tasks[name]={'status':'FAILED','attempts':4,'prompt_hash':collab.digest(prompt),'error':'timeout'}
                    stats[m].update(calls=4,attempts=4,seconds=721.0,tasks=0)
                else:
                    prompt=collab.make_prompt(m,5,context,peer_text,role_name=role_map[m],role_map=role_map)
                    report=f'Round five report by {m}. '*8
                    (run/'reports'/f'{name}.md').write_text(report,encoding='utf-8')
                    tasks[name]={'status':'DONE','attempts':1,'prompt_hash':collab.digest(prompt),'output_hash':collab.digest(report)}
                if m=='claude':
                    tasks[name]['status']='FAILED'
                    stats[m].update(calls=1,attempts=1,seconds=30.0,cost_usd=0.93,tasks=0)
            signature={'sources':{pdf.name:source_hash},'rounds':1,'max_calls':12,'timeout':180,
                'max_seconds':1200,'member_max_calls':8,'member_max_seconds':1200,'member_max_usd':2.0,
                'rotate_roles':False,'section_failover':False,'controller':'old',
                'continue_from':parent_id,'continuation_round':5,'role_map':role_map}
            (run/'manifest.json').write_text(json.dumps({'signature':signature,'context_hash':collab.digest(context)}),encoding='utf-8')
            (run/'state.json').write_text(json.dumps({'calls':6,'elapsed':800.0,'status':'DEGRADED',
                'tasks':tasks,'member_stats':stats,
                'unavailable':{'gemini':'timeout','claude':'Reached maximum budget ($0.8)'},
                'role_map':role_map,'continuation_from':parent_id,'continuation_round':5}),encoding='utf-8')
            (root/'collab/latest.json').write_text(json.dumps({'id':rid}),encoding='utf-8')
            seen=[]
            def invoke(member,prompt,folder,timeout,is_synthesis=False,max_budget_usd=None):
                seen.append((member,timeout,is_synthesis,max_budget_usd))
                return f'Recovered evidence report and falsifier from {member}. '*8
            with patch.object(collab,'ROOT',root), patch.object(collab,'invoke_cli',invoke), \
                 patch.object(collab.signal,'signal'), patch.object(sys,'argv',[
                    'collab.py','--extend-budget','--repair-gemini','--max-rounds','1','--max-calls','16',
                    '--member-max-calls','8','--member-max-seconds','1600','--member-max-usd','2',
                    '--timeout','240','--max-seconds','1800']):
                self.assertEqual(collab.main(),0)
            self.assertEqual(seen[0][:2],('gemini',600))
            self.assertIsNone(seen[0][3])
            claude_proposal=next(x for x in seen if x[0]=='claude' and not x[2])
            self.assertEqual(claude_proposal[3],1.07)
            final=json.loads((run/'state.json').read_text(encoding='utf-8'))
            self.assertEqual(final['tasks']['round-05-gemini']['attempts'],4)
            self.assertEqual(final['tasks']['round-05-gemini']['status'],'DONE')
            self.assertEqual(final['tasks']['round-05-claude']['status'],'DONE')
            self.assertEqual(final['member_stats']['claude']['cost_usd'],0.93)
            self.assertTrue(any(name.startswith('synthesis-reconciled-') for name in final['tasks']))
            self.assertIn('synthesis-reconciled-gemini',final['tasks'])

    def test_extend_budget_resumes_failed_task_without_replaying_done_calls(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'Problem/source.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'fixed source')
            source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            run = root / 'collab/runs/20261012-020000-000001'
            (run / 'inputs').mkdir(parents=True)
            (run / 'reports').mkdir()
            context = '===== paper.pdf PAGE 5 =====\ncontext'
            (run / 'inputs/context.txt').write_text(context, encoding='utf-8')
            sig = {
                'sources': {pdf.name: source_hash}, 'rounds': 1, 'max_calls': 12,
                'timeout': 180, 'max_seconds': 500, 'member_max_calls': 4,
                'member_max_seconds': 480, 'member_max_usd': 2.0,
                'rotate_roles': False, 'section_failover': False, 'controller': 'old'
            }
            role_map = collab.get_owners(0)
            tasks = {}
            stats = {}
            for member in collab.MEMBERS:
                prompt = collab.make_prompt(member, 1, context, '', role_map=role_map)
                name = f'round-01-{member}'
                tasks[name] = {'prompt_hash': collab.digest(prompt), 'attempts': 1}
                stats[member] = {'calls': 1, 'attempts': 1, 'seconds': 10.0,
                                 'cost_usd': 0.0, 'tasks': 1}
                if member == 'gemini':
                    tasks[name].update(status='FAILED', error='transient timeout', attempts=3)
                else:
                    text = f'Complete saved proposal by {member}. ' * 5
                    (run / 'reports' / f'{name}.md').write_text(text, encoding='utf-8')
                    tasks[name].update(status='DONE', output_hash=collab.digest(text))
            (run / 'manifest.json').write_text(json.dumps({'signature': sig, 'context_hash': collab.digest(context)}), encoding='utf-8')
            (run / 'state.json').write_text(json.dumps({
                'calls': 5, 'elapsed': 500.0, 'status': 'BUDGET_EXHAUSTED',
                'scientific_status': 'NEEDS_EVIDENCE', 'tasks': tasks,
                'member_stats': stats, 'unavailable': {}, 'role_map': role_map
            }), encoding='utf-8')
            (root / 'collab/latest.json').write_text(json.dumps({'id': run.name}), encoding='utf-8')
            invoked = []

            def fake_invoke(member, prompt, folder, timeout, is_synthesis=False):
                invoked.append((member, is_synthesis))
                return f'Resumed work from {member}. Audit and synthesis evidence. ' * 5

            with patch.object(collab, 'ROOT', root), patch.object(collab, 'invoke_cli', fake_invoke), \
                 patch.object(collab.signal, 'signal'), \
                 patch.object(sys, 'argv', ['collab.py', '--extend-budget', '--max-rounds', '1',
                     '--max-calls', '12', '--member-max-calls', '6', '--member-max-seconds', '960',
                     '--member-max-usd', '2', '--timeout', '180', '--max-seconds', '1200']):
                self.assertEqual(collab.main(), 0)

            self.assertEqual(sum(1 for _, synthesis in invoked if synthesis), 3)
            self.assertEqual(len(invoked), 4)  # Gemini's unfinished proposal + three syntheses.
            final = json.loads((run / 'state.json').read_text(encoding='utf-8'))
            self.assertGreaterEqual(final['calls'], 9)
            self.assertEqual(final['member_stats']['codex']['calls'], 2)
            self.assertEqual(final['member_stats']['claude']['calls'], 2)
            self.assertEqual(final['member_stats']['gemini']['calls'], 3)
            manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['budget_extensions']), 1)

    def test_extend_budget_resumes_same_continuation_round(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'Problem/source.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'continuation source')
            source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            context = '===== paper.pdf PAGE 5 =====\nverified context'
            role_map = collab.get_owners(0)
            parent_id = '20261012-030000-000001'
            parent = root / 'collab/runs' / parent_id
            (parent / 'reports').mkdir(parents=True)
            parent_tasks = {}
            parent_reports = {}
            for member in collab.MEMBERS:
                name = f'round-03-{member}'
                report = f'Proposal from round three by {member}. ' * 8
                parent_reports[member] = report
                (parent / 'reports' / f'{name}.md').write_text(report, encoding='utf-8')
                parent_tasks[name] = {'status':'DONE', 'output_hash':collab.digest(report)}
            (parent / 'state.json').write_text(json.dumps({'status':'COMPLETED','tasks':parent_tasks}), encoding='utf-8')

            run_id = '20261012-040000-000001'
            run = root / 'collab/runs' / run_id
            (run / 'inputs').mkdir(parents=True)
            (run / 'reports').mkdir()
            (run / 'inputs/context.txt').write_text(context, encoding='utf-8')
            parent_peers = '\n\n'.join(
                f'### {m} proposal (round 3)\n{parent_reports[m][:collab.MAX_PEER_CHARS]}'
                for m in collab.MEMBERS)
            current_tasks = {}
            stats = {}
            for member in collab.MEMBERS:
                name = f'round-04-{member}'
                prompt = collab.make_prompt(member, 4, context, parent_peers,
                                            role_name=role_map[member], role_map=role_map)
                stats[member] = {'calls':1, 'attempts':1, 'seconds':20.0,
                                 'cost_usd':0.0, 'tasks':0}
                if member == 'gemini':
                    current_tasks[name] = {'status':'FAILED', 'attempts':3,
                        'prompt_hash':collab.digest(prompt), 'error':'timeout'}
                    stats[member].update(calls=3, attempts=3, seconds=490.0)
                else:
                    report = f'Saved round four work by {member}. ' * 8
                    (run / 'reports' / f'{name}.md').write_text(report, encoding='utf-8')
                    current_tasks[name] = {'status':'DONE', 'attempts':1,
                        'prompt_hash':collab.digest(prompt), 'output_hash':collab.digest(report)}
                    stats[member]['tasks'] = 1
            signature = {
                'sources':{pdf.name:source_hash}, 'rounds':1, 'max_calls':12,
                'timeout':180, 'max_seconds':500, 'member_max_calls':4,
                'member_max_seconds':480, 'member_max_usd':2.0,
                'rotate_roles':False, 'section_failover':False, 'controller':'old',
                'continue_from':parent_id, 'continuation_round':4, 'role_map':role_map
            }
            stale_synthesis = collab.synthesis_prompt('gemini', context, 'peer reports', 'gemini',
                role_name='novelty', role_map=role_map)
            current_tasks['synthesis-gemini'] = {
                'status':'MEMBER_BUDGET_EXHAUSTED', 'attempts':0,
                'prompt_hash':collab.digest(stale_synthesis)
            }
            for member in ('codex', 'claude'):
                name = f'synthesis-{member}'
                text = f'Partial synthesis by {member} without Gemini report. ' * 6
                (run / 'reports' / f'{name}.md').write_text(text, encoding='utf-8')
                current_tasks[name] = {'status':'DONE', 'attempts':1,
                    'prompt_hash':collab.digest('prompt that omitted Gemini'),
                    'output_hash':collab.digest(text)}
                stats[member].update(calls=2, attempts=2, seconds=30.0, tasks=2)
            (run / 'manifest.json').write_text(json.dumps({
                'signature':signature, 'context_hash':collab.digest(context)}), encoding='utf-8')
            (run / 'state.json').write_text(json.dumps({
                'calls':7, 'elapsed':500.0, 'status':'BUDGET_EXHAUSTED',
                'scientific_status':'NEEDS_EVIDENCE', 'tasks':current_tasks,
                'member_stats':stats, 'unavailable':{'gemini':'Member time quota exhausted (480s)'}, 'role_map':role_map,
                'continuation_from':parent_id, 'continuation_round':4
            }), encoding='utf-8')
            (root / 'collab/latest.json').write_text(json.dumps({'id':run_id}), encoding='utf-8')
            invoked = []
            def invoke(member, prompt, folder, timeout, is_synthesis=False):
                invoked.append((member, prompt, is_synthesis))
                return f'Recovered and audited work by {member}. ' * 8
            with patch.object(collab, 'ROOT', root), patch.object(collab, 'invoke_cli', invoke), \
                 patch.object(collab.signal, 'signal'), patch.object(sys, 'argv', [
                     'collab.py','--extend-budget','--max-rounds','1','--max-calls','12',
                     '--member-max-calls','6','--member-max-seconds','960','--member-max-usd','2',
                     '--timeout','180','--max-seconds','1200']):
                self.assertEqual(collab.main(), 0)

            self.assertEqual(len(invoked), 4)
            gemini_round = next(p for m,p,s in invoked if m == 'gemini' and not s)
            self.assertIn(parent_reports['codex'][:60], gemini_round)
            final = json.loads((run / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(final['tasks']['round-04-codex']['status'], 'DONE')
            self.assertEqual(final['tasks']['round-04-claude']['status'], 'DONE')
            self.assertEqual(final['tasks']['round-04-gemini']['status'], 'DONE')
            self.assertEqual(final['calls'], 11)
            for member in collab.MEMBERS:
                self.assertEqual(final['tasks'][f'synthesis-reconciled-{member}']['status'], 'DONE')

    def test_resume_explicit_older_run_without_changing_latest_pointer(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'Problem/source.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'round four source')
            source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            context = '===== paper.pdf PAGE 5 =====\nsource context'
            role_map = collab.get_owners(0)
            parent_id = '20261012-010000-000001'
            parent = root / 'collab/runs' / parent_id
            (parent / 'reports').mkdir(parents=True)
            parent_tasks = {}
            parent_reports = {}
            for member in collab.MEMBERS:
                name = f'round-03-{member}'
                report = f'Parent proposal from {member}. ' * 8
                parent_reports[member] = report
                (parent / 'reports' / f'{name}.md').write_text(report, encoding='utf-8')
                parent_tasks[name] = {'status':'DONE', 'output_hash':collab.digest(report)}
            (parent / 'state.json').write_text(json.dumps({'status':'COMPLETED','tasks':parent_tasks}), encoding='utf-8')

            run_id = '20261012-010001-000001'
            run = root / 'collab/runs' / run_id
            (run / 'inputs').mkdir(parents=True)
            (run / 'reports').mkdir()
            (run / 'inputs/context.txt').write_text(context, encoding='utf-8')
            parent_peers = '\n\n'.join(
                f'### {m} proposal (round 3)\n{parent_reports[m][:collab.MAX_PEER_CHARS]}'
                for m in collab.MEMBERS)
            tasks = {}
            stats = {}
            round_reports = {}
            for member in collab.MEMBERS:
                name = f'round-04-{member}'
                prompt = collab.make_prompt(member, 4, context, parent_peers,
                                            role_name=role_map[member], role_map=role_map)
                report = f'Round four proposal from {member}. ' * 8
                round_reports[member] = report
                (run / 'reports' / f'{name}.md').write_text(report, encoding='utf-8')
                tasks[name] = {'status':'DONE', 'attempts':1, 'prompt_hash':collab.digest(prompt),
                               'output_hash':collab.digest(report)}
                stats[member] = {'calls':2, 'attempts':2, 'seconds':40.0, 'cost_usd':0.0, 'tasks':1}
            round_peers = '\n\n'.join(
                f'### {m}\n{round_reports[m][:collab.MAX_PEER_CHARS]}\n[report excerpt ends]'
                for m in collab.MEMBERS)
            for member in ('codex', 'claude'):
                name = f'synthesis-reconciled-{member}'
                prompt = collab.synthesis_prompt(member, context, round_peers, '',
                                                 role_name=role_map[member], role_map=role_map)
                report = f'Existing reconciliation from {member}. ' * 8
                (run / 'reports' / f'{name}.md').write_text(report, encoding='utf-8')
                tasks[name] = {'status':'DONE', 'attempts':1, 'prompt_hash':collab.digest(prompt),
                               'output_hash':collab.digest(report)}
                stats[member].update(calls=3, attempts=3, tasks=2)
            gemini_name = 'synthesis-reconciled-gemini'
            gemini_prompt = collab.synthesis_prompt('gemini', context, round_peers, '',
                                                    role_name=role_map['gemini'], role_map=role_map)
            tasks[gemini_name] = {'status':'FAILED', 'attempts':4,
                                  'prompt_hash':collab.digest(gemini_prompt), 'error':'four recorded timeouts'}
            failed_attempt = run / 'logs' / gemini_name / '4'
            failed_attempt.mkdir(parents=True)
            (failed_attempt / 'prompt.txt').write_text(gemini_prompt, encoding='utf-8')
            stats['gemini'].update(calls=6, attempts=6)
            signature = {
                'sources':{pdf.name:source_hash}, 'rounds':1, 'max_calls':12, 'timeout':180,
                'max_seconds':1200, 'member_max_calls':8, 'member_max_seconds':1200,
                'member_max_usd':2.0, 'unlimited_budget':False, 'rotate_roles':False,
                'section_failover':False, 'controller':'old', 'continue_from':parent_id,
                'continuation_round':4, 'role_map':role_map
            }
            (run / 'manifest.json').write_text(json.dumps({
                'signature':signature, 'context_hash':collab.digest(context)}), encoding='utf-8')
            (run / 'state.json').write_text(json.dumps({
                'calls':12, 'elapsed':1100.0, 'status':'BUDGET_EXHAUSTED',
                'scientific_status':'NEEDS_EVIDENCE', 'tasks':tasks, 'member_stats':stats,
                'unavailable':{}, 'role_map':role_map, 'budget_recovered_members':['gemini'],
                'continuation_from':parent_id, 'continuation_round':4
            }), encoding='utf-8')
            latest_id = '20261012-020000-000001'
            (root / 'collab/latest.json').write_text(json.dumps({'id':latest_id}), encoding='utf-8')
            invoked = []
            def fake_invoke(member, prompt, folder, timeout, is_synthesis=False, unlimited_budget=False):
                invoked.append((member, is_synthesis, timeout, unlimited_budget))
                return f'Gemini reconciled synthesis with falsifiers. ' * 8
            with patch.object(collab, 'ROOT', root), patch.object(collab, 'invoke_cli', fake_invoke), \
                 patch.object(collab.signal, 'signal'), patch.object(sys, 'argv', [
                     'collab.py','--extend-budget','--unlimited-budget','--repair-gemini-synthesis',
                     '--resume-run-id',run_id,
                     '--max-rounds','1','--timeout','600']):
                self.assertEqual(collab.main(), 0)
            self.assertEqual(invoked, [('gemini', True, 600, False)])
            self.assertEqual(json.loads((root / 'collab/latest.json').read_text())['id'], latest_id)
            final = json.loads((run / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(final['tasks'][gemini_name]['status'], 'DONE')
            self.assertEqual(final['tasks']['synthesis-reconciled-gemini-repair-01']['status'], 'DONE')
            self.assertEqual(final['calls'], 13)
            extended = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
            self.assertIsNone(extended['signature']['max_calls'])
            self.assertTrue(extended['signature']['unlimited_budget'])

    def test_continue_latest_runs_round_two_from_verified_round_one(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'Problem/source.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'unchanged source')
            source_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
            previous = root / 'collab/runs/20261012-010000-000001'
            (previous / 'inputs').mkdir(parents=True)
            (previous / 'reports').mkdir()
            source = '===== paper.pdf PAGE 5 =====\nverified excerpt'
            (previous / 'inputs/context.txt').write_text(source, encoding='utf-8')
            (previous / 'inputs/paper.txt').write_text('full extracted source', encoding='utf-8')
            tasks = {}
            prior_text = {}
            for member in collab.MEMBERS:
                report = f'Round one proposal from {member}. ' * 8
                prior_text[member] = report
                (previous / 'reports' / f'round-01-{member}.md').write_text(report, encoding='utf-8')
                tasks[f'round-01-{member}'] = {
                    'status': 'DONE', 'output_hash': collab.digest(report), 'attempts': 1
                }
            (previous / 'manifest.json').write_text(json.dumps({
                'signature': {'sources': {pdf.name: source_hash}, 'rounds': 1, 'rotate_roles': False},
                'context_hash': collab.digest(source)
            }), encoding='utf-8')
            (previous / 'state.json').write_text(json.dumps({
                'status': 'COMPLETED', 'scientific_status': 'NEEDS_EVIDENCE', 'tasks': tasks,
                'member_stats': {m: {'calls': 1, 'seconds': 1, 'cost_usd': 0} for m in collab.MEMBERS}
            }), encoding='utf-8')
            (root / 'collab/latest.json').write_text(json.dumps({'id': previous.name}), encoding='utf-8')
            prompts = []

            def invoke(member, prompt, folder, timeout):
                prompts.append((member, prompt))
                return f'Round two audit from {member}. Falsifiers and controls. ' * 6

            with patch.object(collab, 'ROOT', root), patch.object(collab, 'invoke_cli', invoke), \
                 patch.object(collab.signal, 'signal'), \
                 patch.object(sys, 'argv', ['collab.py', '--continue-latest', '--unlimited-budget', '--max-calls', '12',
                                            '--max-rounds', '1', '--member-max-calls', '4']):
                self.assertEqual(collab.main(), 0)

            self.assertEqual(len(prompts), 6)
            round_two = [p for _, p in prompts if 'Round two audit' not in p]
            self.assertTrue(round_two)
            for proposal in prior_text.values():
                self.assertTrue(any(proposal[:40] in p for p in round_two))
            runid = json.loads((root / 'collab/latest.json').read_text(encoding='utf-8'))['id']
            state = json.loads((root / 'collab/runs' / runid / 'state.json').read_text(encoding='utf-8'))
            manifest = json.loads((root / 'collab/runs' / runid / 'manifest.json').read_text(encoding='utf-8'))
            self.assertIn('round-02-codex', state['tasks'])
            self.assertEqual(state['scientific_status'], 'NEEDS_EVIDENCE')
            self.assertTrue(manifest['signature']['unlimited_budget'])
            self.assertIsNone(manifest['signature']['max_calls'])
            self.assertIsNone(manifest['signature']['member_max_usd'])
            self.assertEqual((previous / 'reports/round-01-codex.md').read_text(encoding='utf-8'), prior_text['codex'])

            prompts.clear()
            with patch.object(collab, 'ROOT', root), patch.object(collab, 'invoke_cli', invoke), \
                 patch.object(collab.signal, 'signal'), \
                 patch.object(sys, 'argv', ['collab.py', '--continue-latest', '--max-calls', '12',
                                            '--max-rounds', '1', '--member-max-calls', '4']):
                self.assertEqual(collab.main(), 0)
            self.assertEqual(len(prompts), 6)
            self.assertTrue(any('Round two audit from codex' in p for _, p in prompts))
            third_id = json.loads((root / 'collab/latest.json').read_text(encoding='utf-8'))['id']
            third_state = json.loads((root / 'collab/runs' / third_id / 'state.json').read_text(encoding='utf-8'))
            self.assertIn('round-03-codex', third_state['tasks'])

    def test_codex_adapter_preserves_provider_and_does_not_quote_keys(self):
        from unittest.mock import patch
        from collab import invoke_cli
        import os
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'config.toml').write_text(
                'model="existing-model"\n[mcp_servers.node_repl]\ncommand="unused"\n'
                '[plugins."chrome@openai-bundled"]\nenabled=true\n', encoding='utf-8')
            (p / 'answer.txt').write_text('a validated report', encoding='utf-8')
            with patch.dict(os.environ, {'CODEX_HOME': d}), \
                 patch('collab.run_process', return_value=('', '', 0)) as proc:
                self.assertEqual(invoke_cli('codex','p',p,5), 'a validated report')
                cmd = proc.call_args.args[0]
                self.assertIn('mcp_servers.node_repl.enabled=false', cmd)
                self.assertIn('plugins.chrome@openai-bundled.enabled=false', cmd)
                self.assertNotIn('--ignore-user-config', cmd)
                self.assertNotIn('--model', cmd)

    def test_concurrent_budget_and_other_members_survive(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def invoke(member, *args):
                calls.append(member)
                if member == 'gemini':
                    raise RuntimeError('401 invalid API key')
                time.sleep(0.03)
                return 'Evidence and a falsifiable hypothesis. ' * 5
            r = Runner(Path(d), invoke, max_calls=3, backoff=0)
            with ThreadPoolExecutor(max_workers=3) as pool:
                answers = list(pool.map(lambda m: r.task(m, m, 'p'), ('codex','claude','gemini')))
            self.assertEqual(len(calls), 3)
            self.assertTrue(answers[0] and answers[1])
            self.assertIsNone(answers[2])
            self.assertIsNone(r.task('extra', 'claude', 'p'))
            self.assertEqual(len(calls), 3)

    def test_one_controller_lock(self):
        from collab import run_lock
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'controller.lock'
            with run_lock(path):
                with self.assertRaises(RuntimeError):
                    with run_lock(path):
                        pass
            with run_lock(path):
                pass

    def test_interrupted_attempt_counts_on_resume(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            r = Runner(p, lambda *a: 'test report ' * 20, retries=3)
            from collab import digest, save
            r.state['calls'] = 3
            r.state['tasks']['x'] = {'attempts': 3, 'prompt_hash': digest('p'), 'status': 'RUNNING'}
            r.persist()
            r2 = Runner(p, lambda *a: (_ for _ in ()).throw(TimeoutError('timeout')), backoff=0)
            r2.task('x', 'codex', 'p')
            self.assertEqual(r2.state['calls'], 4)
            self.assertIn('codex', r2.state['unavailable'])

    def test_whole_workflow_no_network_and_resume(self):
        from unittest.mock import patch
        import collab
        import json
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'Problem').mkdir()
            (root / 'Problem/source.pdf').write_bytes(b'fixture')
            calls = []
            def fake_prepare(folder, files):
                (folder / 'inputs').mkdir(parents=True)
                (folder / 'inputs/context.txt').write_text('source', encoding='utf-8')
                return 'source'
            def fake_invoke(member, prompt, folder, timeout):
                calls.append(member)
                return 'A hypothesis, falsifier and experimental evidence required. ' * 10
            with patch.object(collab, 'ROOT', root), patch.object(collab, 'prepare_inputs', fake_prepare), \
                 patch.object(collab, 'invoke_cli', fake_invoke), patch.object(sys, 'argv', ['collab.py']), \
                 patch.object(collab.signal, 'signal'):
                self.assertEqual(collab.main(), 0)
                self.assertEqual(len(calls), 9)
                self.assertEqual(collab.main(), 0)
                self.assertEqual(len(calls), 9)
            runid = json.loads((root / 'collab/latest.json').read_text())['id']
            state = json.loads((root / 'collab/runs' / runid / 'state.json').read_text())
            self.assertEqual(state['scientific_status'], 'NEEDS_EVIDENCE')
            self.assertEqual(state['status'], 'COMPLETED')
            summary = (root / 'collab/runs' / runid / 'SUMMARY.md').read_text(encoding='utf-8')
            self.assertIn('### Workload Distribution (Bảng tải)', summary)

    def test_per_member_quota_isolates_failures(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            calls = []
            def invoke(member, prompt, folder, timeout):
                calls.append(member)
                if member == 'codex':
                    raise TimeoutError('codex timeout')
                return 'A valid research report with predictions and methods. ' * 5
            r = Runner(p, invoke, max_calls=12, member_max_calls=2, retries=3, timeout=1, max_seconds=30, backoff=0)
            # Codex fails twice and hits its member_max_calls limit (2)
            self.assertIsNone(r.task('task1', 'codex', 'prompt'))
            self.assertEqual(r.state['member_stats']['codex']['calls'], 2)
            self.assertIn('codex', r.state['unavailable'])
            # Codex cannot run further tasks
            self.assertIsNone(r.task('task2', 'codex', 'prompt'))
            self.assertEqual(r.state['member_stats']['codex']['calls'], 2)
            # Claude still has its full quota intact and succeeds
            res = r.task('task3', 'claude', 'prompt')
            self.assertIsNotNone(res)
            self.assertEqual(r.state['member_stats']['claude']['calls'], 1)

    def test_circular_audit_mapping_and_role_rotation(self):
        from collab import get_owners, CIRCULAR_AUDIT, make_prompt, synthesis_prompt
        # Rotation cycles
        o0 = get_owners(0)
        self.assertEqual(o0, {'codex': 'pilot', 'claude': 'evidence_gate', 'gemini': 'novelty'})
        o1 = get_owners(1)
        self.assertEqual(o1, {'codex': 'evidence_gate', 'claude': 'novelty', 'gemini': 'pilot'})
        o2 = get_owners(2)
        self.assertEqual(o2, {'codex': 'novelty', 'claude': 'pilot', 'gemini': 'evidence_gate'})
        self.assertEqual(get_owners(3), o0)

        # Circular audit targets
        self.assertEqual(CIRCULAR_AUDIT['evidence_gate'], 'pilot')
        self.assertEqual(CIRCULAR_AUDIT['pilot'], 'novelty')
        self.assertEqual(CIRCULAR_AUDIT['novelty'], 'evidence_gate')

        # Claude (evidence_gate) audits Codex (pilot)
        p_claude_r2 = make_prompt('claude', 2, 'source', 'peers', role_name='evidence_gate', role_map=o0)
        self.assertIn("audit CODEX's proposal for 'pilot'", p_claude_r2)
        self.assertIn("Authors must not self-approve", p_claude_r2)
        self.assertIn("agreement from >= 2 agents", p_claude_r2)

        # Codex (pilot) audits Gemini (novelty)
        p_codex_r2 = make_prompt('codex', 2, 'source', 'peers', role_name='pilot', role_map=o0)
        self.assertIn("audit GEMINI's proposal for 'novelty'", p_codex_r2)

        # Gemini (novelty) audits Claude (evidence_gate)
        p_gemini_r2 = make_prompt('gemini', 2, 'source', 'peers', role_name='novelty', role_map=o0)
        self.assertIn("audit CLAUDE's proposal for 'evidence_gate'", p_gemini_r2)

        # Synthesis prompt circular audit rule
        synth_claude = synthesis_prompt('claude', 'source', 'peers', '', role_name='evidence_gate', role_map=o0)
        self.assertIn("independently audit the work of codex (pilot)", synth_claude)
        self.assertIn("Do not select your own proposal", synth_claude)

    def test_evidence_gate_context_filtering(self):
        from collab import get_role_context
        fake_source = (
            "===== paper.pdf PAGE 1 =====\nAbstract and intro\n"
            "===== paper.pdf PAGE 5 =====\nMethodology formulation\n"
            "===== paper.pdf PAGE 7 =====\nBenchmark setup\n"
            "===== paper.pdf PAGE 9 =====\nAccuracy table\n"
            "===== paper.pdf PAGE 26 =====\nProbe results\n"
            "===== paper.pdf PAGE 29 =====\nRelated works\n"
            "===== paper.pdf PAGE 35 =====\nBackground removal table\n"
            "===== paper.pdf PAGE 36 =====\nLocate then count table\n"
        )
        filtered = get_role_context(fake_source, 'evidence_gate')
        self.assertIn('PAGE 5', filtered)
        self.assertIn('PAGE 7', filtered)
        self.assertIn('PAGE 9', filtered)
        self.assertIn('PAGE 26', filtered)
        self.assertIn('PAGE 35', filtered)
        self.assertIn('PAGE 36', filtered)
        self.assertNotIn('PAGE 1', filtered)
        self.assertNotIn('PAGE 29', filtered)

        # Other roles get full source
        self.assertEqual(get_role_context(fake_source, 'pilot'), fake_source)

    def test_claude_budget_defaults_to_two_dollars(self):
        from unittest.mock import patch
        from collab import invoke_cli
        with tempfile.TemporaryDirectory() as d:
            attempt = Path(d)
            with patch('collab.executable', return_value=['claude']), \
                 patch('collab.run_process', return_value=('{"result":"ok"}', '', 0)) as proc:
                # Direct CLI use without a run ledger still respects the user's $2 cap.
                invoke_cli('claude', 'prompt', attempt, 10, is_synthesis=False)
                cmd_regular = proc.call_args.args[0]
                idx = cmd_regular.index('--max-budget-usd')
                self.assertEqual(cmd_regular[idx + 1], '2.00')

                # Synthesis uses the same explicit ceiling.
                invoke_cli('claude', 'prompt', attempt, 10, is_synthesis=True)
                cmd_synth = proc.call_args.args[0]
                idx = cmd_synth.index('--max-budget-usd')
                self.assertEqual(cmd_synth[idx + 1], '2.00')

    def test_claude_request_budget_uses_remaining_member_cap(self):
        from unittest.mock import patch
        from collab import invoke_cli
        with tempfile.TemporaryDirectory() as d:
            attempt = Path(d)
            with patch('collab.executable', return_value=['claude']), \
                 patch('collab.run_process', return_value=('{"result":"ok"}', '', 0)) as proc:
                invoke_cli('claude', 'prompt', attempt, 10, max_budget_usd=1.07)
                cmd = proc.call_args.args[0]
                idx = cmd.index('--max-budget-usd')
                self.assertEqual(cmd[idx + 1], '1.07')

    def test_claude_request_budget_floors_remaining_total(self):
        from unittest.mock import patch
        from collab import invoke_cli
        with tempfile.TemporaryDirectory() as d:
            with patch('collab.executable', return_value=['claude']), \
                 patch('collab.run_process', return_value=('{"result":"ok"}', '', 0)) as proc:
                invoke_cli('claude', 'prompt', Path(d), 10, max_budget_usd=1.06976)
                cmd = proc.call_args.args[0]
                idx = cmd.index('--max-budget-usd')
                self.assertEqual(cmd[idx + 1], '1.06')

    def test_gemini_usd_cap_is_not_applied(self):
        with tempfile.TemporaryDirectory() as d:
            calls = []
            def invoke(member, prompt, folder, timeout, **kwargs):
                calls.append(member)
                return 'A useful report with evidence and falsifiers. ' * 4
            r = Runner(Path(d), invoke, member_max_usd=2, max_calls=4)
            r.state['member_stats']['gemini']['cost_usd'] = 50
            self.assertIsNotNone(r.task('gemini-task', 'gemini', 'prompt'))
            self.assertEqual(calls, ['gemini'])

    def test_workload_summary_table_and_skew_warning(self):
        from collab import build_workload_summary
        with tempfile.TemporaryDirectory() as d:
            r = Runner(Path(d), lambda *a: 'ok' * 30, backoff=0)
            r.state['member_stats'] = {
                'codex': {'tasks': 1, 'attempts': 1, 'seconds': 10.0, 'cost_usd': 0.0},
                'claude': {'tasks': 4, 'attempts': 5, 'seconds': 80.0, 'cost_usd': 1.15},
                'gemini': {'tasks': 1, 'attempts': 1, 'seconds': 10.0, 'cost_usd': 0.0},
            }
            table, warnings = build_workload_summary(r)
            self.assertIn('| claude | 4 | 5 | 80.0s | $1.1500 | 66.7% | 80.0% |', table)
            self.assertEqual(len(warnings), 1)
            self.assertIn("Agent 'claude' chiem hon 50% tai", warnings[0])

    def test_duo_owners_rotation_and_mutual_audit(self):
        from collab import (get_duo_owners, DUO_CIRCULAR_AUDIT, make_prompt,
                            synthesis_prompt)
        o0 = get_duo_owners(0)
        self.assertEqual(o0, {'codex': 'pilot', 'gemini': 'evidence_gate'})
        o1 = get_duo_owners(1)
        self.assertEqual(o1, {'codex': 'evidence_gate', 'gemini': 'pilot'})
        self.assertEqual(get_duo_owners(2), o0)
        self.assertEqual(DUO_CIRCULAR_AUDIT,
                         {'evidence_gate': 'pilot', 'pilot': 'evidence_gate'})
        p = make_prompt('codex', 2, 'source', 'peers', role_name='pilot',
                        role_map=o0, audit=DUO_CIRCULAR_AUDIT, duo=True)
        self.assertIn("audit GEMINI's proposal for 'evidence_gate'", p)
        self.assertIn('two-agent', p)
        s = synthesis_prompt('gemini', 'source', 'peers', '', role_name='evidence_gate',
                             role_map=o0, audit=DUO_CIRCULAR_AUDIT, duo=True)
        self.assertIn('independently audit the work of codex (pilot)', s)
        self.assertIn('two-agent', s)

    def test_duo_fresh_run_uses_only_codex_and_gemini(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'Problem').mkdir()
            (root / 'Problem/source.pdf').write_bytes(b'fixture')
            calls = []
            def fake_prepare(folder, files):
                (folder / 'inputs').mkdir(parents=True)
                (folder / 'inputs/context.txt').write_text('source', encoding='utf-8')
                return 'source'
            def fake_invoke(member, prompt, folder, timeout):
                calls.append(member)
                return 'A hypothesis, falsifier and experimental evidence required. ' * 10
            with patch.object(collab, 'ROOT', root), patch.object(collab, 'prepare_inputs', fake_prepare), \
                 patch.object(collab, 'invoke_cli', fake_invoke), \
                 patch.object(sys, 'argv', ['collab.py', '--duo', '--new-run', '--max-rounds', '1']), \
                 patch.object(collab.signal, 'signal'):
                self.assertEqual(collab.main(), 0)
            self.assertEqual(len(calls), 4)
            self.assertNotIn('claude', calls)
            runid = json.loads((root / 'collab/latest.json').read_text())['id']
            state = json.loads((root / 'collab/runs' / runid / 'state.json').read_text())
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertTrue(state.get('duo_backup'))
            self.assertFalse(any('claude' in name for name in state['tasks']))
            manifest = json.loads((root / 'collab/runs' / runid / 'manifest.json').read_text())
            self.assertTrue(manifest['signature']['duo'])
            self.assertEqual(manifest['signature']['members'], ['codex', 'gemini'])
            self.assertEqual(set(manifest['capabilities']), {'codex', 'gemini'})

    def test_duo_continue_latest_runs_one_more_round_without_claude(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'Problem/source.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'duo source')
            previous = root / 'collab/runs/20261018-001905-740293'
            (previous / 'inputs').mkdir(parents=True)
            (previous / 'reports').mkdir()
            source = '===== paper.pdf PAGE 5 =====\nverified excerpt'
            (previous / 'inputs/context.txt').write_text(source, encoding='utf-8')
            reports = {}
            tasks = {}
            role_map = {'codex': 'pilot', 'gemini': 'evidence_gate'}
            for member in ('codex', 'gemini'):
                report = f'Round two proposal from {member}. ' * 8
                reports[member] = report
                task_name = f'round-02-{member}'
                (previous / 'reports' / f'{task_name}.md').write_text(report, encoding='utf-8')
                tasks[task_name] = {'status': 'DONE', 'output_hash': collab.digest(report), 'attempts': 1}
            signature = {
                'sources': {pdf.name: hashlib.sha256(pdf.read_bytes()).hexdigest()},
                'rounds': 2, 'max_calls': 12, 'timeout': 240, 'max_seconds': 1200,
                'member_max_calls': 4, 'member_max_seconds': None, 'member_max_usd': None,
                'unlimited_budget': False, 'duo': True, 'members': ['codex', 'gemini'],
                'roles': ['pilot', 'evidence_gate'], 'rotate_roles': False,
                'section_failover': False, 'controller': 'old'
            }
            (previous / 'manifest.json').write_text(json.dumps({
                'signature': signature, 'context_hash': collab.digest(source)
            }), encoding='utf-8')
            (previous / 'state.json').write_text(json.dumps({
                'status': 'COMPLETED', 'scientific_status': 'NEEDS_EVIDENCE',
                'tasks': tasks, 'role_map': role_map,
                'member_stats': {m: {'calls': 3, 'attempts': 3, 'seconds': 1, 'cost_usd': 0, 'tasks': 1}
                                 for m in ('codex', 'gemini')}
            }), encoding='utf-8')
            (root / 'collab/latest.json').write_text(json.dumps({'id': previous.name}), encoding='utf-8')
            calls = []

            def fake_invoke(member, prompt, folder, timeout):
                calls.append(member)
                return f'Round three audit from {member}. ' * 8

            with patch.object(collab, 'ROOT', root), patch.object(collab, 'invoke_cli', fake_invoke), \
                 patch.object(collab.signal, 'signal'), \
                 patch.object(sys, 'argv', ['collab.py', '--duo', '--continue-latest',
                                            '--max-rounds', '1', '--max-calls', '12',
                                            '--member-max-calls', '4']):
                self.assertEqual(collab.main(), 0)

            self.assertEqual(len(calls), 4)
            self.assertNotIn('claude', calls)
            runid = json.loads((root / 'collab/latest.json').read_text())['id']
            state = json.loads((root / 'collab/runs' / runid / 'state.json').read_text())
            self.assertIn('round-03-codex', state['tasks'])
            self.assertIn('round-03-gemini', state['tasks'])
            self.assertFalse(any('claude' in name for name in state['tasks']))

    def test_duo_refuses_to_take_over_three_agent_ledger(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'Problem/source.pdf'
            pdf.parent.mkdir()
            pdf.write_bytes(b'source')
            run = root / 'collab/runs/20261012-020000-000001'
            (run / 'inputs').mkdir(parents=True)
            (run / 'inputs/context.txt').write_text('source', encoding='utf-8')
            sig = {'sources': {pdf.name: hashlib.sha256(pdf.read_bytes()).hexdigest()},
                   'rounds': 1, 'max_calls': 12, 'timeout': 180, 'max_seconds': 500,
                   'member_max_calls': 4, 'member_max_seconds': 480, 'member_max_usd': 2.0,
                   'rotate_roles': False, 'section_failover': False, 'controller': 'old'}
            (run / 'manifest.json').write_text(json.dumps(
                {'signature': sig, 'context_hash': collab.digest('source')}), encoding='utf-8')
            (run / 'state.json').write_text(json.dumps(
                {'calls': 1, 'elapsed': 10.0, 'tasks': {}, 'member_stats': {}}), encoding='utf-8')
            (root / 'collab/latest.json').write_text(json.dumps({'id': run.name}), encoding='utf-8')
            with patch.object(collab, 'ROOT', root), patch.object(collab.signal, 'signal'), \
                 patch.object(sys, 'argv', ['collab.py', '--duo', '--extend-budget',
                                            '--max-rounds', '1', '--max-calls', '12',
                                            '--member-max-calls', '4', '--timeout', '180',
                                            '--max-seconds', '1200']):
                with self.assertRaises(ValueError):
                    collab.main()

    def test_section_failover_in_workflow(self):
        from unittest.mock import patch
        import collab
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'Problem').mkdir()
            (root / 'Problem/source.pdf').write_bytes(b'fixture')
            executed_tasks = []
            def fake_prepare(folder, files):
                (folder / 'inputs').mkdir(parents=True)
                (folder / 'inputs/context.txt').write_text('source', encoding='utf-8')
                return 'source'
            def fail_gemini(member, prompt, folder, timeout, is_synthesis=False):
                executed_tasks.append((member, folder.parent.name))
                if member == 'gemini':
                    raise RuntimeError('401 invalid API key')
                return 'A valid hypothesis with falsifiers and controls. ' * 5

            with patch.object(collab, 'ROOT', root), patch.object(collab, 'prepare_inputs', fake_prepare), \
                 patch.object(collab, 'invoke_cli', fail_gemini), \
                 patch.object(sys, 'argv', ['collab.py', '--section-failover', '--max-rounds', '1']), \
                 patch.object(collab.signal, 'signal'):
                collab.main()

            runid = json.loads((root / 'collab/latest.json').read_text())['id']
            state = json.loads((root / 'collab/runs' / runid / 'state.json').read_text())
            # Gemini was unavailable, so its synthesis section was failed over to another available agent
            failover_tasks = [name for name in state['tasks'] if 'failover' in name]
            self.assertTrue(len(failover_tasks) > 0)


if __name__ == '__main__':
    unittest.main()
