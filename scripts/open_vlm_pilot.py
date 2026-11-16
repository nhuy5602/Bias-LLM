"""CLI entrypoint for offline open-VLM run preparation and phases."""

import argparse
import json
from pathlib import Path

from open_vlm.qwen_backend import QwenBackend, VISUAL_ICL_MODEL_ID
from open_vlm.runner import run_development, run_held_out, run_logit_lens, run_smoke
from open_vlm.dataset import prepare_vlmbias_run
from open_vlm.schema import execution_source_fingerprint, is_model_commit_sha
from open_vlm.visual_icl import PHASE as ICL_PHASE, run_visual_icl


def main(argv=None):
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=False)
    phase.add_argument('--prepare', action='store_true')
    phase.add_argument('--prepare-visual-icl', action='store_true')
    phase.add_argument('--dry-run', action='store_true')
    phase.add_argument('--smoke', action='store_true')
    phase.add_argument('--development', action='store_true')
    phase.add_argument('--held-out', action='store_true')
    phase.add_argument('--no-regression', action='store_true')
    phase.add_argument('--logit-lens', action='store_true')
    phase.add_argument('--visual-icl', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--domains', nargs='+')
    parser.add_argument('--max-per-domain', type=int)
    parser.add_argument('--calibration-max-per-domain', type=int)
    parser.add_argument('--dataset-license', choices=('pending', 'verified'), default='pending')
    parser.add_argument('--model-license', choices=('pending', 'verified'), default='pending')
    parser.add_argument('--model-revision', help='Immutable 40-character Hugging Face model commit SHA')
    parser.add_argument('--gpu-cost-per-hour', type=float)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--budget-usd', type=float)
    budget.add_argument('--unlimited-budget', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare or args.prepare_visual_icl:
        if args.root is None:
            raise SystemExit('Preparation requires --root')
        if not is_model_commit_sha(args.model_revision):
            raise SystemExit('--prepare requires --model-revision as a 40-character commit SHA')
        if args.gpu_cost_per_hour is None or args.gpu_cost_per_hour <= 0:
            raise SystemExit('--prepare requires a positive --gpu-cost-per-hour')
        if args.budget_usd is None and not args.unlimited_budget:
            raise SystemExit('--prepare requires --budget-usd or explicit --unlimited-budget')
        if args.prepare_visual_icl:
            domains = args.domains if args.domains is not None else ['animals']
            max_per_domain = 40 if args.max_per_domain is None else args.max_per_domain
            originals = 20 if args.calibration_max_per_domain is None else args.calibration_max_per_domain
            if domains != ['animals'] or max_per_domain < 40 or max_per_domain % 4 or originals < 20:
                raise SystemExit('Visual ICL requires animals, a multiple of 4 main families (minimum 40), and 20+ originals')
        else:
            domains = args.domains if args.domains is not None else ['animals', 'logos']
            max_per_domain = 20 if args.max_per_domain is None else args.max_per_domain
            originals = args.calibration_max_per_domain
        patch_size = 32 if args.prepare_visual_icl else 28
        config = {
            'dataset_revision': '3761f9fd7163577534a7816c8bc1004035f2e2a0',
            'license_attestations': {'dataset': args.dataset_license, 'model': args.model_license},
            'seed': 20261024,
            'model': {'id': VISUAL_ICL_MODEL_ID if args.prepare_visual_icl else 'Qwen/Qwen2.5-VL-3B-Instruct',
                      'revision': args.model_revision},
            'processor': {'min_pixels': 256 * patch_size**2, 'max_pixels': 512 * patch_size**2},
            'runtime': {'device': 'cuda', 'dtype': 'bfloat16', 'attention': 'sdpa',
                        'gpu_cost_usd_per_hour': args.gpu_cost_per_hour},
            'budget_usd': args.budget_usd,
            'budget_mode': 'unlimited' if args.unlimited_budget else 'capped',
        }
        run_dir = prepare_vlmbias_run(
            args.root, ICL_PHASE if args.prepare_visual_icl else 'development', domains, config,
            max_per_domain=max_per_domain, calibration_max_per_domain=originals)
        print(run_dir)
        return 0
    if not any((args.dry_run, args.smoke, args.development, args.held_out,
                args.no_regression, args.logit_lens, args.visual_icl)):
        raise SystemExit('one phase flag is required')
    if args.run_dir is None:
        raise SystemExit('phase execution requires --run-dir')
    manifest_path = args.run_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'Missing manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('phase') == ICL_PHASE and not (args.visual_icl or args.dry_run):
        raise SystemExit('Visual ICL manifests support --visual-icl and --dry-run')
    if manifest.get('execution_blocked') and not args.dry_run:
        raise SystemExit('Run is blocked until dataset and model license attestations are verified')
    runtime = manifest.get('runtime', {})
    if args.visual_icl and any(manifest.get('license_attestations', {}).get(name) != 'verified'
                              for name in ('dataset', 'model')):
        raise SystemExit('Visual ICL requires verified dataset and model license attestations')
    if not args.dry_run and (args.visual_icl or runtime.get('device', 'cuda') == 'cuda'):
        if not is_model_commit_sha(manifest.get('model', {}).get('revision')):
            raise SystemExit('GPU execution requires an immutable model.revision commit SHA')
        registered_sources = manifest.get('execution_sources')
        if not registered_sources:
            raise SystemExit('GPU execution requires manifest execution_sources fingerprint; prepare a new run')
        current_sources = execution_source_fingerprint()
        if registered_sources != current_sources:
            raise SystemExit('GPU execution source fingerprint differs from the prepared manifest')
    if not args.dry_run and runtime.get('device', 'cuda') == 'cuda':
        mode = manifest.get('budget_mode') or ('capped' if manifest.get('budget_usd') is not None else None)
        if runtime.get('gpu_cost_usd_per_hour') in (None, 0, 0.0):
            raise SystemExit('GPU execution requires manifest gpu_cost_usd_per_hour')
        if mode not in {'capped', 'unlimited'}:
            raise SystemExit('GPU execution requires explicit manifest budget_mode: capped or unlimited')
        if mode == 'capped' and manifest.get('budget_usd') is None:
            raise SystemExit('Capped GPU execution requires manifest budget_usd')
        if mode == 'unlimited' and manifest.get('budget_usd') is not None:
            raise SystemExit('Unlimited GPU execution must not specify budget_usd')
    backend = QwenBackend.from_manifest(manifest)
    if args.visual_icl or (args.dry_run and manifest.get('phase') == ICL_PHASE):
        result = run_visual_icl(manifest, backend, args.run_dir, dry_run=args.dry_run)
    elif args.dry_run:
        result = {'phase': manifest.get('phase', 'unknown'), 'status': 'DRY_RUN', 'calls': 0}
    elif args.smoke:
        result = run_smoke(manifest, backend)
    elif args.development:
        result = run_development(manifest, backend)
    elif args.held_out:
        result = run_held_out(manifest, backend)
    elif args.logit_lens:
        result = run_logit_lens(manifest, backend)
    else:
        from open_vlm.runner import run_no_regression
        result = run_no_regression(manifest, backend)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get('status') in {'SMOKE_FAILED', 'BUDGET_STOPPED', 'GATE_STOPPED',
                                'INCONCLUSIVE', 'UNSUPPORTED', 'FAIL'}:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
