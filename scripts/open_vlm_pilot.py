"""CLI entrypoint for offline open-VLM run preparation and phases."""

import argparse
import json
from pathlib import Path

from open_vlm.runner import run_development, run_held_out, run_smoke


class NoopBackend:
    def run(self, task, method):
        raise RuntimeError('No model backend configured; use --dry-run')


def main(argv=None):
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument('--dry-run', action='store_true')
    phase.add_argument('--smoke', action='store_true')
    phase.add_argument('--development', action='store_true')
    phase.add_argument('--held-out', action='store_true')
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.run_dir / 'manifest.json'
    if not manifest_path.is_file():
        raise SystemExit(f'Missing manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    backend = NoopBackend()
    if args.dry_run:
        result = {'phase': manifest.get('phase', 'unknown'), 'status': 'DRY_RUN', 'calls': 0}
    elif args.smoke:
        result = run_smoke(manifest, backend)
    elif args.development:
        result = run_development(manifest, backend)
    else:
        result = run_held_out(manifest, backend)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
