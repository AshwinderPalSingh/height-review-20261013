#!/usr/bin/env python3
"""Check import/config identity, then run the unmodified standalone test.py."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

from _height_simulator_probe import collect_import_evidence, install_private_direct_adapter, load_config, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    result = {'status': 'failed', 'stage': 'standalone_test_20'}
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        import torch
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result['physics_isolation'] = install_private_direct_adapter()
        result['import_evidence'] = collect_import_evidence()
        assert not torch.cuda.is_initialized()
        config, config_evidence = load_config('exp_nospeak_s1')
        result['config_evidence'] = config_evidence
        test_path = (Path.cwd() / 'test.py').resolve(strict=True)
        result['test_source_sha256'] = sha(test_path)
        original_argv = sys.argv[:]
        try:
            # Imported classes are not instantiated here; test.py retains its own
            # seeding, policy mode, strict loading and evaluation behavior.
            sys.argv = [str(test_path), '--model_dir', 'data/exp_nospeak_s1',
                        '--test_model', '59522.pt', '--cpu']
            result['original_test_argv'] = sys.argv[:]
            runpy.run_path(str(test_path), run_name='__main__')
        finally:
            sys.argv = original_argv
        assert not torch.cuda.is_initialized(), 'Standalone CPU evaluation initialized CUDA'
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
    Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('HEIGHT_SMOKE_RESULT ' + json.dumps({'stage': result['stage'], 'status': result['status']}), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
