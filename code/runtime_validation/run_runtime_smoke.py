#!/usr/bin/env python3
"""Run the prepared artifact in new scratch; never use or edit a live checkout.

This checks compatibility with an EXISTING runtime. It neither installs packages
nor validates the separate fresh-container installation recipe.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROWS = {'exp_nospeak_s1': '59522.pt', 'exp_nopredict_s1': '416665.pt',
        'exp_heightft': '178570.pt'}
DERIVED = {'env.test_size': '20', 'training.cuda': 'False',
           'sim.render': 'False', 'camera.render_checkpoint': 'None'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_artifact(root):
    manifest_path = root / 'runtime_smoke_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = manifest['files']
    expected = {}
    for row in files:
        name = row['path']
        posix = PurePosixPath(name)
        assert not posix.is_absolute() and '..' not in posix.parts and '\\' not in name
        assert name not in expected and name != manifest_path.name
        path = root / name
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name
        expected[name] = row
    actual = set()
    for path in root.rglob('*'):
        assert not path.is_symlink(), 'Artifact must contain no symlinks: ' + str(path)
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    assert actual == set(expected), (sorted(actual - set(expected)), sorted(set(expected) - actual))
    required = ['source/test.py', 'simulator_probe.py', 'standalone_test_probe.py', 'run_runtime_smoke.py',
                'checkpoints/occlusion_vae/label_vae_weight_300.pth']
    for run, weight in ROWS.items():
        required.extend(['configs/runs/' + run + '/config.py',
                         'checkpoints/' + run + '/' + weight])
    assert set(required) <= set(expected), 'Incomplete smoke artifact'
    return manifest, expected


def assignment_key(target):
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        return target.value.id + '.' + target.attr
    return None


def normalize_config(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(assignment_key(t) in DERIVED for t in node.targets):
            node.value = ast.Constant(None)
    return ast.dump(tree, include_attributes=False)


def derive_config(source):
    """Replace only four literal RHS values; assert no other AST change."""
    lines = source.splitlines(keepends=True)
    replacements = []
    found = {key: 0 for key in DERIVED}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                key = assignment_key(target)
                if key in DERIVED:
                    assert len(node.targets) == 1 and isinstance(node.value, ast.Constant), key
                    start = sum(len(line.encode()) for line in lines[:node.value.lineno - 1]) + node.value.col_offset
                    end = sum(len(line.encode()) for line in lines[:node.value.end_lineno - 1]) + node.value.end_col_offset
                    replacements.append((start, end, DERIVED[key].encode()))
                    found[key] += 1
    assert all(n == 1 for n in found.values()), found
    raw = source.encode()
    for start, end, value in sorted(replacements, reverse=True):
        raw = raw[:start] + value + raw[end:]
    derived = raw.decode()
    assert normalize_config(source) == normalize_config(derived)
    return derived


def assert_tree_matches(directory, expected, prefix):
    subset = {name[len(prefix):]: row for name, row in expected.items() if name.startswith(prefix)}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    assert actual == set(subset), 'Copied source inventory differs'
    for name, row in subset.items():
        path = directory / name
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], name


def stop_group(process):
    """Workers share this new session; stop all of them on a stage timeout."""
    for sig, wait in ((signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass
    # Even if parent exited on TERM, a child may have ignored it.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)


def run_stage(command, cwd, log, env, timeout):
    started = time.monotonic()
    result = {'command': command, 'timeout_seconds': timeout, 'log': str(log)}
    with Path(log).open('wb') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        result['pid'] = process.pid
        try:
            result['exit_code'] = process.wait(timeout=timeout)
            result['timed_out'] = False
        except subprocess.TimeoutExpired:
            result['timed_out'] = True
            stop_group(process)
            result['exit_code'] = process.returncode
        except BaseException:
            stop_group(process)
            raise
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def execute(root, scratch_parent):
    scratch = Path(tempfile.mkdtemp(prefix='height-runtime-smoke-20261013-', dir=scratch_parent)).resolve()
    out = scratch / 'outputs'
    out.mkdir()
    report = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'artifact': str(root), 'scratch': str(scratch), 'status': 'failed',
              'validation_scope': ('prepared artifact in existing runtime with a harness-only '
                                   'private PyBullet DIRECT adapter; not a fresh installation'),
              'python': sys.version, 'executable': sys.executable, 'stages': []}
    print('HEIGHT_SMOKE_SCRATCH=' + str(scratch), flush=True)
    try:
        manifest, expected = validate_artifact(root)
        report['manifest_sha256'] = sha(root / 'runtime_smoke_manifest.json')
        report['verified_artifact_files'] = len(expected)
        report['packages'] = {}
        for name in ('torch', 'gym', 'numpy', 'pybullet', 'pyglet', 'pyrvo2', 'matplotlib', 'imageio', 'Cython'):
            try:
                report['packages'][name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                report['packages'][name] = None
        work = scratch / 'checkout'
        shutil.copytree(root / 'source', work)
        assert_tree_matches(work, expected, 'source/')
        data = work / 'data'
        data.mkdir(exist_ok=True)
        (data / '__init__.py').touch()
        configs = []
        for run, weight in ROWS.items():
            run_dir = data / run
            config_dir = run_dir / 'configs'
            checkpoint_dir = run_dir / 'checkpoints'
            config_dir.mkdir(parents=True)
            checkpoint_dir.mkdir()
            (run_dir / '__init__.py').touch()
            (config_dir / '__init__.py').touch()
            original = root / 'configs/runs' / run / 'config.py'
            preserved = scratch / 'original_configs' / run / 'config.py'
            preserved.parent.mkdir(parents=True)
            shutil.copy2(original, preserved)
            assert sha(preserved) == sha(original)
            derived = config_dir / 'config.py'
            derived.write_text(derive_config(original.read_text()))
            shutil.copy2(root / 'checkpoints' / run / weight, checkpoint_dir / weight)
            assert sha(checkpoint_dir / weight) == sha(root / 'checkpoints' / run / weight)
            configs.append({'run': run, 'original_sha256': sha(original),
                            'preserved_original_sha256': sha(preserved),
                            'derived_sha256': sha(derived), 'changed_settings': DERIVED})
        teacher_dir = data / 'occlusion_vae/label_vae_ckpt'
        teacher_dir.mkdir(parents=True)
        teacher = root / 'checkpoints/occlusion_vae/label_vae_weight_300.pth'
        shutil.copy2(teacher, teacher_dir / teacher.name)
        assert sha(teacher) == sha(teacher_dir / teacher.name)
        report['configs'] = configs
        # Keep source imports local and all caches/output inside the new scratch.
        env = os.environ.copy()
        for key in ('PYBULLET_EGL', 'PYTHONPATH', 'DISPLAY'):
            env.pop(key, None)
        env.update({'CUDA_VISIBLE_DEVICES': '', 'NVIDIA_VISIBLE_DEVICES': 'none',
                    'PYTHONDONTWRITEBYTECODE': '1', 'MPLBACKEND': 'Agg',
                    'MPLCONFIGDIR': str(scratch / 'matplotlib'),
                    'XDG_CACHE_HOME': str(scratch / 'cache'),
                    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
        probe = root / 'simulator_probe.py'
        # Invoking a probe outside checkout would put its bundle directory first on sys.path.
        # Copy it into scratch checkout; this is harness material, never an artifact source edit.
        scratch_probe = work / '_height_simulator_probe.py'
        assert not scratch_probe.exists()
        shutil.copy2(probe, scratch_probe)
        standalone_probe = work / '_height_standalone_test_probe.py'
        assert not standalone_probe.exists()
        shutil.copy2(root / 'standalone_test_probe.py', standalone_probe)
        jobs = [('episodes_' + run, 'episodes', run, 1200) for run in ROWS]
        jobs += [('headless_cases', 'cases', 'exp_nospeak_s1', 300),
                 ('single_worker_train', 'train', 'exp_nospeak_s1', 300),
                 ('workers_ordinary', 'workers', 'exp_nospeak_s1', 300),
                 ('workers_pending', 'pending', 'exp_nospeak_s1', 300)]
        for label, stage, run, timeout in jobs:
            print('HEIGHT_SMOKE_STAGE=' + label, flush=True)
            result_path = out / (label + '.json')
            command = [sys.executable, '-B', str(scratch_probe), '--stage', stage,
                       '--run', run, '--result', str(result_path)]
            record = run_stage(command, work, out / (label + '.log'), env, timeout)
            record['label'] = label
            if result_path.is_file():
                record['probe_result'] = json.loads(result_path.read_text())
            record['passed'] = (record['exit_code'] == 0 and not record['timed_out'] and
                                record.get('probe_result', {}).get('status') == 'passed')
            report['stages'].append(record)
            write_json(out / 'runtime_smoke_report.json', report)
        label = 'standalone_test_20'
        print('HEIGHT_SMOKE_STAGE=' + label, flush=True)
        result_path = out / (label + '.json')
        command = [sys.executable, '-B', str(standalone_probe), '--result', str(result_path)]
        record = run_stage(command, work, out / (label + '.log'), env, 1800)
        if result_path.is_file():
            record['probe_result'] = json.loads(result_path.read_text())
        text = (out / (label + '.log')).read_text(errors='replace')
        episode_ids = [int(n) for n in re.findall(r'^Episode\s+(\d+)\s+ends in\s+\d+', text, re.MULTILINE)]
        record.update({'label': label, 'completed_episode_ids': episode_ids,
                       'config_fallback_detected': 'Failed to get Config function' in text})
        record['passed'] = (record['exit_code'] == 0 and not record['timed_out'] and
                            episode_ids == list(range(20)) and not record['config_fallback_detected'] and
                            record.get('probe_result', {}).get('status') == 'passed')
        report['stages'].append(record)
        # Recheck immutable inputs after execution. Source scratch can acquire harness/results,
        # but every prepared source member must still be exactly its packaged bytes.
        for name, row in expected.items():
            assert sha(root / name) == row['sha256'], 'Artifact changed during smoke: ' + name
            if name.startswith('source/'):
                assert sha(work / name[len('source/'):]) == row['sha256'], 'Prepared source changed: ' + name
        for config_record in configs:
            run = config_record['run']
            assert sha(scratch / 'original_configs' / run / 'config.py') == config_record['original_sha256']
            assert sha(data / run / 'configs/config.py') == config_record['derived_sha256']
            assert sha(data / run / 'checkpoints' / ROWS[run]) == sha(root / 'checkpoints' / run / ROWS[run])
        assert sha(teacher_dir / teacher.name) == sha(teacher)
        report['source_and_artifact_bytes_preserved'] = True
        report['status'] = 'passed' if all(s['passed'] for s in report['stages']) else 'failed'
    except BaseException as error:
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
    write_json(out / 'runtime_smoke_report.json', report)
    print('HEIGHT_SMOKE_REPORT=' + str(out / 'runtime_smoke_report.json'), flush=True)
    print('HEIGHT_SMOKE_STATUS=' + report['status'], flush=True)
    return 0 if report['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--scratch-parent', type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    assert os.name == 'posix', 'Process-group cleanup requires POSIX'
    root = args.artifact.resolve(strict=True)
    scratch_parent = args.scratch_parent.resolve(strict=True)
    assert root.is_dir() and scratch_parent.is_dir()
    assert root != scratch_parent and root not in scratch_parent.parents, 'Scratch must be outside artifact'
    return execute(root, str(scratch_parent))


if __name__ == '__main__':
    raise SystemExit(main())
