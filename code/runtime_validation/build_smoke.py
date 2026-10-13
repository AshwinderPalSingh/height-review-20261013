"""Stage the validated CPU smoke in a new folder without running a simulator.

Source, saved configs and weights must match validated_inputs.json exactly.
Generated Python caches and Finder metadata are excluded. Existing files are
never overwritten; the output folder must not already exist.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
ROWS = {
    'exp_nospeak_s1': '59522.pt',
    'exp_nopredict_s1': '416665.pt',
    'exp_heightft': '178570.pt',
}
HARNESS = ('run_runtime_smoke.py', 'simulator_probe.py',
           'standalone_test_probe.py', 'README.md')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def excluded(path):
    return ('__pycache__' in path.parts or path.suffix == '.pyc'
            or path.name in ('.DS_Store', '_DS_Store'))


def build(output):
    assert not output.exists() and not output.is_symlink(), 'Use a new output folder'
    destination = output.resolve()
    assert not destination.is_relative_to((ROOT / 'source').resolve())
    assert not destination.is_relative_to((ROOT / 'configs').resolve())
    assert not destination.is_relative_to((ROOT / 'checkpoints').resolve())
    files = {}
    source = ROOT / 'source'
    assert source.is_dir() and not source.is_symlink()
    for path in sorted(source.rglob('*')):
        assert not path.is_symlink(), 'Source symlinks are not permitted: ' + str(path)
        if path.is_file() and not excluded(path):
            files[path.relative_to(ROOT).as_posix()] = path
    for run, weight in ROWS.items():
        for name in ('configs/runs/' + run + '/config.py',
                     'checkpoints/' + run + '/' + weight):
            files[name] = ROOT / name
    teacher = 'checkpoints/occlusion_vae/label_vae_weight_300.pth'
    files[teacher] = ROOT / teacher

    lock_path = TOOLS / 'validated_inputs.json'
    lock = json.loads(lock_path.read_text())
    expected = {}
    for row in lock['files']:
        assert set(row) == {'path', 'bytes', 'sha256'}
        name = row['path']
        posix = PurePosixPath(name)
        assert not posix.is_absolute() and '..' not in posix.parts and '\\' not in name
        assert name == posix.as_posix() and name not in expected
        assert type(row['bytes']) is int and row['bytes'] >= 0
        assert re.fullmatch(r'[0-9a-f]{64}', row['sha256'])
        expected[name] = row
    assert set(files) == set(expected), {
        'unexpected_inputs': sorted(set(files) - set(expected)),
        'missing_inputs': sorted(set(expected) - set(files)),
    }
    for name, path in files.items():
        assert path.is_file() and not path.is_symlink(), name
        assert path.stat().st_size == expected[name]['bytes'] and sha(path) == expected[name]['sha256'], name

    for name in HARNESS:
        path = TOOLS / name
        assert path.is_file() and not path.is_symlink(), name
        assert name not in files
        files[name] = path
    specification = importlib.util.spec_from_file_location(
        'height_smoke_staging_bootstrap', TOOLS / 'run_runtime_smoke.py')
    bootstrap = importlib.util.module_from_spec(specification)
    # Importing the stdlib-only bootstrap must not create a cache in the artifact.
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(bootstrap)
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    assert bootstrap.ROWS == ROWS
    for run in ROWS:
        original = files['configs/runs/' + run + '/config.py'].read_text()
        assert bootstrap.normalize_config(original) == bootstrap.normalize_config(bootstrap.derive_config(original))

    rows = [{'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
            for name, path in sorted(files.items())]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    for row in rows:
        target = destination / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy payload bytes only, without Finder resource metadata or xattrs.
        shutil.copyfile(files[row['path']], target)
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256'], row['path']
    manifest = {
        'scope': 'Validated source/config/weight bytes with metadata excluded; staging performs no simulator execution',
        'validated_inputs_sha256': sha(lock_path),
        'derived_config_settings': bootstrap.DERIVED,
        'files': rows,
    }
    (destination / 'runtime_smoke_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    bootstrap.validate_artifact(destination)
    return destination, len(expected), len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New staging folder; refuses any existing folder or symlink')
    args = parser.parse_args()
    folder, validated_inputs, bundle_files = build(args.output)
    print('HEIGHT_SMOKE_ARTIFACT=' + str(folder))
    print('HEIGHT_SMOKE_VALIDATED_INPUTS=' + str(validated_inputs))
    print('HEIGHT_SMOKE_BUNDLE_FILES=' + str(bundle_files))
    print('Run the staged smoke with:')
    print(shlex.join([sys.executable, '-B', str(folder / 'run_runtime_smoke.py'),
                      '--artifact', str(folder)]))
