"""Create an isolated evaluation model directory without editing a saved run.

Only the requested config is copied and the checkpoint is linked read-only by
the evaluation code. Relative identifier paths keep test.py's config import
working. No simulator, checkpoint load, or upload occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(source, destination, checkpoint):
    source = Path(source).resolve()
    destination = Path(destination)
    if destination.is_absolute() or not all(part.isidentifier() for part in destination.parts):
        raise ValueError('destination must be a relative Python package path')
    config = source / 'configs/config.py'
    weight = source / 'checkpoints' / checkpoint
    if Path(checkpoint).name != checkpoint or not config.is_file() or not weight.is_file():
        raise ValueError('source config and requested checkpoint must exist')
    original_hash = sha256(config)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'configs').mkdir()
    (destination / 'checkpoints').mkdir()
    # Give every newly used package directory an initializer; existing files
    # are never overwritten. No initializer is added to the source run.
    for package in [destination / 'configs', destination, *destination.parents]:
        if package == Path('.'):
            break
        initializer = package / '__init__.py'
        if not initializer.exists():
            initializer.touch(exist_ok=False)
    effective = destination / 'configs/config.py'
    shutil.copy2(config, effective)
    effective.chmod(effective.stat().st_mode | stat.S_IWUSR)
    shutil.copy2(config, destination / 'configs/config_original.py')
    (destination / 'checkpoints' / checkpoint).symlink_to(weight.resolve())
    provenance = {'source_model_dir': str(source), 'original_config_path': str(config),
                  'original_config_sha256': original_hash,
                  'checkpoint_path': str(weight.resolve()), 'checkpoint_sha256': sha256(weight),
                  'derived_model_dir': str(destination)}
    (destination / 'reproduction_origin.json').write_text(json.dumps(provenance, indent=2) + '\n')
    assert sha256(config) == original_hash
    return provenance


def verify_original(destination):
    destination = Path(destination)
    provenance = json.loads((destination / 'reproduction_origin.json').read_text())
    expected = provenance['original_config_sha256']
    if sha256(Path(provenance['original_config_path'])) != expected:
        raise ValueError('source run config changed during reproduction')
    if sha256(destination / 'configs/config_original.py') != expected:
        raise ValueError('clean copied snapshot changed during reproduction')
    print('original config and clean copied snapshot unchanged:', expected)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source_or_destination')
    ap.add_argument('destination', nargs='?')
    ap.add_argument('checkpoint', nargs='?')
    ap.add_argument('--verify-original', action='store_true')
    args = ap.parse_args()
    if args.verify_original:
        verify_original(args.source_or_destination)
    else:
        if args.destination is None or args.checkpoint is None:
            ap.error('source, destination and checkpoint are required')
        print(json.dumps(prepare(args.source_or_destination, args.destination, args.checkpoint), indent=2))


if __name__ == '__main__':
    main()
