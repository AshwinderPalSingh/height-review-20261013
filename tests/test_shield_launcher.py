"""Run the launchers' actual Python flag checks against controlled configs.

No launcher, policy, simulator, or upload is executed. Missing/incorrect flags
must be rejected before a separately named reproduction can start.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))


def check(name, config, argv):
    path = os.path.join(HERE, '..', 'code', name)
    source = open(path).read()
    body = source.split("<<'PYEOF'\n", 1)[1].split('\nPYEOF', 1)[0]
    with patch('importlib.import_module', return_value=SimpleNamespace(Config=lambda: config)):
        with patch.object(sys, 'argv', ['-'] + argv):
            exec(compile(body, path + ':flag-check', 'exec'), {})


def config(enabled=False, obstacles=False, n=3500):
    return SimpleNamespace(cbf=SimpleNamespace(enabled=enabled, horizon=10, margin=0.10,
                                             obstacles=obstacles),
                           env=SimpleNamespace(test_size=n, seed=1, traj_dump=False),
                           training=SimpleNamespace(output_dir='data/checkpoint'))


def rejected(name, c, argv):
    try:
        check(name, c, argv)
    except (AssertionError, AttributeError):
        return
    raise AssertionError('invalid config accepted')


def test_historical_variant_checks_explicit_obstacles_and_size():
    name = 'cbf_2x2.sh'
    check(name, config(), ['data/checkpoint', 'False'])
    check(name, config(enabled=True), ['data/checkpoint', 'True'])
    rejected(name, config(obstacles=True), ['data/checkpoint', 'False'])
    rejected(name, config(n=1000), ['data/checkpoint', 'False'])
    c = config()
    del c.cbf.obstacles
    rejected(name, c, ['data/checkpoint', 'False'])
    c = config()
    c.cbf.obstacles = None
    rejected(name, c, ['data/checkpoint', 'False'])


def test_sweep_checks_all_effective_flags():
    name = 'cbf_sweep.sh'
    argv = ['data/checkpoint', 'True', '10', '0.10', 'True', '1000']
    check(name, config(enabled=True, obstacles=True, n=1000), argv)
    rejected(name, config(enabled=True, obstacles=False, n=1000), argv)
    rejected(name, config(enabled=True, obstacles=True, n=3500), argv)
    c = config(enabled=True, obstacles=True, n=1000)
    c.env.traj_dump = True
    rejected(name, c, argv)


def test_output_names_preserve_original_logs_and_pipeline_fails_fast():
    for name, family in [('cbf_2x2.sh', 'cbf_2x2'), ('cbf_sweep.sh', 'cbf_sweep')]:
        source = open(os.path.join(HERE, '..', 'code', name)).read()
        assert 'set -euo pipefail' in source
        assert 'LOG_DIR="logs/' + family + '_reproduction_' in source
        assert '| tee "$LOG_DIR/' in source
        assert 'sha256sum cbf_filter.py training/evaluation.py' in source
        assert 'local DIR="$REPRO_ROOT/$TAG"' in source
        assert 'prepare_reproduction_run.py "$SOURCE_DIR" "$DIR" "$CK"' in source
        assert '"$SOURCE_DIR/configs/config.py" "$DIR/configs/config_original.py"' in source
        assert '--verify-original' in source
        assert 'set_config_value.py data/' not in source


def test_derived_configs_preserve_source_bytes_and_checkpoint():
    import importlib.util
    import json
    from pathlib import Path
    import subprocess
    import tempfile

    def load(name):
        path = Path(HERE) / '..' / 'code' / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    prep, setter = load('prepare_reproduction_run'), load('set_config_value')
    config_text = """class Section: pass
class Config:
    env = Section()
    env.test_size = 3500
    env.traj_dump = True
    training = Section()
    training.output_dir = 'source'
    cbf = Section()
    cbf.enabled = False
    cbf.obstacles = True
    cbf.horizon = 10
    cbf.margin = 0.1
"""
    initial_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            source = Path('source')
            (source / 'configs').mkdir(parents=True)
            (source / 'checkpoints').mkdir()
            original = source / 'configs/config.py'
            original.write_text(config_text)
            checkpoint = source / 'checkpoints/frozen.pt'
            checkpoint.write_bytes(b'frozen checkpoint fixture')
            original_bytes, checkpoint_bytes = original.read_bytes(), checkpoint.read_bytes()
            for family in ['cbf_2x2', 'cbf_sweep']:
                destination = Path('reproductions') / (family + '_fixture') / 'on'
                provenance = prep.prepare(source, destination, 'frozen.pt')
                effective = destination / 'configs/config.py'
                setter.main(str(effective), 'cbf.enabled', 'True')
                setter.main(str(effective), 'cbf.obstacles', 'False')
                setter.main(str(effective), 'training.output_dir', repr(str(destination)))
                prep.verify_original(destination)
                assert original.read_bytes() == original_bytes
                assert checkpoint.read_bytes() == checkpoint_bytes
                assert (destination / 'configs/config_original.py').read_bytes() == original_bytes
                assert (destination / 'checkpoints/frozen.pt').is_symlink()
                assert (destination / 'checkpoints/frozen.pt').resolve() == checkpoint.resolve()
                module_path = str(destination).replace('/', '.') + '.configs.config'
                subprocess.run([sys.executable, '-c',
                                'from importlib import import_module; c=import_module(' + repr(module_path) +
                                ").Config(); assert c.cbf.enabled is True and c.cbf.obstacles is False"],
                               check=True)
                assert json.loads((destination / 'reproduction_origin.json').read_text()) == provenance
                try:
                    prep.prepare(source, destination, 'frozen.pt')
                except FileExistsError:
                    pass
                else:
                    raise AssertionError('existing reproduction overwritten')
        finally:
            os.chdir(initial_cwd)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
