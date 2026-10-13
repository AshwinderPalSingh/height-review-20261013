"""--test_case reaches the simulator with ax=None (the PyBullet path), and the default -1
leaves it untouched. The production make_env body is extracted from
source/training/networks/envs.py and run against stub Gym/Monitor objects, so neither Torch
nor the simulator is needed."""
import ast
import os
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ENVS = os.path.join(HERE, '..', 'source', 'training', 'networks', 'envs.py')


class Inner:
    """Stands in for the unwrapped CrowdSim env (crowd_sim.py initialises test_case=None)."""
    def __init__(self):
        self.test_case = None
        self.observation_space = SimpleNamespace(shape=(1,))
        self.action_space = None

    def configure(self, c):
        self.config = c


class Wrapper:
    """Stands in for Gym's OrderEnforcing wrapper: attribute writes stay on the wrapper."""
    def __init__(self, inner):
        self.unwrapped = inner
        self.observation_space = inner.observation_space

    def configure(self, c):
        self.unwrapped.configure(c)


def make_env_factory():
    tree = ast.parse(open(ENVS).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_env')
    made = []

    def gym_make(*a, **kw):
        w = Wrapper(Inner())
        made.append(w)
        return w
    g = {'gym': SimpleNamespace(make=gym_make, envs=SimpleNamespace()), 'Box': type('Box', (), {}),
         'bench': SimpleNamespace(Monitor=lambda env, *a, **kw: env)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), ENVS, 'exec'), g)
    return g['make_env'], made


def test_requested_case_reaches_unwrapped_env_without_axis():
    make_env, made = make_env_factory()
    env = make_env('CrowdSim3DTbObs-v0', 1, 0, None, True, config=SimpleNamespace(), envNum=1, ax=None,
                   test_case=1485)()
    assert env.unwrapped.test_case == 1485
    assert env.unwrapped.phase == 'test' and env.unwrapped.thisSeed == 1


def test_default_leaves_case_unset():
    make_env, _ = make_env_factory()
    env = make_env('CrowdSim3DTbObs-v0', 1, 0, None, True, config=SimpleNamespace(), envNum=1, ax=None)()
    assert env.unwrapped.test_case is None


def test_case_zero_and_axis_path():
    make_env, _ = make_env_factory()
    env = make_env('CrowdSim3DTbObs-v0', 5, 2, None, True, config=SimpleNamespace(), envNum=1, ax='axis',
                   test_case=0)()
    assert env.unwrapped.test_case == 0 and env.render_axis == 'axis' and env.unwrapped.thisSeed == 7


def test_training_workers_keep_train_phase():
    make_env, _ = make_env_factory()
    env = make_env('CrowdSim3DTbObs-v0', 0, 3, None, True, config=SimpleNamespace(), envNum=28)()
    assert env.unwrapped.phase == 'train' and env.unwrapped.test_case is None


def test_explicit_phase_is_independent_of_worker_count():
    make_env, _ = make_env_factory()
    env = make_env('CrowdSim3DTbObs-v0', 0, 0, None, True, config=SimpleNamespace(), envNum=1,
                   phase='train')()
    assert env.unwrapped.phase == 'train'
    env = make_env('CrowdSim3DTbObs-v0', 0, 0, None, True, config=SimpleNamespace(), envNum=2,
                   phase='test')()
    assert env.unwrapped.phase == 'test'
    try:
        make_env('CrowdSim3DTbObs-v0', 0, 0, None, True, phase='unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid phase accepted')


def test_main_entrypoints_pass_explicit_phase():
    for name, phase in [('train.py', 'train'), ('test.py', 'test')]:
        path = os.path.join(HERE, '..', 'source', name)
        calls = [n for n in ast.walk(ast.parse(open(path).read())) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'make_vec_envs']
        assert len(calls) == 1
        assert any(k.arg == 'phase' and isinstance(k.value, ast.Constant) and k.value.value == phase
                   for k in calls[0].keywords)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
