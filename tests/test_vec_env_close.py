"""ShmemVecEnv shutdown with the stand-in Baselines VecEnv (source/baselines).

close() must reach close_extras(), close each worker's env, join every worker, survive a
pending step and a dead worker, and be safe to call twice. Two lightweight workers with a
NumPy-only env; Gym is replaced by a minimal stub when it is not installed.
"""
import multiprocessing as mp
import os
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'source')
sys.path.insert(0, SRC)
try:
    import gym  # noqa: F401
except ImportError:
    gym = types.ModuleType('gym')
    spaces = types.ModuleType('gym.spaces')

    class Dict:
        pass
    spaces.Dict = Dict
    gym.spaces = spaces
    sys.modules['gym'] = gym
    sys.modules['gym.spaces'] = spaces

from training.networks.shmem_vec_env import ShmemVecEnv  # noqa: E402

MARK = tempfile.mkdtemp(prefix='vecenv_close_')


class Space:
    shape = (3,)
    dtype = np.float32


class MockEnv:
    def __init__(self, tag):
        self.tag = tag

    def reset(self):
        return np.zeros(3, np.float32)

    def step(self, action):
        return np.full(3, float(action), np.float32), 0.0, False, {}

    def close(self):
        open(os.path.join(MARK, 'closed_%s' % self.tag), 'w').close()


def maker(tag):
    return lambda: MockEnv(tag)


def make(prefix):
    return ShmemVecEnv([maker(prefix + '0'), maker(prefix + '1')], spaces=(Space(), None), context='fork')


def closed(prefix):
    return sorted(f for f in os.listdir(MARK) if f.startswith('closed_' + prefix))


def test_idle_close_joins_workers_and_is_idempotent():
    v = make('idle')
    v.reset()
    v.close()
    assert not any(p.is_alive() for p in v.procs)
    assert closed('idle') == ['closed_idle0', 'closed_idle1']
    v.close()                                           # second close is a no-op
    assert v.closed


def test_close_with_pending_step():
    v = make('pending')
    v.reset()
    v.step_async(np.array([1, 2]))
    v.close()
    assert not v.waiting_step and not any(p.is_alive() for p in v.procs)
    assert closed('pending') == ['closed_pending0', 'closed_pending1']


def test_close_with_a_dead_worker_returns():
    v = make('dead')
    v.reset()
    v.procs[0].terminate()
    v.procs[0].join(5)
    t = time.time()
    v.close()
    assert time.time() - t < 30 and not any(p.is_alive() for p in v.procs)
    assert closed('dead') == ['closed_dead1']


class StuckEnv(MockEnv):
    def step(self, action):
        # A stuck simulator call cannot consume a close command until it returns.
        time.sleep(60)
        return super().step(action)


def test_pending_step_with_hung_worker_is_bounded():
    v = ShmemVecEnv([lambda: StuckEnv('stuck'), maker('responsive')],
                    spaces=(Space(), None), context='fork')
    v.reset()
    v.step_async(np.array([1, 2]))
    start = time.monotonic()
    v.close()
    assert time.monotonic() - start < 16
    assert not v.waiting_step and not any(p.is_alive() for p in v.procs)
    v.close()


if __name__ == '__main__':
    if 'fork' not in mp.get_all_start_methods():
        sys.exit('fork start method unavailable on this platform')
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
