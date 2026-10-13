#!/usr/bin/env python3
"""Bounded CPU probe; run only through run_runtime_smoke.py in its scratch tree."""
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROWS = {
    'exp_nospeak_s1': ('59522.pt', 'vae', 'truth'),
    'exp_nopredict_s1': ('416665.pt', 'vae', 'none'),
    'exp_heightft': ('178570.pt', 'none', 'none'),
}
SOURCE_IMPORTS = {
    'crowd_sim': 'crowd_sim/__init__.py',
    'baselines': 'baselines/__init__.py',
    'training.networks.envs': 'training/networks/envs.py',
    'training.networks.shmem_vec_env': 'training/networks/shmem_vec_env.py',
    'training.networks.model': 'training/networks/model.py',
    'training.networks.occlusion_vae': 'training/networks/occlusion_vae.py',
    'training.networks.selfAttn_srnn_merge': 'training/networks/selfAttn_srnn_merge.py',
    'training.networks.selfAttn_srnn_merge_lidar': 'training/networks/selfAttn_srnn_merge_lidar.py',
    'baselines.common.vec_env.vec_env': 'baselines/common/vec_env/vec_env.py',
}
RUNTIME_IMPORTS = ('torch', 'gym', 'numpy', 'pybullet', 'rvo2')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_import_evidence():
    """Reject installed-package shadowing; pin the actual imported entry files."""
    evidence = {'prepared_source': {}, 'installed_runtime': {}}
    for name, relative_path in SOURCE_IMPORTS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        expected = (Path.cwd() / relative_path).resolve(strict=True)
        assert path == expected, {'module': name, 'imported': str(path), 'expected': str(expected)}
        evidence['prepared_source'][name] = {'file': str(path), 'sha256': sha(path)}
    for name in RUNTIME_IMPORTS:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve(strict=True)
        evidence['installed_runtime'][name] = {'file': str(path), 'sha256': sha(path)}
    return evidence


def install_private_direct_adapter():
    """Keep this probe out of any pre-existing PyBullet shared-memory server.

    BulletClient(None) tries SHARED_MEMORY before DIRECT in the installed helper.
    Only this subprocess is adapted; prepared artifact bytes remain untouched.
    """
    bullet = importlib.import_module('pybullet')
    clients = importlib.import_module('pybullet_utils.bullet_client')
    original_connect = bullet.connect
    original_init = clients.BulletClient.__init__
    evidence = {
        'adapter': 'harness-only BulletClient(None) -> explicit private DIRECT',
        'lifetime': 'this stage subprocess; fork workers inherit the guards',
        'allowed_connection_method': 'DIRECT',
        'shared_memory_gui_remote_connections': 'rejected before connect',
        'recorded_connection_ids_in_this_process': [],
    }

    def private_connect(connection_mode=None, *args, **kwargs):
        if connection_mode is None and 'connectionMode' in kwargs:
            connection_mode = kwargs.pop('connectionMode')
        assert connection_mode == bullet.DIRECT, 'Probe permits only private PyBullet DIRECT connections'
        assert not args and set(kwargs) <= {'options'}, 'Probe rejects remote/host/shared-memory connection parameters'
        connection_id = original_connect(connection_mode, **kwargs)
        assert connection_id >= 0, 'Private PyBullet DIRECT connection failed'
        info = bullet.getConnectionInfo(physicsClientId=connection_id)
        if info.get('connectionMethod') != bullet.DIRECT:
            bullet.disconnect(physicsClientId=connection_id)
            raise AssertionError('PyBullet returned a non-DIRECT connection despite an explicit private request')
        evidence['recorded_connection_ids_in_this_process'].append(int(connection_id))
        return connection_id

    def private_client_init(self, connection_mode=None, hostName=None, options=''):
        assert hostName is None, 'Probe rejects remote BulletClient hosts'
        if connection_mode is None:
            connection_mode = bullet.DIRECT
        assert connection_mode == bullet.DIRECT, 'Probe rejects GUI/shared-memory BulletClient modes'
        original_init(self, connection_mode=connection_mode, hostName=None, options=options)
        info = bullet.getConnectionInfo(physicsClientId=self._client)
        assert info.get('connectionMethod') == bullet.DIRECT, 'BulletClient is not private DIRECT'

    bullet.connect = private_connect
    clients.BulletClient.__init__ = private_client_init
    return evidence


def load_config(run):
    name = 'data.' + run + '.configs.config'
    module = importlib.import_module(name)
    expected = (Path.cwd() / 'data' / run / 'configs/config.py').resolve()
    assert Path(module.__file__).resolve() == expected, (module.__file__, expected)
    config = copy.deepcopy(module.Config())
    assert config.training.cuda is False
    assert config.env.test_size == 20
    assert config.sim.render is False
    assert config.camera.render_checkpoint is None
    weight, occlusion, prediction = ROWS[run]
    assert config.sim.occlusion_method == occlusion
    assert config.sim.predict_method == prediction
    return config, {'config_module': name, 'resolved_config': str(expected),
                    'derived_config_sha256': sha(expected), 'policy_checkpoint': weight,
                    'occlusion': occlusion, 'prediction': prediction,
                    'configured_seed': config.env.seed}


def scene(env):
    """Hash actual initial scene, including goals/reactivity; exclude physics object IDs."""
    robot = env.robot
    payload = {
        'robot': [float(getattr(robot, name)) for name in
                  ('px', 'py', 'vx', 'vy', 'gx', 'gy', 'radius', 'theta')],
        'humans': [[float(getattr(human, name)) for name in
                    ('px', 'py', 'vx', 'vy', 'gx', 'gy', 'radius')] +
                   [bool(getattr(human, 'react_to_robot', False))]
                   for human in env.humans],
        'obstacles': __import__('numpy').asarray(env.obstacle_coord).tolist(),
        'phase': env.phase, 'rand_seed': int(env.rand_seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                         allow_nan=False).encode()
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'state': payload}


def check_observation(envs, observation, config, nenv):
    import torch
    spaces = envs.observation_space.spaces
    assert set(observation) == set(spaces), (set(observation), set(spaces))
    observed = {}
    for key, box in spaces.items():
        tensor = observation[key]
        assert tensor.device.type == 'cpu', (key, tensor.device)
        assert tuple(tensor.shape) == (nenv,) + tuple(box.shape), (key, tensor.shape, box.shape)
        assert torch.isfinite(tensor).all().item(), key
        observed[key] = list(tensor.shape)
    acoustic = getattr(config, 'acoustic', None)
    acoustic_enabled = acoustic is not None and acoustic.enabled
    robot_width = (5 if config.ob_space.robot_state == 'absolute' else 3) + int(acoustic_enabled)
    spatial_width = (4 if config.ob_space.add_human_vel else 2)
    if config.sim.predict_method != 'none':
        spatial_width += config.sim.predict_steps * 2
    if acoustic_enabled and getattr(acoustic, 'confidence_channel', False):
        spatial_width += 1
    assert tuple(spaces['robot_node'].shape) == (1, robot_width)
    assert spaces['spatial_edges'].shape[-1] == spatial_width
    assert tuple(spaces['point_clouds'].shape) == (1, int(360 / config.lidar.angular_res))
    if config.sim.occlusion_method == 'vae':
        for key in ('grid', 'label_grid'):
            assert tuple(spaces[key].shape) == (1, config.pas.grid_size, config.pas.grid_size)
    else:
        assert 'grid' not in spaces and 'label_grid' not in spaces
    return observed


def make_env(config, nenv=1, phase='test', case=-1):
    import torch
    from training.networks.envs import make_vec_envs
    return make_vec_envs(config.env.env_name, config.env.seed, nenv,
                         config.reward.gamma, None, torch.device('cpu'),
                         allow_early_resets=True, config=config, ax=None,
                         test_case=case, phase=phase)


def assert_cpu(model):
    import torch
    assert all(p.device.type == 'cpu' for p in model.parameters())
    assert all(b.device.type == 'cpu' for b in model.buffers())
    assert not torch.cuda.is_initialized(), 'CUDA initialized during CPU-only probe'


def episodes(run):
    import torch
    from training.networks.model import Policy
    from training.networks.occlusion_vae import Label_VAE
    config, result = load_config(run)
    torch.manual_seed(config.env.seed)
    random.seed(config.env.seed)
    __import__('numpy').random.seed(config.env.seed)
    envs = None
    try:
        envs = make_env(config)
        obs = envs.reset()
        result['shapes'] = check_observation(envs, obs, config, 1)
        teacher_path = Path('data/occlusion_vae/label_vae_ckpt/label_vae_weight_300.pth')
        if config.sim.occlusion_method == 'vae':
            teacher = Label_VAE(config)
            teacher.load_state_dict(torch.load(teacher_path, map_location='cpu', weights_only=True), strict=True)
            assert_cpu(teacher)
            result['strict_teacher_load'] = True
            result['teacher_sha256'] = sha(teacher_path)
            del teacher
        else:
            result['strict_teacher_load'] = 'not applicable: no-VAE row'
        policy = Policy(envs.observation_space.spaces, envs.action_space,
                        base_kwargs=config, base=config.robot.policy)
        weight_path = Path('data') / run / 'checkpoints' / ROWS[run][0]
        policy.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
        policy.base.nenv = 1
        assert_cpu(policy)
        result['strict_policy_load'] = True
        result['policy_sha256'] = sha(weight_path)
        result['model_training_mode'] = policy.training  # matches unmodified test.py
        state = {'rnn': torch.zeros(1, 1, config.SRNN.human_node_rnn_size)}
        masks = torch.zeros(1, 1)
        result['episodes'] = []
        base = envs.venv.envs[0].env.unwrapped
        step_limit = math.ceil(config.env.time_limit / config.env.time_step) + 4
        for episode in range(2):
            initial = scene(base)
            for step in range(1, step_limit + 1):
                with torch.no_grad():
                    value, action, logp, state = policy.act(obs, state, masks, deterministic=True)
                for tensor in (value, action, logp, *state.values()):
                    assert tensor.device.type == 'cpu' and torch.isfinite(tensor).all().item()
                obs, reward, done, infos = envs.step(action)
                check_observation(envs, obs, config, 1)
                masks = torch.tensor([[0.0 if bool(done[0]) else 1.0]])
                if bool(done[0]):
                    outcome = type(infos[0]['info']).__name__
                    assert outcome in ('ReachGoal', 'CollisionHuman', 'CollisionObs', 'CollisionWall', 'Timeout'), outcome
                    result['episodes'].append({'index': episode, 'steps': step,
                                               'outcome': outcome, 'initial_scene': initial})
                    break
            else:
                raise AssertionError('Episode exceeded derived maximum step count')
        assert len(result['episodes']) == 2
        assert result['episodes'][0]['initial_scene']['state']['rand_seed'] != result['episodes'][1]['initial_scene']['state']['rand_seed']
        assert_cpu(policy)
        return result
    finally:
        if envs is not None:
            envs.close()
            envs.close()


def case_replay():
    config, result = load_config('exp_nospeak_s1')
    snapshots = []
    for case in (7, 7, 8):
        envs = None
        try:
            envs = make_env(config, case=case)
            obs = envs.reset()
            check_observation(envs, obs, config, 1)
            base = envs.venv.envs[0].env.unwrapped
            assert base.test_case == case and base.phase == 'test'
            assert base.rand_seed == base.case_capacity['val'] + case + config.env.seed
            snapshots.append({'test_case': case, 'initial_scene': scene(base)})
        finally:
            if envs is not None:
                envs.close()
                envs.close()
    assert snapshots[0]['initial_scene']['sha256'] == snapshots[1]['initial_scene']['sha256']
    assert snapshots[0]['initial_scene']['sha256'] != snapshots[2]['initial_scene']['sha256']
    result['cases'] = snapshots
    return result


def train_phase():
    import torch
    config, result = load_config('exp_nospeak_s1')
    envs = None
    try:
        envs = make_env(config, phase='train')
        obs = envs.reset()
        base = envs.venv.envs[0].env.unwrapped
        assert base.phase == 'train' and base.nenv == 1
        assert base.rand_seed == base.case_capacity['val'] + base.case_capacity['test'] + config.env.seed
        result['initial_scene'] = scene(base)
        result['shapes'] = check_observation(envs, obs, config, 1)
        obs, reward, done, info = envs.step(torch.zeros(1, 1, dtype=torch.long))
        check_observation(envs, obs, config, 1)
        result['one_step_completed'] = True
        return result
    finally:
        if envs is not None:
            envs.close()
            envs.close()


def workers(pending):
    import torch
    from training.networks.shmem_vec_env import ShmemVecEnv
    config, result = load_config('exp_nospeak_s1')
    envs = None
    owned = []
    try:
        envs = make_env(config, nenv=2, phase='train')
        assert isinstance(envs.venv, ShmemVecEnv)
        owned = list(envs.venv.procs)
        assert len(owned) == 2 and all(p.is_alive() for p in owned)
        obs = envs.reset()
        result['shapes'] = check_observation(envs, obs, config, 2)
        actions = torch.zeros(2, 1, dtype=torch.long)
        obs, reward, done, infos = envs.step(actions)
        check_observation(envs, obs, config, 2)
        assert len(infos) == 2 and tuple(reward.shape) == (2, 1)
        if pending:
            envs.step_async(actions)
            assert envs.venv.waiting_step is True
        started = time.monotonic()
        envs.close()
        elapsed = time.monotonic() - started
        envs.close()
        assert all(not p.is_alive() for p in owned), [(p.pid, p.exitcode) for p in owned]
        assert all(p.exitcode == 0 for p in owned), {
            'error': 'A worker was killed or exited abnormally during ordinary/pending close',
            'pending_step_close': pending,
            'workers': [(p.pid, p.exitcode) for p in owned]}
        result.update({'pending_step_close': pending, 'close_seconds': elapsed,
                       'workers': [{'pid': p.pid, 'exitcode': p.exitcode,
                                    'alive_after_double_close': p.is_alive()} for p in owned]})
        return result
    finally:
        if envs is not None:
            envs.close()
        for process in owned:
            if process.is_alive():
                process.terminate()
                process.join(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['episodes', 'cases', 'train', 'workers', 'pending'], required=True)
    parser.add_argument('--run', choices=list(ROWS), default='exp_nospeak_s1')
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    result = {'stage': args.stage, 'run': args.run, 'status': 'failed'}
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden by the bootstrap'
        import torch
        assert not torch.cuda.is_initialized()
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result['physics_isolation'] = install_private_direct_adapter()
        # This also imports crowd_sim to register the verified local Gym envs.
        result['import_evidence'] = collect_import_evidence()
        assert not torch.cuda.is_initialized()
        if args.stage == 'episodes':
            result.update(episodes(args.run))
        elif args.stage == 'cases':
            result.update(case_replay())
        elif args.stage == 'train':
            result.update(train_phase())
        else:
            result.update(workers(args.stage == 'pending'))
        assert not torch.cuda.is_initialized()
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
    Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('HEIGHT_SMOKE_RESULT ' + json.dumps({'stage': args.stage, 'run': args.run, 'status': result['status']}), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
