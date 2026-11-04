"""Run one lab-report arm in a new output directory; archived inputs are read only."""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'height': ('exp_heightpub', '237800.pt'),
        'nospeak': ('exp_nospeak_s1', '59522.pt'),
        'noreturn': ('exp_noreturn_s1', '59522.pt'),
        'full': ('exp_full_s1', '59522.pt'),
        'nocal': ('exp_nocal_s1', '59522.pt')}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--setting', choices=('nominal', 'humans12', 'obstacles15'), default='nominal')
    parser.add_argument('--scene-seed', choices=(1, 2, 3), type=int, default=1)
    parser.add_argument('--episodes', type=int, default=3500)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inspect', action='store_true', help='Load strictly and count parameters; do not evaluate')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 3500:
        parser.error('--episodes must be between 1 and 3500')
    supported = args.setting == 'nominal' or (args.scene_seed == 1 and
        ((args.setting == 'humans12' and args.arm != 'nocal') or
         (args.setting == 'obstacles15' and args.arm in ('height', 'nospeak', 'noreturn'))))
    if not supported or (args.scene_seed != 1 and args.arm not in ('full', 'nospeak')):
        parser.error('This arm/setting/scene-seed combination is not reported in the lab report')
    sys.path.insert(0, str(ROOT / 'source'))
    import torch
    import crowd_sim  # registers the environment
    from training.networks.envs import make_vec_envs
    from training.networks.model import Policy
    from training.evaluation import evaluate

    run, weight_name = ARMS[args.arm]
    cfg_path = ROOT / 'configs/runs' / run / 'config.py'
    weight_path = ROOT / 'checkpoints' / run / weight_name
    spec = importlib.util.spec_from_file_location('lab_run_config', cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.Config()
    # The saved snapshots were mutable during historical evaluation. Record explicit
    # test-scene settings instead of rewriting those snapshots or inferring training seeds.
    config.env.seed = args.scene_seed
    config.env.test_size = args.episodes
    config.training.cuda = False
    config.training.num_threads = 1
    config.sim.render = False
    config.lidar.visualize_rays = False
    config.pybullet.mediaPath = str(ROOT / 'source/crowd_sim/pybullet/media') + '/'
    if hasattr(config, 'pas'):
        config.pas.label_vae_ckpt = str(ROOT / 'checkpoints/occlusion_vae/label_vae_weight_300.pth')
    if args.setting == 'humans12':
        config.sim.human_num, config.sim.human_num_range = 12, 2
    elif args.setting == 'obstacles15':
        config.sim.static_obs_num, config.sim.static_obs_num_range = 15, 2
        config.sim.obs_min_dist = 0.5
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents and 'work' not in out.relative_to(ROOT).parts:
        parser.error('Outputs inside the repository must be under work/')
    out.mkdir(parents=True, exist_ok=False)
    # Upstream also has fixed relative URDF paths; resolve them within the fresh
    # output folder while keeping all runtime writes away from the source tree.
    (out / 'crowd_sim').symlink_to(ROOT / 'source/crowd_sim', target_is_directory=True)
    config.training.output_dir = str(out)
    logging.basicConfig(level=logging.INFO, format='%(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(out / 'evaluation.log')])
    torch.set_num_threads(1)
    torch.manual_seed(args.scene_seed)
    device = torch.device('cpu')
    os.chdir(out)
    envs = None
    try:
        envs = make_vec_envs(config.env.env_name, config.env.seed, 1, config.reward.gamma,
            None, device, True, config=config, phase='test')
        model = Policy(envs.observation_space.spaces, envs.action_space,
                       base_kwargs=config, base=config.robot.policy)
        model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
        model.base.nenv = 1
        model.to(device)
        record = {'arm': args.arm, 'setting': args.setting, 'scene_seed': args.scene_seed,
            'episodes_requested': 0 if args.inspect else args.episodes,
            'config_sha256': sha(cfg_path), 'checkpoint_sha256': sha(weight_path),
            'trainable_actor_critic_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'checkpoint_bytes': weight_path.stat().st_size,
            'checkpoint_MiB': weight_path.stat().st_size / 2**20,
            'strict_checkpoint_load': True, 'torch_version': torch.__version__,
            'scope': 'New CPU execution; archived logs are the manuscript evidence'}
        (out / 'inputs.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2), flush=True)
        if not args.inspect:
            evaluate(model, envs, 1, device, config, logging,
                     SimpleNamespace(dwa=False, visualize=False, random_speak=False))
    finally:
        if envs is not None:
            envs.close()
    assert sha(cfg_path) == record['config_sha256']
    assert sha(weight_path) == record['checkpoint_sha256']

if __name__ == '__main__':
    main()
