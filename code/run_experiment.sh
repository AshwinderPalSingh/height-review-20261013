#!/bin/bash
#   bash run_experiment.sh <full|noreturn|nospeak> <seed>
set -e
cd /workspace/HEIGHT

COND=$1; SEED=$2
[ -z "$COND" ] || [ -z "$SEED" ] && { echo "usage: bash run_experiment.sh <full|noreturn|nospeak|nocal|oracle> <seed>"; exit 1; }
NAME="exp_${COND}_s${SEED}"; OUT="data/${NAME}"
STAGE2="data/stage2_HEIGHT_intention_occlusion/checkpoints/357141.pt"
STEPS="50e6"

# CONF is set explicitly for every condition -- it was previously untouched, so a run
# would silently inherit whatever the last run left in config.py.
case "$COND" in
  full)     ACOUSTIC=True;  RETURN=True;  INFO=0.05; CONF=True  ;;
  noreturn) ACOUSTIC=True;  RETURN=False; INFO=0.05; CONF=True  ;;
  nospeak)  ACOUSTIC=False; RETURN=False; INFO=0.0 ; CONF=False ;;
  # nocal: identical to `full` EXCEPT the confidence channel is off. This is the single
  # -variable test of whether calibrated uncertainty is what keeps injected belief nodes
  # from poisoning the attention graph. v2 changed several things at once, so its 77%
  # collapse could not be attributed to calibration alone.
  nocal)    ACOUSTIC=True;  RETURN=True;  INFO=0.05; CONF=False ;;
  # oracle: perfect perception -- every human visible, occlusion bypassed entirely.
  # Acoustics OFF: with nothing occluded there is nothing for the acoustic channel to
  # sense, so leaving it on would add a dead 18-action space and confound the comparison.
  # This measures the CEILING: no perception module can beat an oracle.
  oracle)   ACOUSTIC=False; RETURN=False; INFO=0.0 ; CONF=False ;;
  # heightonly: PLAIN HEIGHT -- no intention prediction, no occlusion inference, no
  # acoustics. This is the matched internal baseline the paper has been missing: every
  # published comparison so far is against HEIGHT's n=500 numbers from a DIFFERENT
  # implementation. Trained FROM SCRATCH (shapes do not match Stage 2: spatial_edges 4
  # vs 14, robot_node 5 vs 5+latent), and at the FULL 350M budget, because `nospeak` is
  # Stage-2 (~376M: 76M + a 300M relaunch) + 50M. A 50M from-scratch baseline would be undertrained and would
  # flatter our result.
  heightonly) ACOUSTIC=False; RETURN=False; INFO=0.0 ; CONF=False ;;
  # nopredict: `nospeak` with human trajectory prediction turned OFF, PaS occlusion
  # inference left ON. The single-variable ablation of the prediction module inside our
  # OWN codebase. Needed because every run so far used predict_method='truth', which
  # reads each human's ACTUAL future out of the simulator -- privileged state no deployed
  # robot has. This run is the deployable version, and matches HEIGHT's own setting
  # (their released config uses predict_method='none').
  # Trains FROM SCRATCH: spatial_edges shrinks 14 -> 4, so Stage-2 weights cannot load.
  nopredict) ACOUSTIC=False; RETURN=False; INFO=0.0 ; CONF=False ;;
  *) echo "unknown condition: $COND"; exit 1 ;;
esac
# assigned here (not later) so the banner below reports it accurately
if [ "$COND" = "oracle" ]; then ORACLE=True; else ORACLE=False; fi

# heightonly strips the whole stack and trains from scratch at the full budget
if [ "$COND" = "heightonly" ]; then
  OCCL="'none'"; PREDICT="'none'"; RESUME='none'; STEPS="350e6"; SCRATCH=1
elif [ "$COND" = "nopredict" ]; then
  # occlusion STAYS 'vae' -- this isolates prediction, nothing else.
  # Budget is overridable: the 50M default is a quick smoke run, but the PUBLISHABLE
  # ablation was budgeted at 350M; AUDIT 17 SEP: nospeak actually received ~426M
  # (Stage-2 was 76M + a full 300M relaunch, then +50M). 350M matches HEIGHT-350M instead.
  # A 50M-from-scratch arm compared against a 350M warm-started one is confounded by a
  # 7x step disadvantage and a referee discards it.
  #   NOPREDICT_STEPS=350e6 bash Experiments/run_experiment.sh nopredict 1
  OCCL="'vae'"; PREDICT="'none'"; RESUME='none'; STEPS="${NOPREDICT_STEPS:-50e6}"; SCRATCH=1
else
  OCCL="'vae'";  PREDICT="'truth'"; RESUME='rl';  SCRATCH=0
fi

echo "=== $NAME | acoustic=$ACOUSTIC return=$RETURN conf=$CONF oracle=$ORACLE occl=$OCCL predict=$PREDICT resume=$RESUME info=$INFO seed=$SEED steps=$STEPS ==="
tmux has-session -t "$NAME" 2>/dev/null && { echo "session $NAME exists -- aborting"; exit 1; }
[ -d "$OUT" ] && { echo "$OUT exists -- remove it for a clean run"; exit 1; }

C=crowd_nav/configs/config.py
S='[[:space:]]*'          # POSIX class: portable across GNU and BSD sed
cp "$C" "${C}.bak_$(date +%s)"
sed -i "s|^\(${S}env\.seed = \).*|\1${SEED}|"                          "$C"
sed -i "s|^\(${S}sim\.occlusion_method = \).*|\1${OCCL}|"             "$C"
sed -i "s|^\(${S}sim\.predict_method = \).*|\1${PREDICT}|"           "$C"
sed -i "s|^\(${S}sim\.oracle_visibility = \).*|\1${ORACLE}|"           "$C"
sed -i "s|^\(${S}acoustic\.enabled = \).*|\1${ACOUSTIC}|"              "$C"
sed -i "s|^\(${S}acoustic\.return_enabled = \).*|\1${RETURN}|"         "$C"
sed -i "s|^\(${S}acoustic\.info_gain_coef = \).*|\1${INFO}|"           "$C"
sed -i "s|^\(${S}acoustic\.confidence_channel = \).*|\1${CONF}|"      "$C"
sed -i "s|^\(${S}training\.num_env_steps = \).*|\1 ${STEPS}|"          "$C"
sed -i "s|^\(${S}training\.output_dir = \).*|\1'${OUT}'|"              "$C"
sed -i "s|^\(${S}training\.resume = \).*|\1'${RESUME}'|"              "$C"
sed -i "s|^\(${S}training\.load_path = \).*|\1'${OUT}/checkpoints/000000.pt'|" "$C"

mkdir -p "${OUT}/checkpoints"
if [ "$SCRATCH" = "1" ]; then
  # From scratch: NO checkpoint, NO placeholder. training.resume='none' means train.py
  # skips both the weight load and the epoch load, so start_epoch stays 0.
  echo "heightonly: training FROM SCRATCH (no checkpoint, resume='none')"
elif [ "$ACOUSTIC" = "True" ]; then
  # 18 actions, robot_node +1, spatial_edges +1  -> migrate (also writes an epoch-0 placeholder)
  python3 migrate_weights_stage3.py --input_ckpt "$STAGE2" --output_ckpt "${OUT}/checkpoints/000000.pt"
else
  # Shapes already match Stage 2, so no migration. But we must NOT point load_path at the
  # Stage-2 file directly: train.py would then find 357141_checkpoint.pt and set
  # start_epoch=357141, while num_updates is only ~59523 -- range() would be EMPTY and the
  # run would exit instantly having trained nothing. Copy it in with an epoch-0 placeholder.
  cp "$STAGE2" "${OUT}/checkpoints/000000.pt"
  python3 -c "
import torch
torch.save({'epoch':0,'optimizer_data_dict':None,'time_diff':0},
           '${OUT}/checkpoints/000000_checkpoint.pt')
print('nospeak: copied Stage 2 weights + epoch-0 placeholder (no migration needed)')
"
fi

python3 - << PYEOF
import ast, os, torch
ast.parse(open('crowd_nav/configs/config.py').read())
c = __import__('crowd_nav.configs.config', fromlist=['Config']).Config()
SCRATCH = ${SCRATCH}
if SCRATCH:
    # Plain HEIGHT: the whole stack must be OFF, and the budget must match what
    # `nospeak` nominally received (350M; the true figure is ~426M, see audit), or the baseline is
    # undertrained and the comparison flatters us.
    assert c.sim.occlusion_method == ${OCCL}, c.sim.occlusion_method
    assert c.sim.predict_method == 'none', c.sim.predict_method
    assert c.training.resume == 'none', c.training.resume
    assert c.acoustic.enabled is False
    _b = 4 if c.ob_space.add_human_vel else 2
    _rn = 5 + (c.SRNN.occlusion_latent_size if c.sim.occlusion_method == 'vae' else 0)
    if c.sim.occlusion_method == 'vae':
        assert os.path.exists(c.pas.label_vae_ckpt), 'PaS Label_VAE checkpoint missing'
    print('verified SCRATCH: occl=%s predict=none spatial_edges=%d robot_node_in=%d '
          'seed=%d steps=%s num_updates=%d'
          % (c.sim.occlusion_method, _b, _rn, c.env.seed, c.training.num_env_steps,
             int(c.training.num_env_steps)//c.ppo.num_steps//c.training.num_processes))
    raise SystemExit(0)
assert c.sim.occlusion_method == 'vae'
assert c.env.seed == ${SEED}
assert str(c.acoustic.enabled) == '${ACOUSTIC}'
assert str(c.acoustic.return_enabled) == '${RETURN}'
assert str(getattr(c.acoustic,'confidence_channel',False)) == '${CONF}'
assert str(getattr(c.sim,'oracle_visibility',False)) == '${ORACLE}', 'oracle flag mismatch'
assert c.training.output_dir == '${OUT}'
assert os.path.exists(c.pas.label_vae_ckpt)
# the bug that would silently train nothing:
ck = torch.load('${OUT}/checkpoints/000000_checkpoint.pt', map_location='cpu')
nu = int(c.training.num_env_steps)//c.ppo.num_steps//c.training.num_processes
assert ck['epoch'] == 0, 'start_epoch must be 0, got %s' % ck['epoch']
assert ck['epoch'] < nu, 'start_epoch >= num_updates -> loop would be EMPTY'
import torch as _t
_sd = _t.load('${OUT}/checkpoints/000000.pt', map_location='cpu')
_b = 4 if c.ob_space.add_human_vel else 2
_want_se = _b + c.sim.predict_steps*2 + (1 if (c.acoustic.enabled and getattr(c.acoustic,'confidence_channel',False)) else 0)
_want_rl = 5 + c.SRNN.occlusion_latent_size + (1 if c.acoustic.enabled else 0)
for _k,_v in _sd.items():
    if _k.endswith('spatial_attn.embedding_layer.0.weight'):
        assert _v.shape[1] == _want_se, 'spatial_attn %d != model %d (would be DROPPED)' % (_v.shape[1], _want_se)
    if _k.endswith('robot_linear.0.weight'):
        assert _v.shape[1] == _want_rl, 'robot_linear %d != model %d (would be DROPPED)' % (_v.shape[1], _want_rl)
n_act = 18 if c.acoustic.enabled else 9
b = 4 if c.ob_space.add_human_vel else 2
conf = 1 if (c.acoustic.enabled and getattr(c.acoustic,'confidence_channel',False)) else 0
print('verified: actions=%d spatial_edges=%d robot_node=%d conf=%s seed=%d start_epoch=0 num_updates=%d'
      % (n_act, b + c.sim.predict_steps*2 + conf,
         5 + c.SRNN.occlusion_latent_size + (1 if c.acoustic.enabled else 0),
         getattr(c.acoustic,'confidence_channel',False), c.env.seed, nu))
PYEOF

mkdir -p logs
cat > /tmp/_launch_${NAME}.sh << LAUNCH
cd /workspace/HEIGHT
python3 -u train.py 2>&1 | tee logs/${NAME}_train.log
echo EXIT=\$? > logs/${NAME}_DONE.flag
exec bash
LAUNCH
chmod +x /tmp/_launch_${NAME}.sh
tmux new-session -d -s "$NAME" -c /workspace/HEIGHT "bash -l /tmp/_launch_${NAME}.sh"
echo "launched: $NAME"
sleep 150
tail -n 6 "logs/${NAME}_train.log"
echo
echo "SANITY: the log MUST show 'Updates 20...' -- if it jumped straight to EXIT, training did nothing."
