"""
Stage 3: Weight Migration Script
---------------------------------
Converts a Stage 2 checkpoint (9 discrete actions) to Stage 3 format (18 actions).

Usage (on DGX):
    python3 migrate_weights_stage3.py \
        --input_ckpt data/stage2_HEIGHT_intention_occlusion/checkpoints/357141.pt \
        --output_ckpt data/stage3_acoustic/checkpoints/000000.pt

What it does:
1. Loads the Stage 2 model state dict (9-action Categorical distribution)
2. Duplicates the 9-action linear weights into an 18-action linear layer:
   - Actions 0-8  (movement only) = original weights
   - Actions 9-17 (movement + speak) = copy of original weights
3. Leaves all other weights untouched (they transfer perfectly)
4. New acoustic layers (acoustic_linear, acoustic_fuse) are NOT in the
   checkpoint — they'll be randomly initialized by the model constructor
"""

import argparse
import os
import torch
import copy


def migrate_checkpoint(input_path, output_path):
    print(f"Loading Stage 2 checkpoint: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu')
    
    # ── 1. Expand dist layer: 9 actions → 18 actions ──
    # The Categorical distribution in model.py creates:
    #   dist.linear.weight  [9, output_size]
    #   dist.linear.bias    [9]
    # We need to expand these to [18, output_size] and [18]
    
    # Find the distribution layer keys
    weight_key = None
    bias_key = None
    for key in state_dict:
        if 'dist' in key and 'weight' in key:
            weight_key = key
        if 'dist' in key and 'bias' in key:
            bias_key = key
    
    if weight_key is None:
        print("WARNING: Could not find distribution weight key. Available keys:")
        for k in state_dict:
            print(f"  {k}: {state_dict[k].shape if hasattr(state_dict[k], 'shape') else type(state_dict[k])}")
        return
    
    old_weight = state_dict[weight_key]  # [9, hidden_size]
    old_bias = state_dict[bias_key]      # [9]
    
    print(f"  Found {weight_key}: {old_weight.shape}")
    print(f"  Found {bias_key}: {old_bias.shape}")
    
    assert old_weight.shape[0] == 9, f"Expected 9 actions, got {old_weight.shape[0]}"
    
    # Create new 18-action weights by duplicating
    new_weight = torch.cat([old_weight, old_weight.clone()], dim=0)  # [18, hidden_size]
    new_bias = torch.cat([old_bias, old_bias.clone()], dim=0)        # [18]
    
    state_dict[weight_key] = new_weight
    state_dict[bias_key] = new_bias
    
    print(f"  Expanded {weight_key}: {old_weight.shape} -> {new_weight.shape}")
    print(f"  Expanded {bias_key}: {old_bias.shape} -> {new_bias.shape}")
    
    # ── 2. Expand robot_linear input: +1 for speak flag ──
    # robot_linear.0.weight has shape [embed_size, robot_input_size]
    # where robot_input_size = 5 (or 3) + occlusion_latent_size (if using PaS-VAE)
    # We add 1 column of zeros for the speak flag dimension
    robot_linear_key = None
    for key in state_dict:
        # Two different layers are named "robot_linear":
        #   1. base.robot_linear.0.weight -- Sequential(Linear, ReLU), takes the RAW
        #      robot_node (+occlusion latent, +acoustic speak flag). Needs the +1.
        #   2. base.humanNodeRNN.robot_linear.weight -- plain nn.Linear inside
        #      EndRNNLidar, fed an ALREADY-EMBEDDED fixed 64-dim vector. Its input size
        #      never changes and padding it CORRUPTS the layer.
        # A bare "'robot_linear' in key" test matches #2 first (it is registered earlier)
        # and expanded the wrong one, so both layers then mismatched and train.py silently
        # dropped and re-initialized them.
        if key.endswith('robot_linear.0.weight'):
            robot_linear_key = key
            break
    
    if robot_linear_key is not None:
        old_rl = state_dict[robot_linear_key]  # [embed_size, robot_input_size]
        print(f"  Found {robot_linear_key}: {old_rl.shape}")
        # Add a zero column for the speak flag (starts with no influence)
        zero_col = torch.zeros(old_rl.shape[0], 1)
        new_rl = torch.cat([old_rl, zero_col], dim=1)  # [embed_size, robot_input_size+1]
        state_dict[robot_linear_key] = new_rl
        print(f"  Expanded {robot_linear_key}: {old_rl.shape} -> {new_rl.shape}")
    
    # ── 2b. Widen the spatial-edge embedding for the confidence channel ──
    # spatial_edges gains a trailing per-agent confidence column, so
    # spatial_attn.embedding_layer.0.weight grows by one input column. Padding it with
    # zeros preserves every trained weight and starts the new channel at zero influence,
    # instead of letting train.py silently drop the whole layer on a shape mismatch and
    # re-initialize the human encoder from scratch.
    spatial_key = None
    for key in state_dict:
        if key.endswith('spatial_attn.embedding_layer.0.weight'):
            spatial_key = key
            break
    if spatial_key is not None:
        old_se = state_dict[spatial_key]
        print(f"  Found {spatial_key}: {tuple(old_se.shape)}")
        state_dict[spatial_key] = torch.cat(
            [old_se, torch.zeros(old_se.shape[0], 1)], dim=1)
        print(f"  Expanded {spatial_key}: {tuple(old_se.shape)} -> "
              f"{tuple(state_dict[spatial_key].shape)}")
    else:
        print("  WARNING: spatial_attn embedding not found - it will be re-initialized")

    # ── 3. Remove acoustic layer keys (will be randomly initialized) ──
    keys_to_remove = [k for k in state_dict if 'acoustic_linear' in k or 'acoustic_fuse' in k]
    for k in keys_to_remove:
        del state_dict[k]
        print(f"  Removed {k} (will be randomly initialized)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state_dict, output_path)
    print(f"Saved migrated checkpoint to: {output_path}")
    
    # Also create a dummy optimizer checkpoint so training can resume
    # (the optimizer will be re-initialized for the new parameters)
    opt_output = output_path.replace('.pt', '_checkpoint.pt')
    opt_data = {
        'epoch': 0,
        'optimizer_data_dict': None,  # Will force optimizer re-init
        'time_diff': 0
    }
    torch.save(opt_data, opt_output)
    print(f"Saved optimizer placeholder to: {opt_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_ckpt', type=str, required=True,
                        help='Path to Stage 2 checkpoint (9 actions)')
    parser.add_argument('--output_ckpt', type=str, required=True,
                        help='Path to save Stage 3 checkpoint (18 actions)')
    args = parser.parse_args()
    
    migrate_checkpoint(args.input_ckpt, args.output_ckpt)


if __name__ == '__main__':
    main()
