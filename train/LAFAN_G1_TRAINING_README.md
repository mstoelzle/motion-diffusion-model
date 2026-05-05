# Train MDM on G1-Retargeted LAFAN

The `lafan_g1` dataset trains on retargeted LAFAN motions saved as G1 robot
`.npz` files. By default it uses all matching `.npz` files from the configured
data directory and conditions on action/style labels parsed from filenames.

The default G1 LAFAN data directory is inside this repo:

```text
dataset/lafan_g1
```

## Full Dataset Training

Run from the `motion-diffusion-model` repo root:

```bash
python -m train.train_mdm \
  --save_dir save/my_g1_lafan_DiP \
  --dataset lafan_g1 \
  --arch trans_dec \
  --diffusion_steps 10 \
  --context_len 20 \
  --pred_len 40 \
  --mask_frames \
  --use_ema \
  --autoregressive \
  --gen_guidance_param 7.5
```

This trains on all `.npz` files in `dataset/lafan_g1`. If you keep the data
somewhere else, pass that location explicitly with `--data_dir /path/to/lafan_g1`.

## Debug/Subsample Training

Use `--motion_filter` to train on a subset of files. This example trains only
on dance motions:

```bash
python -m train.train_mdm \
  --save_dir save/debug_g1_lafan_dance_DiP \
  --dataset lafan_g1 \
  --motion_filter "dance*.npz" \
  --arch trans_dec \
  --diffusion_steps 10 \
  --context_len 20 \
  --pred_len 40 \
  --mask_frames \
  --use_ema \
  --autoregressive \
  --gen_guidance_param 7.5
```

`--motion_filter` accepts comma-separated shell-style filename patterns, for
example:

```bash
--motion_filter "dance*.npz,jumps*.npz"
```

## Timing

The converted G1 LAFAN files are 50 FPS. With:

```bash
--context_len 20 --pred_len 40
```

the model uses 20 past frames, or 0.4 seconds, and predicts 40 future frames,
or 0.8 seconds, per autoregressive chunk.

## Outputs

The training save directory contains:

- `model*.pt`: model checkpoints
- `opt*.pt`: optimizer checkpoints
- `args.json`: training arguments
- `lafan_g1_mean.npy` and `lafan_g1_std.npy`: normalization statistics
- `lafan_g1_metadata.json`: feature layout, joint names, labels, FPS, and bounds
