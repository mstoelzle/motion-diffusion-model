# Train and Sample Unconstrained MDM on LATENT Tennis G1

The `latent_tennis_g1` dataset trains on LATENT tennis motions saved as G1
robot `.npz` files under:

```text
dataset/latent_tennis_g1
```

The loader reads files recursively, so the current layout
`dataset/latent_tennis_g1/p1/*.npz` works without extra arguments. The dataset
uses the same 39D `g1_vec` motion/trajectory representation as `lafan_g1`:
root yaw velocity, root local XY velocity, root height, root orientation
residual in 6D, and 29 G1 joint angles.

## What Unconstrained Means

`--unconstrained` sets the model conditioning mode to `no_cond`. In this mode,
the model does not condition on text prompts or action labels.

For the DiP-style prefix model shown below, the model is still conditioned on
the previous motion context through `--context_len`: those prefix frames are
G1 robot states in the same normalized `g1_vec` representation. So
"unconstrained" here means no semantic conditioning, while autoregressive
sampling still uses prior motion frames as the local physical/state context.

If both `--context_len` and `--pred_len` are left at zero, there is no prior
motion prefix conditioning either.

## Full Dataset Training

Run from the `motion-diffusion-model` repo root:

```bash
python -m train.train_mdm \
  --save_dir save/my_latent_tennis_g1_uncond_DiP \
  --dataset latent_tennis_g1 \
  --unconstrained \
  --arch trans_dec \
  --diffusion_steps 10 \
  --context_len 20 \
  --pred_len 40 \
  --mask_frames \
  --use_ema \
  --train_platform_type TensorboardPlatform \
  --log_interval 1000 \
  --autoregressive
```

This trains on all `.npz` files under `dataset/latent_tennis_g1`. If the data
lives elsewhere, pass:

```bash
--data_dir /path/to/latent_tennis_g1
```

## Debug/Subsample Training

Use `--motion_filter` to train on a subset of tennis files. It matches the file
name, even when files are nested in player directories:

```bash
python -m train.train_mdm \
  --save_dir save/my_latent_tennis_g1_random001_debug \
  --dataset latent_tennis_g1 \
  --unconstrained \
  --motion_filter "Random_001*.npz" \
  --arch trans_dec \
  --diffusion_steps 10 \
  --context_len 20 \
  --pred_len 40 \
  --mask_frames \
  --use_ema \
  --train_platform_type TensorboardPlatform \
  --log_interval 1000 \
  --autoregressive
```

`--motion_filter` accepts comma-separated shell-style filename patterns:

```bash
--motion_filter "Random_001*.npz,Random_002*.npz"
```

## Timing

The LATENT tennis G1 files currently use 50 FPS. With:

```bash
--context_len 20 --pred_len 40
```

the model sees 20 prior frames, or 0.4 seconds, and predicts 40 future frames,
or 0.8 seconds, per autoregressive chunk.

## Outputs

The training save directory contains:

- `model*.pt`: model checkpoints
- `opt*.pt`: optimizer checkpoints
- `args.json`: training arguments
- `events.out.tfevents.*`: TensorBoard scalar logs when using `TensorboardPlatform`
- `latent_tennis_g1_mean.npy` and `latent_tennis_g1_std.npy`: normalization statistics
- `latent_tennis_g1_metadata.json`: feature layout, joint names, FPS, and bounds

## Loss Curves

With `--train_platform_type TensorboardPlatform`, scalar losses are logged under
the `Loss/` namespace every `--log_interval` training steps:

```bash
tensorboard --logdir save/my_latent_tennis_g1_uncond_DiP
```

The most useful first curves are `Loss/loss` and `Loss/rot_mse`.

## Generate Samples

After training, generate from a checkpoint with:

```bash
python -m sample.generate \
  --model_path save/my_latent_tennis_g1_uncond_DiP/model000600000.pt \
  --output_dir save/my_latent_tennis_g1_uncond_DiP/samples_600000_g1 \
  --num_samples 6 \
  --num_repetitions 3 \
  --motion_length 6.0 \
  --autoregressive \
  --guidance_param 1.0
```

Use `--guidance_param 1.0` for unconstrained models. Classifier-free guidance
only applies when there is text or action conditioning to amplify.

For prefix/autoregressive sampling, `sample.generate` chooses initial context
windows from the dataset. To restrict the source files for those initial
prefixes:

```bash
python -m sample.generate \
  --model_path save/my_latent_tennis_g1_uncond_DiP/model000600000.pt \
  --output_dir save/my_latent_tennis_g1_uncond_DiP/samples_600000_random001_prefix \
  --prefix_motion_filter "Random_001*.npz" \
  --num_samples 6 \
  --num_repetitions 3 \
  --motion_length 6.0 \
  --autoregressive \
  --guidance_param 1.0
```

For an exact source file, use:

```bash
--prefix_file "Random_001_Tennis 001.npz"
```

To use a specific window start frame from that file, add:

```bash
--prefix_start 120
```

If `--prefix_start` selects a single valid prefix window, use `--num_samples 1`
or broaden the prefix selector.

## Sample Outputs

For `latent_tennis_g1`, `sample.generate` writes G1 robot samples directly to
`results.npy` in the output directory. The `motion` array stores reconstructed
G1 qpos with shape `(num_outputs, frames, 36)`. The file also stores:

- `motion_format`: `g1_qpos`
- `g1_vec`: generated trajectories before qpos reconstruction
- `fps`: dataset FPS
- `joint_names`: G1 joint order
- `prefix_sources`: dataset windows used as initial context

## Export for Viser

The existing G1 exporter can convert a sample/repetition pair to a viser `.npz`:

```bash
python -m visualize.export_lafan_g1_to_viser \
  --results_path save/my_latent_tennis_g1_uncond_DiP/samples_600000_g1/results.npy \
  --output_path save/my_latent_tennis_g1_uncond_DiP/samples_600000_g1/viser_sample00_rep00.npz \
  --sample_idx 0 \
  --rep_idx 0
```

Then open the exported `.npz` with your G1-compatible viser player.
