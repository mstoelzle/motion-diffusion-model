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

The released LATENT tennis subset may contain zero-filled `qvel` arrays. For
`latent_tennis_g1`, root XY and yaw velocities are therefore derived from
finite differences of `qpos` at the dataset FPS before being converted to the
same root-local `g1_vec` representation.

## What Unconstrained Means

`--unconstrained` sets the model conditioning mode to `no_cond`. In this mode,
the model does not condition on semantic inputs such as text prompts, action
labels, or latent embeddings.

For the DiP-style prefix model shown below, the model is still conditioned on
the previous motion context through `--context_len`: those prefix frames are
G1 robot states in the same normalized `g1_vec` representation. So
"unconstrained" here means no semantic conditioning, while autoregressive
sampling still uses prior motion frames as the local physical/state context.

If both `--context_len` and `--pred_len` are left at zero, there is no prior
motion prefix conditioning either.

## Latent-Conditioned DiP with Current-State Prefix

The pre-windowed latent embedding file:

```text
dataset/latent_tennis_g1/Random_001-004_Tennis_with_embeddings.npz
```

contains `states`, `latents`, and `chunks`. For this mode, the model uses the
existing prefix-completion convention with `--context_len 1`: `chunks[:, 0]` is
the current robot state/prefix, and the non-overlapping target is
`chunks[:, 1:]`. Since the chunks have length 64, the default prediction length
is 63. The raw feature dimension is 80 and the latent condition dimension is
16 for the default file.

When `--cond_mode latent` is used with `latent_tennis_g1`, these defaults are
filled in automatically before `args.json` is written:

- `--latent_embeddings_path dataset/latent_tennis_g1/Random_001-004_Tennis_with_embeddings.npz`
- `--context_len 1`
- `--pred_len 63`

The model still uses the normal DiP prefix machinery internally. The only new
semantic condition is the latent embedding; latent conditioning dropout uses
`--cond_mask_prob`, while the one-frame current-state prefix is always present.

```bash
python -m train.train_mdm \
  --save_dir save/my_latent_tennis_g1_latent_DiP \
  --dataset latent_tennis_g1 \
  --cond_mode latent \
  --arch trans_dec \
  --diffusion_steps 10 \
  --mask_frames \
  --use_ema \
  --train_platform_type TensorboardPlatform \
  --log_interval 1000
```

Use `--latent_embeddings_path /path/to/embeddings.npz` to train from another
pre-windowed file with the same `states`, `latents`, and `chunks` keys.
If you override lengths, `context_len + pred_len` must stay within the chunk
length.

### Generate Latent-Conditioned Samples

Generate from a latent-conditioned checkpoint the same way as other MDM
checkpoints. The model arguments, including `--cond_mode latent`,
`--context_len`, `--pred_len`, and `--latent_embeddings_path`, are loaded from
the checkpoint directory's `args.json`.

```bash
python -m sample.generate \
  --model_path save/my_latent_tennis_g1_latent_DiP/model000600000.pt \
  --output_dir save/my_latent_tennis_g1_latent_DiP/samples_600000_latent \
  --num_samples 6 \
  --num_repetitions 3 \
  --motion_length 1.26 \
  --guidance_param 1.0
```

For non-autoregressive latent sampling, the output length defaults to the
trained prediction length, 63 frames. `--motion_length` is only a cap in this
case. To roll out a longer sequence in repeated 63-frame DiP calls, add
`--autoregressive` and choose a longer `--motion_length`.

The initial current-state prefix and latent embedding are sampled from rows of
the embedding NPZ. To use one exact row, pass its row index through
`--prefix_start` and set `--num_samples 1`:

```bash
python -m sample.generate \
  --model_path save/my_latent_tennis_g1_latent_DiP/model000600000.pt \
  --output_dir save/my_latent_tennis_g1_latent_DiP/samples_row120 \
  --prefix_start 120 \
  --num_samples 1 \
  --num_repetitions 3 \
  --guidance_param 1.0
```

Latent-conditioned generation writes raw feature chunks to `results.npy`; it
does not reconstruct G1 qpos. The result dictionary stores:

- `motion_format`: `latent_motion_chunk`
- `motion`: generated future features with shape `(num_outputs, pred_len, 80)`
- `lengths`: generated future lengths, normally 63 for the default file
- `feature_dim`: `80`
- `latent_cond_dim`: `16`
- `context_len`: `1`
- `prefix_sources`: embedding rows used as current-state/latent sources

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

For the qpos-backed `latent_tennis_g1` mode, `sample.generate` writes G1 robot
samples directly to `results.npy` in the output directory. The `motion` array
stores reconstructed G1 qpos with shape `(num_outputs, frames, 36)`. The file
also stores:

- `motion_format`: `g1_qpos`
- `g1_vec`: generated trajectories before qpos reconstruction
- `fps`: dataset FPS
- `joint_names`: G1 joint order
- `prefix_sources`: dataset windows used as initial context

For `--cond_mode latent`, sample outputs instead use
`motion_format: latent_motion_chunk` and store raw generated future features in
`motion`; these are not Viser-ready G1 qpos trajectories.

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
