# Train and Sample MDM on G1-Retargeted LAFAN

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
  --train_platform_type TensorboardPlatform \
  --log_interval 1000 \
  --autoregressive \
  --gen_guidance_param 7.5
```

This trains on all `.npz` files in `dataset/lafan_g1`. If you keep the data
somewhere else, pass that location explicitly with `--data_dir /path/to/lafan_g1`.
`TensorboardPlatform` writes loss scalars into the save directory so convergence
curves can be plotted after training.

## Debug/Subsample Training

Use `--motion_filter` to train on a subset of files. This example trains only
on dance motions:

```bash
python -m train.train_mdm \
  --save_dir save/my_g1_lafan_fightAndSports1_subject4_DiP \
  --dataset lafan_g1 \
  --motion_filter "fightAndSports1_subject4*.npz" \
  --arch trans_dec \
  --diffusion_steps 10 \
  --context_len 20 \
  --pred_len 40 \
  --mask_frames \
  --use_ema \
  --train_platform_type TensorboardPlatform \
  --log_interval 1000 \
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
- `events.out.tfevents.*`: TensorBoard scalar logs, including training losses
- `lafan_g1_mean.npy` and `lafan_g1_std.npy`: normalization statistics
- `lafan_g1_metadata.json`: feature layout, joint names, labels, FPS, and bounds

## Loss Curves

With `--train_platform_type TensorboardPlatform`, scalar losses are logged under
the `Loss/` namespace every `--log_interval` training steps. Launch TensorBoard
from the repo root with:

```bash
tensorboard --logdir save/my_g1_lafan_DiP
```

Then open the printed local URL and view the `Loss/loss`, `Loss/rot_mse`, and
other scalar curves. If `tensorboard` is not installed in the active
environment, install it first:

```bash
python -m pip install tensorboard
```

## Generate Samples

After training `save/my_g1_lafan_DiP`, generate samples from a checkpoint with:

```bash
python -m sample.generate \
  --model_path save/my_g1_lafan_DiP/model000600000.pt \
  --output_dir save/my_g1_lafan_DiP/samples_600000_g1 \
  --num_samples 6 \
  --num_repetitions 3 \
  --motion_length 6.0 \
  --autoregressive \
  --guidance_param 7.5
```

Replace `model000600000.pt` with the checkpoint you want to sample. With
`--motion_length 6.0`, the generated motions are 6 seconds long at the dataset
FPS.

To condition on a specific filename-derived LAFAN action label, pass
`--action_name`. `--num_samples` controls how many independent samples use that
same action label:

```bash
python -m sample.generate \
  --model_path save/my_g1_lafan_DiP/model000600000.pt \
  --output_dir save/my_g1_lafan_DiP/samples_600000_g1_fightAndSports \
  --action_name fightAndSports \
  --num_samples 10 \
  --num_repetitions 3 \
  --motion_length 6.0 \
  --autoregressive \
  --guidance_param 7.5
```

To choose which dataset files provide the initial prefix/context frames, use the
sampling-only prefix selectors. This example generates dance-conditioned motion
whose first context window is sampled from files matching `dance2_subject1*.npz`:

```bash
python -m sample.generate \
  --model_path save/my_g1_lafan_DiP/model000600000.pt \
  --output_dir save/my_g1_lafan_DiP/samples_600000_g1_dance2_subject1 \
  --action_name dance \
  --prefix_motion_filter "dance2_subject1*.npz" \
  --num_samples 6 \
  --num_repetitions 3 \
  --motion_length 6.0 \
  --autoregressive \
  --guidance_param 7.5
```

For an exact source file, use `--prefix_file`:

```bash
python -m sample.generate \
  --model_path save/my_g1_lafan_DiP/model000600000.pt \
  --output_dir save/my_g1_lafan_DiP/samples_600000_g1_exact_prefix \
  --prefix_file dance2_subject1_mj_fps50.npz \
  --num_samples 1 \
  --num_repetitions 3 \
  --motion_length 6.0 \
  --autoregressive \
  --guidance_param 7.5
```

To use a specific window start frame from that file, add `--prefix_start`:

```bash
--prefix_start 120
```

`--prefix_motion_filter` accepts the same comma-separated shell-style filename
patterns as `--motion_filter`, but it is only used during sampling to select
initial conditions. It does not change the checkpoint's training data filter or
normalization statistics. If `--prefix_start` selects a single valid prefix
window, use `--num_samples 1` or broaden the prefix selector.

For `lafan_g1`, `sample.generate` writes robot samples directly to
`results.npy` in the output directory. The `motion` array stores reconstructed
G1 qpos with shape `(num_outputs, frames, 36)`, and the file also stores the
sample FPS. The `prefix_sources` entry records the dataset window used as the
initial context for each generated sample, e.g.
`dance2_subject1_mj_fps50:120`.

To visualize a generated sample with HoloSoma's `viser_player.py`, first export
one sample/repetition pair to a viser `.npz`:

```bash
python -m visualize.export_lafan_g1_to_viser \
  --results_path save/my_g1_lafan_DiP/samples_600000_g1/results.npy \
  --output_path save/my_g1_lafan_DiP/samples_600000_g1/viser_sample00_rep00.npz \
  --sample_idx 0 \
  --rep_idx 0
```

Then open the exported `.npz` from the HoloSoma retargeting repo:

```bash
cd ~/src/holosoma/src/holosoma_retargeting/holosoma_retargeting

python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz /home/maxi/src/motion-diffusion-model/save/my_g1_lafan_DiP/samples_600000_g1/viser_sample00_rep00.npz
```

See `visualize/LAFAN_G1_VISER_README.md` for exporting all samples and choosing
different sample/repetition indices.
