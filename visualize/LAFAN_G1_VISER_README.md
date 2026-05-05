# Visualize LAFAN G1 Samples With HoloSoma Viser

Generated `lafan_g1` samples are saved by `sample.generate` as a single
`results.npy` file containing all sample/repetition pairs. HoloSoma's
`viser_player.py` expects one `.npz` file with `qpos` and `fps`, so first export
the sample you want to inspect.

## Export One Sample

Run this from the `motion-diffusion-model` repo root:

```bash
python -m visualize.export_lafan_g1_to_viser \
  --results_path save/debug_g1_lafan_dance_DiP/samples_600000_g1_dance/results.npy \
  --output_path save/debug_g1_lafan_dance_DiP/samples_600000_g1_dance/viser_sample00_rep00.npz \
  --sample_idx 0 \
  --rep_idx 0
```

Change `--sample_idx` and `--rep_idx` to choose another generated motion. The
exported file contains:

- `qpos`: `(frames, 36)` in MuJoCo order: root xyz, root quaternion, 29 G1 joints
- `fps`: playback FPS
- optional metadata: text/action label, length, joint names

## Export All Samples

```bash
python -m visualize.export_lafan_g1_to_viser \
  --results_path save/debug_g1_lafan_dance_DiP/samples_600000_g1_dance/results.npy \
  --output_dir save/debug_g1_lafan_dance_DiP/samples_600000_g1_dance/viser_npz \
  --all
```

This writes files named like:

```text
sample00_rep00_dance.npz
sample01_rep00_dance.npz
sample00_rep01_dance.npz
```

## Open In HoloSoma Viser

Run this from the HoloSoma retargeting repo:

```bash
cd ~/src/holosoma/src/holosoma_retargeting/holosoma_retargeting

python viser_player.py \
  --robot_urdf models/g1/g1_29dof.urdf \
  --qpos_npz /home/maxi/src/motion-diffusion-model/save/debug_g1_lafan_dance_DiP/samples_600000_g1_dance/viser_npz/sample00_rep00_dance.npz
```

`viser_player.py` visualizes exactly one `.npz` at a time. To view a different
sample, export that sample to a different `.npz` and pass it as `--qpos_npz`.
