## Data

* Data dirs should be placed here.

* The `opt` files are configurations for how to read the data according to [text-to-motion](https://github.com/EricGuo5513/text-to-motion).
* The `*_mean.npy` and `*_std.npy` files, are stats used for evaluation only, according to [text-to-motion](https://github.com/EricGuo5513/text-to-motion).

## LAFAN G1

The `lafan_g1/` directory contains LAFAN motions retargeted to the Unitree G1 using [holosoma](https://github.com/amazon-far/holosoma). The robot asset used for the retargeting lives at `body_models/g1/g1_29dof.urdf`, with the adjacent mesh files kept in the same folder so the URDF remains loadable.

Both `dataset/lafan_g1/` and `body_models/g1/` are ignored by git because they are large generated or third-party payloads. To populate the G1 asset in a fresh checkout, run:

```bash
bash prepare/download_g1_asset.sh
```

Retargeting provenance:

1. Downloaded the original LAFAN BVH archive (`lafan1.zip`) from the Ubisoft LaForge Animation Dataset.
2. In holosoma's retargeting package, converted BVH files to global joint-position `.npy` files with `data_utils/extract_global_positions.py`.
3. Ran holosoma's LAFAN robot-only G1 retargeting flow, equivalent to:

   ```bash
   python examples/parallel_robot_retarget.py \
     --data-dir demo_data/lafan \
     --task-type robot_only \
     --data_format lafan \
     --save_dir demo_results_parallel/g1/robot_only/lafan \
     --task-config.object-name ground \
     --task-config.ground-range -10 10 \
     --retargeter.foot-sticking-tolerance 0.02
   ```

4. Converted the retargeted G1 trajectories to the MuJoCo training format at 50 FPS, equivalent to:

   ```bash
   python data_conversion/convert_data_format_mj.py \
     --input_file demo_results_parallel/g1/robot_only/lafan/<motion>.npz \
     --output_fps 50 \
     --output_name converted_res/robot_only/<motion>_mj_fps50.npz \
     --data_format lafan \
     --object_name ground \
     --once
   ```

The copied files came from:

* Motions: `/home/maxi/src/holosoma/src/holosoma_retargeting/holosoma_retargeting/converted_res/robot_only`
* G1 asset: `/home/maxi/src/holosoma/src/holosoma/holosoma/data/robots/g1`
