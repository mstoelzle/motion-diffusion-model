import json

import numpy as np

from data_loaders.lafan_g1 import (
    LAFANG1,
    filter_motion_paths,
    g1_vec_to_qpos,
    parse_lafan_action,
    qpos_qvel_to_g1_vec,
)


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _yaw_quat(yaw):
    half = yaw * 0.5
    return np.stack(
        [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)],
        axis=-1,
    )


def _write_motion(path, num_frames=8, yaw_offset=0.0):
    t = np.arange(num_frames, dtype=np.float64)
    joint_pos = np.zeros((num_frames, 36), dtype=np.float64)
    joint_pos[:, 0] = 0.1 * t
    joint_pos[:, 1] = 0.05 * t
    joint_pos[:, 2] = 0.75 + 0.01 * t
    joint_pos[:, 3:7] = _yaw_quat(yaw_offset + 0.02 * t)
    joint_pos[:, 7:] = 0.01 * t[:, None] + np.linspace(-0.1, 0.1, 29)[None]

    joint_vel = np.zeros((num_frames, 35), dtype=np.float64)
    joint_vel[:, 0] = 0.1
    joint_vel[:, 1] = 0.05
    joint_vel[:, 5] = 0.02
    joint_vel[:, 6:] = 0.01

    np.savez(
        path,
        fps=np.array([50]),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=np.zeros((num_frames, 51, 3)),
        body_quat_w=np.zeros((num_frames, 51, 4)),
        body_lin_vel_w=np.zeros((num_frames, 51, 3)),
        body_ang_vel_w=np.zeros((num_frames, 51, 3)),
        joint_names=np.array(JOINT_NAMES),
        body_names=np.array([f"body_{i}" for i in range(51)]),
    )
    return joint_pos, joint_vel


def test_parse_lafan_action():
    assert parse_lafan_action("dance2_subject1_mj_fps50.npz") == "dance"
    assert parse_lafan_action("pushAndStumble1_subject3_mj_fps50.npz") == "pushAndStumble"
    assert parse_lafan_action("fallAndGetUp3_subject1.npz") == "fallAndGetUp"


def test_motion_filter_patterns(tmp_path):
    _write_motion(tmp_path / "dance2_subject1_mj_fps50.npz")
    _write_motion(tmp_path / "jumps1_subject2_mj_fps50.npz")
    paths = filter_motion_paths(tmp_path, "dance*.npz,jumps*.npz")
    assert [path.name for path in paths] == [
        "dance2_subject1_mj_fps50.npz",
        "jumps1_subject2_mj_fps50.npz",
    ]
    assert [path.name for path in filter_motion_paths(tmp_path, "dance*.npz")] == [
        "dance2_subject1_mj_fps50.npz"
    ]


def test_g1_vec_conversion_shapes_and_quaternion_unit(tmp_path):
    joint_pos, joint_vel = _write_motion(tmp_path / "dance2_subject1_mj_fps50.npz")
    vec = qpos_qvel_to_g1_vec(joint_pos, joint_vel)
    assert vec.shape == (joint_pos.shape[0], 39)
    assert np.isfinite(vec).all()

    rec = g1_vec_to_qpos(
        vec,
        fps=50,
        init_xy=joint_pos[0, :2],
        init_yaw=0.0,
    )
    assert rec.shape == (joint_pos.shape[0], 36)
    assert np.isfinite(rec).all()
    assert np.allclose(np.linalg.norm(rec[:, 3:7], axis=-1), 1.0, atol=1e-5)


def test_dataset_prefix_normalization_and_metadata(tmp_path):
    _write_motion(tmp_path / "dance2_subject1_mj_fps50.npz")
    _write_motion(tmp_path / "jumps1_subject2_mj_fps50.npz", yaw_offset=0.3)
    dataset = LAFANG1(data_dir=tmp_path, fixed_len=6, pred_len=2)

    assert dataset.num_actions == 2
    assert dataset.feature_dim == 39
    assert dataset.mean.shape == (39,)
    assert dataset.std.shape == (39,)

    item = dataset[0]
    assert item["prefix"].shape == (39, 1, 4)
    assert item["inp"].shape == (39, 1, 2)
    assert item["lengths"] == 2
    assert item["action_text"] in {"dance", "jumps"}

    normed = dataset.transform(dataset.motions[0]["features"][:3])
    denormed = dataset.inv_transform(normed)
    assert np.allclose(denormed, dataset.motions[0]["features"][:3], atol=1e-6)

    dataset.save_metadata(tmp_path)
    metadata = json.loads((tmp_path / "lafan_g1_metadata.json").read_text())
    assert metadata["feature_dim"] == 39
    assert metadata["joint_names"] == JOINT_NAMES


def test_dataset_prefix_file_selection_keeps_full_normalization(tmp_path):
    _write_motion(tmp_path / "dance2_subject1_mj_fps50.npz")
    _write_motion(tmp_path / "jumps1_subject2_mj_fps50.npz", yaw_offset=0.3)

    full_dataset = LAFANG1(data_dir=tmp_path, fixed_len=6, pred_len=2)
    prefix_dataset = LAFANG1(
        data_dir=tmp_path,
        fixed_len=6,
        pred_len=2,
        prefix_motion_filter="dance*.npz",
        prefix_start=1,
    )

    assert len(prefix_dataset.motion_paths) == 2
    assert len(prefix_dataset) == 1
    assert prefix_dataset[0]["key"] == "dance2_subject1_mj_fps50:1"
    assert prefix_dataset[0]["action_text"] == "dance"
    assert np.allclose(prefix_dataset.mean, full_dataset.mean)
    assert np.allclose(prefix_dataset.std, full_dataset.std)

    exact_file_dataset = LAFANG1(
        data_dir=tmp_path,
        fixed_len=6,
        pred_len=2,
        prefix_file="jumps1_subject2_mj_fps50.npz",
    )

    assert len(exact_file_dataset) == 3
    assert all(exact_file_dataset[i]["key"].startswith("jumps1_subject2_mj_fps50:") for i in range(3))
