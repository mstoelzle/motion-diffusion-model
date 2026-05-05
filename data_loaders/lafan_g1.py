import fnmatch
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LAFAN_G1_DIR = str(_REPO_ROOT / "dataset" / "lafan_g1")
DEFAULT_G1_ROBOT_MODEL = str(_REPO_ROOT / "body_models" / "g1" / "g1_29dof.urdf")

FEATURE_LAYOUT = {
    "root_yaw_velocity": [0, 1],
    "root_local_xy_velocity": [1, 3],
    "root_height": [3, 4],
    "root_orientation_residual_6d": [4, 10],
    "joint_angles": [10, 39],
}


def parse_lafan_action(path_or_name):
    stem = Path(path_or_name).stem
    stem = re.sub(r"_mj_fps\d+$", "", stem)
    stem = stem.split("_subject", 1)[0]
    action = re.sub(r"\d+$", "", stem)
    if not action:
        raise ValueError(f"Could not parse LAFAN action from [{path_or_name}]")
    return action


def expand_motion_filter(motion_filter):
    if not motion_filter:
        return ["*.npz"]
    return [pattern.strip() for pattern in motion_filter.split(",") if pattern.strip()]


def filter_motion_paths(data_dir, motion_filter="*.npz"):
    data_dir = Path(data_dir or DEFAULT_LAFAN_G1_DIR).expanduser()
    patterns = expand_motion_filter(motion_filter)
    paths = sorted(
        path for path in data_dir.glob("*.npz")
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
    )
    if not paths:
        raise FileNotFoundError(
            f"No .npz files matched [{motion_filter}] in [{data_dir}]"
        )
    return paths


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, 1e-12)


def quat_conjugate(q):
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_yaw(q):
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw):
    yaw = np.asarray(yaw, dtype=np.float64)
    half = yaw * 0.5
    return np.stack(
        [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)],
        axis=-1,
    )


def rotate_xy_by_yaw(xy, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def rotate_xy_by_inverse_yaw(xy, yaw):
    return rotate_xy_by_yaw(xy, -yaw)


def quat_to_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.stack(
        [
            np.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], axis=-1),
        ],
        axis=-2,
    )


def matrix_to_quat(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=np.float64)
    for i, mat in enumerate(flat):
        trace = np.trace(mat)
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            out[i] = [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ]
        else:
            axis = int(np.argmax(np.diag(mat)))
            if axis == 0:
                s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
                out[i] = [
                    (mat[2, 1] - mat[1, 2]) / s,
                    0.25 * s,
                    (mat[0, 1] + mat[1, 0]) / s,
                    (mat[0, 2] + mat[2, 0]) / s,
                ]
            elif axis == 1:
                s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
                out[i] = [
                    (mat[0, 2] - mat[2, 0]) / s,
                    (mat[0, 1] + mat[1, 0]) / s,
                    0.25 * s,
                    (mat[1, 2] + mat[2, 1]) / s,
                ]
            else:
                s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
                out[i] = [
                    (mat[1, 0] - mat[0, 1]) / s,
                    (mat[0, 2] + mat[2, 0]) / s,
                    (mat[1, 2] + mat[2, 1]) / s,
                    0.25 * s,
                ]
    return quat_normalize(out.reshape(m.shape[:-2] + (4,)))


def matrix_to_rotation_6d(matrix):
    return np.asarray(matrix, dtype=np.float64)[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6):
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def qpos_qvel_to_g1_vec(joint_pos, joint_vel):
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    joint_vel = np.asarray(joint_vel, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 36:
        raise ValueError(f"Expected joint_pos shape (T, 36), got {joint_pos.shape}")
    if joint_vel.ndim != 2 or joint_vel.shape[1] != 35:
        raise ValueError(f"Expected joint_vel shape (T, 35), got {joint_vel.shape}")

    root_quat = quat_normalize(joint_pos[:, 3:7])
    yaw = quat_to_yaw(root_quat)
    yaw_quat = yaw_to_quat(yaw)
    residual_quat = quat_mul(quat_conjugate(yaw_quat), root_quat)
    residual_6d = matrix_to_rotation_6d(quat_to_matrix(residual_quat))

    root_local_xy_velocity = rotate_xy_by_inverse_yaw(joint_vel[:, 0:2], yaw)
    root_yaw_velocity = joint_vel[:, 5:6]
    root_height = joint_pos[:, 2:3]
    joint_angles = joint_pos[:, 7:]

    return np.concatenate(
        [
            root_yaw_velocity,
            root_local_xy_velocity,
            root_height,
            residual_6d,
            joint_angles,
        ],
        axis=-1,
    ).astype(np.float32)


def g1_vec_to_qpos(vec, fps=50.0, init_xy=None, init_yaw=0.0, clamp_joint_bounds=None):
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 2 or vec.shape[1] != 39:
        raise ValueError(f"Expected g1_vec shape (T, 39), got {vec.shape}")

    dt = 1.0 / float(fps)
    yaw_vel = vec[:, 0]
    local_xy_vel = vec[:, 1:3]
    root_height = vec[:, 3]
    residual_6d = vec[:, 4:10]
    joint_angles = vec[:, 10:].copy()

    if clamp_joint_bounds is not None:
        low, high = clamp_joint_bounds
        joint_angles = np.clip(joint_angles, low, high)

    yaw = np.empty(vec.shape[0], dtype=np.float64)
    yaw[0] = float(init_yaw)
    if vec.shape[0] > 1:
        yaw[1:] = yaw[0] + np.cumsum(yaw_vel[:-1] * dt)

    xy = np.zeros((vec.shape[0], 2), dtype=np.float64)
    if init_xy is not None:
        xy[0] = np.asarray(init_xy, dtype=np.float64)
    for i in range(1, vec.shape[0]):
        xy[i] = xy[i - 1] + rotate_xy_by_yaw(local_xy_vel[i - 1], yaw[i - 1]) * dt

    residual_quat = matrix_to_quat(rotation_6d_to_matrix(residual_6d))
    root_quat = quat_normalize(quat_mul(yaw_to_quat(yaw), residual_quat))
    root_xyz = np.column_stack([xy, root_height])
    return np.concatenate([root_xyz, root_quat, joint_angles], axis=-1).astype(np.float32)


class LAFANG1(torch.utils.data.Dataset):
    dataname = "lafan_g1"

    def __init__(
        self,
        split="train",
        num_frames=60,
        data_dir="",
        fixed_len=0,
        pred_len=0,
        motion_filter="*.npz",
        **_kwargs,
    ):
        super().__init__()
        self.split = split
        self.data_dir = Path(data_dir or DEFAULT_LAFAN_G1_DIR).expanduser()
        self.motion_filter = motion_filter
        self.fixed_len = int(fixed_len or num_frames or 60)
        self.pred_len = int(pred_len or 0)
        self.context_len = self.fixed_len - self.pred_len if self.pred_len > 0 else 0
        if self.pred_len > 0 and self.context_len <= 0:
            raise ValueError(
                f"fixed_len must be larger than pred_len, got fixed_len={self.fixed_len}, "
                f"pred_len={self.pred_len}"
            )

        self.motion_paths = filter_motion_paths(self.data_dir, self.motion_filter)
        self.motions = []
        self.index = []
        labels = sorted({parse_lafan_action(path.name) for path in self.motion_paths})
        self._action_classes = labels
        self._action_to_label = {action: idx for idx, action in enumerate(labels)}
        self._label_to_action = {idx: action for action, idx in self._action_to_label.items()}
        self.num_actions = len(labels)

        all_features = []
        fps_values = set()
        joint_names_ref = None
        for motion_idx, path in enumerate(self.motion_paths):
            data = np.load(path, allow_pickle=True)
            joint_pos = data["joint_pos"]
            joint_vel = data["joint_vel"]
            joint_names = data["joint_names"].tolist()
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            fps_values.add(fps)

            if joint_pos.shape[1] != 36 or joint_vel.shape[1] != 35 or len(joint_names) != 29:
                raise ValueError(
                    f"Unexpected G1 schema in {path}: joint_pos={joint_pos.shape}, "
                    f"joint_vel={joint_vel.shape}, joint_names={len(joint_names)}"
                )
            if joint_names_ref is None:
                joint_names_ref = joint_names
            elif joint_names != joint_names_ref:
                raise ValueError(f"Joint name order differs in {path}")

            features = qpos_qvel_to_g1_vec(joint_pos, joint_vel)
            action_name = parse_lafan_action(path.name)
            action = self._action_to_label[action_name]
            self.motions.append(
                {
                    "path": path,
                    "features": features,
                    "action": action,
                    "action_text": action_name,
                    "fps": fps,
                    "joint_pos_min": joint_pos[:, 7:].min(axis=0),
                    "joint_pos_max": joint_pos[:, 7:].max(axis=0),
                }
            )
            all_features.append(features)
            max_start = features.shape[0] - self.fixed_len
            for start in range(max_start + 1):
                self.index.append((motion_idx, start))

        if not self.index:
            raise ValueError(
                f"No motions in [{self.data_dir}] are at least {self.fixed_len} frames long"
            )

        stacked = np.concatenate(all_features, axis=0)
        self.mean = stacked.mean(axis=0).astype(np.float32)
        self.std = np.maximum(stacked.std(axis=0), 1e-6).astype(np.float32)
        self.nfeats = 1
        self.njoints = stacked.shape[1]
        self.feature_dim = stacked.shape[1]
        self.fps = sorted(fps_values)[0]
        self.joint_names = joint_names_ref
        self.joint_pos_min = np.min([m["joint_pos_min"] for m in self.motions], axis=0).astype(np.float32)
        self.joint_pos_max = np.max([m["joint_pos_max"] for m in self.motions], axis=0).astype(np.float32)

    def action_to_label(self, action):
        return self._action_to_label[action]

    def label_to_action(self, label):
        return self._label_to_action[int(label)]

    def action_name_to_action(self, action_name):
        if isinstance(action_name, (list, tuple, np.ndarray)):
            return np.array([self.action_to_label(name) for name in action_name])
        return self.action_to_label(action_name)

    def action_to_action_name(self, action):
        return self.label_to_action(action)

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        motion_idx, start = self.index[item]
        motion = self.motions[motion_idx]
        window = motion["features"][start:start + self.fixed_len]
        window = self.transform(window).astype(np.float32)
        tensor = torch.from_numpy(window.T).float().unsqueeze(1)

        output = {
            "action": motion["action"],
            "action_text": motion["action_text"],
            "key": f"{motion['path'].stem}:{start}",
        }
        if self.pred_len > 0:
            output["prefix"] = tensor[..., :self.context_len]
            output["inp"] = tensor[..., self.context_len:]
            output["lengths"] = self.pred_len
            output["orig_lengths"] = self.fixed_len
        else:
            output["inp"] = tensor
            output["lengths"] = self.fixed_len
        return output

    def save_metadata(self, save_dir):
        save_dir = Path(save_dir)
        np.save(save_dir / "lafan_g1_mean.npy", self.mean)
        np.save(save_dir / "lafan_g1_std.npy", self.std)
        metadata = {
            "data_dir": str(self.data_dir),
            "motion_filter": self.motion_filter,
            "num_motions": len(self.motion_paths),
            "num_windows": len(self.index),
            "fps": self.fps,
            "feature_dim": self.feature_dim,
            "feature_layout": FEATURE_LAYOUT,
            "joint_names": self.joint_names,
            "action_classes": self._action_classes,
            "robot_model": DEFAULT_G1_ROBOT_MODEL,
            "velocity_frame_convention": (
                "root linear velocity from joint_vel[:,0:3] rotated by inverse root yaw; "
                "yaw velocity from joint_vel[:,5]"
            ),
            "joint_pos_min": self.joint_pos_min.tolist(),
            "joint_pos_max": self.joint_pos_max.tolist(),
        }
        with open(save_dir / "lafan_g1_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
