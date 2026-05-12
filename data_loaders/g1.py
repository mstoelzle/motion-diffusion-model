import fnmatch
import json
import re
from pathlib import Path

import numpy as np
import torch

from utils.rotation_conversions_np import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_invert,
    quaternion_raw_multiply,
    quaternion_normalize,
    quaternion_to_matrix,
    quaternion_to_yaw,
    rotate_xy_by_inverse_yaw,
    rotate_xy_by_yaw,
    rotation_6d_to_matrix,
    yaw_to_quaternion,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LAFAN_G1_DIR = str(_REPO_ROOT / "dataset" / "lafan_g1")
DEFAULT_LATENT_TENNIS_G1_DIR = str(_REPO_ROOT / "dataset" / "latent_tennis_g1")
DEFAULT_LATENT_TENNIS_G1_EMBEDDINGS = str(
    _REPO_ROOT / "dataset" / "latent_tennis_g1" / "Random_001-004_Tennis_with_embeddings.npz"
)
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


def filter_motion_paths(data_dir, motion_filter="*.npz", recursive=False):
    if not data_dir:
        raise ValueError("data_dir must be specified for G1 motion path filtering")
    data_dir = Path(data_dir).expanduser()
    patterns = expand_motion_filter(motion_filter)
    glob_pattern = "**/*.npz" if recursive else "*.npz"
    paths = sorted(
        path for path in data_dir.glob(glob_pattern)
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)
    )
    if not paths:
        raise FileNotFoundError(
            f"No .npz files matched [{motion_filter}] in [{data_dir}]"
        )
    return paths


def matches_motion_filter(path, motion_filter):
    patterns = expand_motion_filter(motion_filter)
    return any(fnmatch.fnmatch(Path(path).name, pattern) for pattern in patterns)


def matches_motion_file(path, motion_file):
    if not motion_file:
        return True
    path = Path(path)
    motion_file = Path(motion_file).expanduser()
    if motion_file.is_absolute():
        return path.resolve() == motion_file.resolve()
    return path.name == motion_file.name


def parse_latent_tennis_action(_path_or_name):
    return "tennis"


def _first_existing(data, keys, path):
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"Expected one of {keys} in [{path}]")


def load_g1_motion_arrays(path):
    data = np.load(path, allow_pickle=True)
    joint_pos = _first_existing(data, ("joint_pos", "qpos"), path)
    joint_vel = _first_existing(data, ("joint_vel", "qvel"), path)
    fps = float(np.asarray(_first_existing(data, ("fps", "frequency"), path)).reshape(-1)[0])
    joint_names = _first_existing(data, ("joint_names",), path).tolist()
    if joint_names and joint_names[0] == "root":
        joint_names = joint_names[1:]
    return joint_pos, joint_vel, joint_names, fps


def finite_difference_root_velocities(joint_pos, fps):
    """Derive root-local XY and yaw velocities from consecutive qpos frames."""
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 36:
        raise ValueError(f"Expected joint_pos shape (T, 36), got {joint_pos.shape}")

    root_quaternion = quaternion_normalize(joint_pos[:, 3:7])
    yaw = np.unwrap(quaternion_to_yaw(root_quaternion))
    dt = 1.0 / float(fps)
    world_xy_velocity = np.zeros((joint_pos.shape[0], 2), dtype=np.float64)
    root_yaw_velocity = np.zeros((joint_pos.shape[0], 1), dtype=np.float64)

    if joint_pos.shape[0] > 1:
        world_xy_velocity[:-1] = (joint_pos[1:, 0:2] - joint_pos[:-1, 0:2]) / dt
        world_xy_velocity[-1] = world_xy_velocity[-2]
        root_yaw_velocity[:-1, 0] = np.diff(yaw) / dt
        root_yaw_velocity[-1] = root_yaw_velocity[-2]

    root_local_xy_velocity = rotate_xy_by_inverse_yaw(world_xy_velocity, yaw)
    return root_yaw_velocity, root_local_xy_velocity


def qpos_qvel_to_g1_vec(joint_pos, joint_vel, fps=50.0, root_velocity_source="qvel"):
    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    joint_vel = np.asarray(joint_vel, dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 36:
        raise ValueError(f"Expected joint_pos shape (T, 36), got {joint_pos.shape}")
    if joint_vel.ndim != 2 or joint_vel.shape[1] != 35:
        raise ValueError(f"Expected joint_vel shape (T, 35), got {joint_vel.shape}")
    if root_velocity_source not in {"qvel", "finite_difference"}:
        raise ValueError(f"Unsupported root_velocity_source [{root_velocity_source}]")

    root_quaternion = quaternion_normalize(joint_pos[:, 3:7])
    yaw = quaternion_to_yaw(root_quaternion)
    yaw_quaternion = yaw_to_quaternion(yaw)
    residual_quaternion = quaternion_raw_multiply(quaternion_invert(yaw_quaternion), root_quaternion)
    residual_6d = matrix_to_rotation_6d(quaternion_to_matrix(residual_quaternion))

    if root_velocity_source == "finite_difference":
        root_yaw_velocity, root_local_xy_velocity = finite_difference_root_velocities(joint_pos, fps)
    else:
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

    residual_quaternion = matrix_to_quaternion(rotation_6d_to_matrix(residual_6d))
    root_quaternion = quaternion_normalize(quaternion_raw_multiply(yaw_to_quaternion(yaw), residual_quaternion))
    root_xyz = np.column_stack([xy, root_height])
    return np.concatenate([root_xyz, root_quaternion, joint_angles], axis=-1).astype(np.float32)


class G1MotionDataset(torch.utils.data.Dataset):
    dataname = "g1"
    default_data_dir = ""
    metadata_stem = "g1"
    recursive_motion_paths = False
    root_velocity_source = "qvel"

    @classmethod
    def from_loader_args(
        cls,
        split="train",
        num_frames=60,
        abs_path="",
        fixed_len=0,
        pred_len=0,
        motion_filter="*.npz",
        prefix_motion_filter="",
        prefix_file="",
        prefix_start=-1,
        **_kwargs,
    ):
        return cls(
            split=split,
            num_frames=num_frames,
            data_dir=abs_path,
            fixed_len=fixed_len,
            pred_len=pred_len,
            motion_filter=motion_filter,
            prefix_motion_filter=prefix_motion_filter,
            prefix_file=prefix_file,
            prefix_start=prefix_start,
        )

    def __init__(
        self,
        split="train",
        num_frames=60,
        data_dir="",
        fixed_len=0,
        pred_len=0,
        motion_filter="*.npz",
        prefix_motion_filter="",
        prefix_file="",
        prefix_start=-1,
        **_kwargs,
    ):
        super().__init__()
        self.split = split
        self.data_dir = Path(data_dir or self.default_data_dir).expanduser()
        self.motion_filter = motion_filter
        self.prefix_motion_filter = prefix_motion_filter
        self.prefix_file = prefix_file
        self.prefix_start = int(prefix_start)
        self.fixed_len = int(fixed_len or num_frames or 60)
        self.pred_len = int(pred_len or 0)
        self.context_len = self.fixed_len - self.pred_len if self.pred_len > 0 else 0
        if self.pred_len > 0 and self.context_len <= 0:
            raise ValueError(
                f"fixed_len must be larger than pred_len, got fixed_len={self.fixed_len}, "
                f"pred_len={self.pred_len}"
            )

        self.motion_paths = filter_motion_paths(
            self.data_dir,
            self.motion_filter,
            recursive=self.recursive_motion_paths,
        )
        self.motions = []
        self.index = []
        labels = sorted({self.parse_action(path) for path in self.motion_paths})
        self._action_classes = labels
        self._action_to_label = {action: idx for idx, action in enumerate(labels)}
        self._label_to_action = {idx: action for action, idx in self._action_to_label.items()}
        self.num_actions = len(labels)

        all_features = []
        fps_values = set()
        joint_names_ref = None
        for motion_idx, path in enumerate(self.motion_paths):
            joint_pos, joint_vel, joint_names, fps = load_g1_motion_arrays(path)
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

            features = qpos_qvel_to_g1_vec(
                joint_pos,
                joint_vel,
                fps=fps,
                root_velocity_source=self.root_velocity_source,
            )
            action_name = self.parse_action(path)
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

            use_for_prefix = True
            if self.prefix_motion_filter:
                use_for_prefix = matches_motion_filter(path, self.prefix_motion_filter)
            if use_for_prefix and self.prefix_file:
                use_for_prefix = matches_motion_file(path, self.prefix_file)
            if not use_for_prefix:
                continue

            max_start = features.shape[0] - self.fixed_len
            if self.prefix_start >= 0:
                if self.prefix_start > max_start:
                    raise ValueError(
                        f"Requested prefix_start={self.prefix_start} for {path.name}, "
                        f"but valid starts are 0..{max_start}"
                    )
                self.index.append((motion_idx, self.prefix_start))
            else:
                for start in range(max_start + 1):
                    self.index.append((motion_idx, start))

        if not self.index:
            raise ValueError(
                f"No prefix windows matched prefix_motion_filter=[{self.prefix_motion_filter or '*'}], "
                f"prefix_file=[{self.prefix_file or '*'}] in [{self.data_dir}] with fixed_len={self.fixed_len}"
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

    def parse_action(self, path):
        raise NotImplementedError

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
        np.save(save_dir / f"{self.metadata_stem}_mean.npy", self.mean)
        np.save(save_dir / f"{self.metadata_stem}_std.npy", self.std)
        metadata = {
            "data_dir": str(self.data_dir),
            "motion_filter": self.motion_filter,
            "prefix_motion_filter": self.prefix_motion_filter,
            "prefix_file": self.prefix_file,
            "prefix_start": self.prefix_start,
            "num_motions": len(self.motion_paths),
            "num_windows": len(self.index),
            "fps": self.fps,
            "feature_dim": self.feature_dim,
            "feature_layout": FEATURE_LAYOUT,
            "joint_names": self.joint_names,
            "action_classes": self._action_classes,
            "robot_model": DEFAULT_G1_ROBOT_MODEL,
            "velocity_frame_convention": (
                "root local XY velocity is expressed in the root-yaw frame; "
                f"root velocity source is {self.root_velocity_source}"
            ),
            "joint_pos_min": self.joint_pos_min.tolist(),
            "joint_pos_max": self.joint_pos_max.tolist(),
        }
        with open(save_dir / f"{self.metadata_stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


class LAFANG1(G1MotionDataset):
    dataname = "lafan_g1"
    default_data_dir = DEFAULT_LAFAN_G1_DIR
    metadata_stem = "lafan_g1"
    recursive_motion_paths = False

    def parse_action(self, path):
        return parse_lafan_action(path.name)


class LatentTennisG1(G1MotionDataset):
    dataname = "latent_tennis_g1"
    default_data_dir = DEFAULT_LATENT_TENNIS_G1_DIR
    metadata_stem = "latent_tennis_g1"
    recursive_motion_paths = True
    root_velocity_source = "finite_difference"

    @classmethod
    def from_loader_args(
        cls,
        split="train",
        num_frames=60,
        abs_path="",
        fixed_len=0,
        pred_len=0,
        motion_filter="*.npz",
        prefix_motion_filter="",
        prefix_file="",
        prefix_start=-1,
        latent_embeddings_path="",
        **kwargs,
    ):
        if latent_embeddings_path:
            return LatentConditionedChunkDataset(
                split=split,
                num_frames=num_frames,
                latent_embeddings_path=latent_embeddings_path,
                fixed_len=fixed_len,
                pred_len=pred_len,
                prefix_start=prefix_start,
            )
        return super().from_loader_args(
            split=split,
            num_frames=num_frames,
            abs_path=abs_path,
            fixed_len=fixed_len,
            pred_len=pred_len,
            motion_filter=motion_filter,
            prefix_motion_filter=prefix_motion_filter,
            prefix_file=prefix_file,
            prefix_start=prefix_start,
            **kwargs,
        )

    def parse_action(self, path):
        return parse_latent_tennis_action(path.name)


class LatentConditionedChunkDataset(torch.utils.data.Dataset):
    dataname = "latent_tennis_g1"
    metadata_stem = "latent_tennis_g1"
    default_embeddings_path = DEFAULT_LATENT_TENNIS_G1_EMBEDDINGS

    def __init__(
        self,
        split="train",
        num_frames=0,
        latent_embeddings_path="",
        fixed_len=0,
        pred_len=0,
        prefix_start=-1,
        **_kwargs,
    ):
        super().__init__()
        self.split = split
        self.path = Path(latent_embeddings_path or self.default_embeddings_path).expanduser()
        with np.load(self.path, allow_pickle=False) as data:
            self.states = np.asarray(data["states"], dtype=np.float32)
            self.latents = np.asarray(data["latents"], dtype=np.float32)
            self.chunks = np.asarray(data["chunks"], dtype=np.float32)

        if self.states.ndim != 2:
            raise ValueError(f"Expected states shape (N, D), got {self.states.shape}")
        if self.latents.ndim != 2:
            raise ValueError(f"Expected latents shape (N, Z), got {self.latents.shape}")
        if self.chunks.ndim != 3:
            raise ValueError(f"Expected chunks shape (N, T, D), got {self.chunks.shape}")
        if self.states.shape[0] != self.latents.shape[0] or self.states.shape[0] != self.chunks.shape[0]:
            raise ValueError(
                f"states, latents, and chunks must have the same row count; got "
                f"{self.states.shape[0]}, {self.latents.shape[0]}, {self.chunks.shape[0]}"
            )
        if self.states.shape[1] != self.chunks.shape[2]:
            raise ValueError(f"State dim {self.states.shape[1]} does not match chunk dim {self.chunks.shape[2]}")
        if not np.allclose(self.states, self.chunks[:, 0, :], atol=1e-6):
            raise ValueError(f"states must match chunks[:, 0, :] in [{self.path}]")

        self.chunk_len = self.chunks.shape[1]
        self.feature_dim = self.chunks.shape[2]
        self.latent_cond_dim = self.latents.shape[1]
        self.njoints = self.feature_dim
        self.nfeats = 1
        self.num_actions = 1
        self.fps = 50

        if pred_len is None or pred_len <= 0:
            self.context_len = 1
            self.pred_len = self.chunk_len - self.context_len
            self.fixed_len = self.chunk_len
        else:
            self.pred_len = int(pred_len)
            self.fixed_len = int(fixed_len or num_frames or self.pred_len + 1)
            self.context_len = self.fixed_len - self.pred_len
        if self.context_len <= 0:
            raise ValueError(f"context_len must be positive for latent chunk conditioning, got {self.context_len}")
        if self.fixed_len > self.chunk_len:
            raise ValueError(f"fixed_len={self.fixed_len} exceeds latent chunk length {self.chunk_len}")
        if self.context_len + self.pred_len > self.chunk_len:
            raise ValueError(
                f"context_len + pred_len must be <= {self.chunk_len}, got "
                f"{self.context_len} + {self.pred_len}"
            )

        flat_chunks = self.chunks.reshape(-1, self.feature_dim)
        self.mean = flat_chunks.mean(axis=0).astype(np.float32)
        self.std = np.maximum(flat_chunks.std(axis=0), 1e-6).astype(np.float32)
        self.latent_mean = self.latents.mean(axis=0).astype(np.float32)
        self.latent_std = np.maximum(self.latents.std(axis=0), 1e-6).astype(np.float32)
        self.prefix_start = int(prefix_start)
        if self.prefix_start >= 0:
            if self.prefix_start >= self.chunks.shape[0]:
                raise ValueError(
                    f"Requested prefix_start={self.prefix_start}, but valid latent row indices are "
                    f"0..{self.chunks.shape[0] - 1}"
                )
            self.index = [self.prefix_start]
        else:
            self.index = list(range(self.chunks.shape[0]))

    def __len__(self):
        return len(self.index)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform_latent(self, data):
        return (data - self.latent_mean) / self.latent_std

    def __getitem__(self, item):
        row_idx = self.index[item]
        window = self.transform(self.chunks[row_idx, :self.fixed_len]).astype(np.float32)
        tensor = torch.from_numpy(window.T).float().unsqueeze(1)
        latent = torch.from_numpy(self.transform_latent(self.latents[row_idx]).astype(np.float32)).float()
        return {
            "prefix": tensor[..., :self.context_len],
            "inp": tensor[..., self.context_len:self.context_len + self.pred_len],
            "latent": latent,
            "lengths": self.pred_len,
            "orig_lengths": self.fixed_len,
            "key": f"{self.path.stem}:{row_idx}",
        }

    def save_metadata(self, save_dir):
        save_dir = Path(save_dir)
        np.save(save_dir / f"{self.metadata_stem}_mean.npy", self.mean)
        np.save(save_dir / f"{self.metadata_stem}_std.npy", self.std)
        np.save(save_dir / f"{self.metadata_stem}_latent_mean.npy", self.latent_mean)
        np.save(save_dir / f"{self.metadata_stem}_latent_std.npy", self.latent_std)
        metadata = {
            "data_dir": str(self.path.parent),
            "latent_embeddings_path": str(self.path),
            "num_windows": len(self),
            "chunk_len": self.chunk_len,
            "context_len": self.context_len,
            "pred_len": self.pred_len,
            "feature_dim": self.feature_dim,
            "latent_cond_dim": self.latent_cond_dim,
            "motion_format": "latent_motion_chunk",
            "conditioning_note": "no_cond means no semantic conditioning; prefix context may still condition the model",
        }
        with open(save_dir / f"{self.metadata_stem}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
