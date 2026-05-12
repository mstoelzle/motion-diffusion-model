import numpy as np

from data_loaders.g1 import qpos_qvel_to_g1_vec
from visualize.export_lafan_g1_to_viser import export_all, export_single, load_results


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


def _write_results(path):
    motion = np.arange(2 * 3 * 5 * 36, dtype=np.float32).reshape(6, 5, 36)
    results = {
        "motion": motion,
        "motion_format": "g1_qpos",
        "text": ["dance"] * 6,
        "lengths": np.array([5, 4, 3, 5, 4, 3]),
        "num_samples": 3,
        "num_repetitions": 2,
        "fps": 50,
        "joint_names": ["j0", "j1"],
    }
    np.save(path, results)
    return results


def _yaw_quat(yaw):
    half = yaw * 0.5
    return np.stack(
        [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)],
        axis=-1,
    )


def _write_source_qpos(path, num_frames=8):
    t = np.arange(num_frames, dtype=np.float64)
    qpos = np.zeros((num_frames, 36), dtype=np.float64)
    qpos[:, 0] = 0.03 * t
    qpos[:, 1] = -0.02 * t
    qpos[:, 2] = 0.72 + 0.005 * t
    qpos[:, 3:7] = _yaw_quat(0.04 * t)
    qpos[:, 7:] = 0.02 * t[:, None] + np.linspace(-0.15, 0.15, 29)[None]
    qvel = np.zeros((num_frames, 35), dtype=np.float64)
    np.savez(
        path,
        qpos=qpos,
        qvel=qvel,
        frequency=np.array(50.0),
        joint_names=np.array(["root"] + JOINT_NAMES),
    )
    return qpos, qvel


def _write_latent_embeddings(path, qpos, qvel, chunk_len=4, feature_dim=80):
    g1_vec = qpos_qvel_to_g1_vec(
        qpos,
        qvel,
        fps=50.0,
        root_velocity_source="finite_difference",
    )
    chunks = []
    for start in range(qpos.shape[0] - chunk_len + 1):
        chunk = np.zeros((chunk_len, feature_dim), dtype=np.float32)
        chunk[:, :g1_vec.shape[1]] = g1_vec[start:start + chunk_len]
        chunks.append(chunk)
    chunks = np.stack(chunks, axis=0)
    states = chunks[:, 0, :].copy()
    latents = np.zeros((chunks.shape[0], 3), dtype=np.float32)
    np.savez(path, states=states, latents=latents, chunks=chunks)
    return chunks


def test_export_single_to_viser_npz(tmp_path):
    expected = _write_results(tmp_path / "results.npy")
    results = load_results(tmp_path / "results.npy")
    output_path = export_single(results, tmp_path / "sample.npz", sample_idx=1, rep_idx=1)

    data = np.load(output_path, allow_pickle=True)
    flat_idx = 1 * expected["num_samples"] + 1
    assert data["qpos"].shape == (4, 36)
    assert np.allclose(data["qpos"], expected["motion"][flat_idx, :4])
    assert int(data["fps"]) == 50
    assert str(data["text"]) == "dance"
    assert int(data["length"]) == 4


def test_export_all_to_viser_npz(tmp_path):
    _write_results(tmp_path / "results.npy")
    results = load_results(tmp_path / "results.npy")
    paths = export_all(results, tmp_path / "viser_npz")

    assert len(paths) == 6
    assert paths[0].name == "sample00_rep00_dance.npz"
    assert paths[-1].name == "sample02_rep01_dance.npz"
    assert np.load(paths[-1], allow_pickle=True)["qpos"].shape == (3, 36)


def test_load_results_converts_latent_motion_chunk_to_g1_qpos(tmp_path):
    source_dir = tmp_path / "latent_tennis_g1" / "p1"
    source_dir.mkdir(parents=True)
    qpos, qvel = _write_source_qpos(source_dir / "Random_001_Tennis 001.npz")
    embeddings_path = tmp_path / "latent_tennis_g1" / "Random_001_Tennis_with_embeddings.npz"
    chunks = _write_latent_embeddings(embeddings_path, qpos, qvel)

    results = {
        "motion": chunks[1:2, 1:],
        "motion_format": "latent_motion_chunk",
        "text": ["latent"],
        "lengths": np.array([3]),
        "num_samples": 1,
        "num_repetitions": 1,
        "context_len": 1,
        "prefix_sources": [f"{embeddings_path.stem}:1"],
        "latent_embeddings_path": str(embeddings_path),
        "latent_data_dir": str(tmp_path / "latent_tennis_g1"),
    }
    np.save(tmp_path / "latent_results.npy", results)

    converted = load_results(tmp_path / "latent_results.npy")

    assert converted["motion_format"] == "g1_qpos"
    assert converted["latent_motion_format"] == "latent_motion_chunk"
    assert converted["motion"].shape == (1, 4, 36)
    assert converted["lengths"].tolist() == [4]
    assert np.isfinite(converted["motion"]).all()
