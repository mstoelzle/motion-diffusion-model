import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_loaders.g1 import (
    G1MotionDataset,
    LAFANG1,
    LatentConditionedChunkDataset,
    LatentTennisG1,
    finite_difference_root_velocities,
    filter_motion_paths,
    g1_vec_to_qpos,
    parse_lafan_action,
    qpos_qvel_to_g1_vec,
)
from data_loaders.get_data import get_dataset, resolve_dataset_defaults
from data_loaders.tensors import collate
from sample.generate import main as generate
from sample.generate_latent_interpolation import (
    build_interpolation_batch,
    generate_latent_interpolation,
    interpolate_normalized_latents,
)
from utils.model_util import create_model_and_diffusion
from utils.rotation_conversions_np import quaternion_to_yaw


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


def _write_latent_tennis_motion(path, num_frames=8, yaw_offset=0.0):
    joint_pos, joint_vel = _write_motion(path, num_frames=num_frames, yaw_offset=yaw_offset)
    path.unlink()
    np.savez(
        path,
        qpos=joint_pos,
        qvel=np.zeros_like(joint_vel),
        frequency=np.array(50.0),
        split_points=np.array([0, num_frames], dtype=np.int32),
        joint_names=np.array(["root"] + JOINT_NAMES),
        body_names=np.array([f"body_{i}" for i in range(31)]),
    )
    return joint_pos, joint_vel


def _write_embedding_chunks(path, num_rows=4, chunk_len=6, feature_dim=5, latent_dim=3):
    chunks = np.arange(num_rows * chunk_len * feature_dim, dtype=np.float32).reshape(
        num_rows, chunk_len, feature_dim
    )
    states = chunks[:, 0, :].copy()
    latents = np.linspace(-1.0, 1.0, num_rows * latent_dim, dtype=np.float32).reshape(num_rows, latent_dim)
    np.savez(path, states=states, latents=latents, chunks=chunks)
    return states, latents, chunks


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


def test_finite_difference_root_velocity_reconstructs_xy_and_yaw(tmp_path):
    joint_pos, joint_vel = _write_motion(tmp_path / "tennis_like.npz", num_frames=8, yaw_offset=0.3)
    zero_joint_vel = np.zeros_like(joint_vel)
    vec = qpos_qvel_to_g1_vec(
        joint_pos,
        zero_joint_vel,
        fps=50,
        root_velocity_source="finite_difference",
    )
    rec = g1_vec_to_qpos(
        vec,
        fps=50,
        init_xy=joint_pos[0, :2],
        init_yaw=quaternion_to_yaw(joint_pos[0, 3:7]),
    )

    assert not np.allclose(vec[:, :3], 0.0)
    assert np.allclose(rec[:, :2], joint_pos[:, :2], atol=1e-6)
    assert np.allclose(
        np.unwrap(quaternion_to_yaw(rec[:, 3:7])),
        np.unwrap(quaternion_to_yaw(joint_pos[:, 3:7])),
        atol=1e-6,
    )


def test_finite_difference_root_velocity_matches_forward_difference(tmp_path):
    joint_pos, _ = _write_motion(tmp_path / "tennis_like.npz", num_frames=8, yaw_offset=0.3)
    yaw_velocity, local_xy_velocity = finite_difference_root_velocities(joint_pos, fps=50)

    assert yaw_velocity.shape == (8, 1)
    assert local_xy_velocity.shape == (8, 2)
    assert np.allclose(yaw_velocity[:-1, 0], 1.0, atol=1e-6)
    assert np.allclose(yaw_velocity[-1], yaw_velocity[-2])


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


def test_latent_conditioned_chunk_dataset_defaults_to_one_state_prefix(tmp_path):
    path = tmp_path / "embeddings.npz"
    states, latents, chunks = _write_embedding_chunks(path, chunk_len=6)

    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path)

    assert dataset.context_len == 1
    assert dataset.pred_len == 5
    assert dataset.fixed_len == 6
    assert dataset.feature_dim == chunks.shape[-1]
    assert dataset.latent_cond_dim == latents.shape[-1]

    item = dataset[0]
    assert item["prefix"].shape == (chunks.shape[-1], 1, 1)
    assert item["inp"].shape == (chunks.shape[-1], 1, 5)
    denorm_prefix = dataset.inv_transform(item["prefix"].squeeze(1).T.numpy())
    denorm_target = dataset.inv_transform(item["inp"].squeeze(1).T.numpy())
    assert np.allclose(denorm_prefix[0], states[0])
    assert np.allclose(denorm_target[0], chunks[0, 1])


def test_resolve_dataset_defaults_keeps_latent_defaults_in_data_layer(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, chunk_len=6)
    args = SimpleNamespace(
        dataset="latent_tennis_g1",
        cond_mode="latent",
        latent_embeddings_path=str(path),
        context_len=None,
        pred_len=None,
    )

    resolve_dataset_defaults(args)

    assert args.context_len == 1
    assert args.pred_len == 5


def test_resolve_dataset_defaults_allows_explicit_zero_latent_context(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, chunk_len=6)
    args = SimpleNamespace(
        dataset="latent_tennis_g1",
        cond_mode="latent",
        latent_embeddings_path=str(path),
        context_len=0,
        pred_len=6,
    )

    resolve_dataset_defaults(args)

    assert args.context_len == 0
    assert args.pred_len == 6


def test_resolve_dataset_defaults_preserves_legacy_prefix_rule():
    args = SimpleNamespace(
        dataset="latent_tennis_g1",
        cond_mode="no_cond",
        latent_embeddings_path=None,
        context_len=20,
        pred_len=None,
    )

    resolve_dataset_defaults(args)

    assert args.context_len == 20
    assert args.pred_len == 20
    assert args.latent_embeddings_path == ""


def test_latent_conditioned_chunk_dataset_rejects_too_long_window(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, chunk_len=6)

    with pytest.raises(ValueError, match="exceeds latent chunk length"):
        LatentConditionedChunkDataset(latent_embeddings_path=path, fixed_len=7, pred_len=6)


def test_latent_conditioned_chunk_dataset_allows_zero_state_prefix(tmp_path):
    path = tmp_path / "embeddings.npz"
    _, _, chunks = _write_embedding_chunks(path, chunk_len=6)

    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path, fixed_len=6, pred_len=6)

    assert dataset.context_len == 0
    assert dataset.pred_len == 6
    assert dataset.fixed_len == 6
    item = dataset[0]
    assert item["prefix"].shape == (chunks.shape[-1], 1, 0)
    assert item["inp"].shape == (chunks.shape[-1], 1, 6)
    denorm_target = dataset.inv_transform(item["inp"].squeeze(1).T.numpy())
    assert np.allclose(denorm_target, chunks[0])


def test_latent_conditioned_collate_includes_prefix_and_latent(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, num_rows=3, chunk_len=6)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path)

    motion, cond = collate([dataset[0], dataset[1]])

    assert motion.shape == (2, dataset.feature_dim, 1, dataset.pred_len)
    assert cond["y"]["prefix"].shape == (2, dataset.feature_dim, 1, dataset.context_len)
    assert cond["y"]["latent"].shape == (2, dataset.latent_cond_dim)


def test_latent_conditioned_prefix_start_selects_embedding_row(tmp_path):
    path = tmp_path / "embeddings.npz"
    states, _, _ = _write_embedding_chunks(path, num_rows=3, chunk_len=6)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path, prefix_start=2)

    item = dataset[0]
    denorm_prefix = dataset.inv_transform(item["prefix"].squeeze(1).T.numpy())

    assert len(dataset) == 1
    assert item["key"] == "embeddings:2"
    assert np.allclose(denorm_prefix[0], states[2])


def test_get_dataset_uses_latent_tennis_loader_hook_for_embeddings(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, num_rows=3, chunk_len=6)

    dataset = get_dataset(
        "latent_tennis_g1",
        num_frames=6,
        fixed_len=6,
        pred_len=5,
        latent_embeddings_path=str(path),
        prefix_start=1,
    )

    assert isinstance(dataset, LatentConditionedChunkDataset)
    assert len(dataset) == 1
    assert dataset[0]["key"] == "embeddings:1"


def test_latent_conditioned_mdm_forward_smoke(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, num_rows=3, chunk_len=6, feature_dim=5, latent_dim=3)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path)
    motion, cond = collate([dataset[0], dataset[1]])
    loader = SimpleNamespace(dataset=dataset)
    args = SimpleNamespace(
        dataset="latent_tennis_g1",
        latent_dim=16,
        layers=1,
        cond_mask_prob=0.1,
        arch="trans_dec",
        emb_trans_dec=False,
        text_encoder_type="clip",
        pos_embed_max_len=32,
        mask_frames=True,
        pred_len=dataset.pred_len,
        context_len=dataset.context_len,
        cond_mode="latent",
        diffusion_steps=10,
        noise_schedule="cosine",
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_target_loc=0.0,
    )

    model, _ = create_model_and_diffusion(args, loader)
    output = model(motion, torch.zeros(motion.shape[0], dtype=torch.long), cond["y"])

    assert output.shape == motion.shape


def test_latent_conditioned_mdm_forward_smoke_without_prefix(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, num_rows=3, chunk_len=6, feature_dim=5, latent_dim=3)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path, fixed_len=6, pred_len=6)
    motion, cond = collate([dataset[0], dataset[1]])
    loader = SimpleNamespace(dataset=dataset)
    args = SimpleNamespace(
        dataset="latent_tennis_g1",
        latent_dim=16,
        layers=1,
        cond_mask_prob=0.1,
        arch="trans_dec",
        emb_trans_dec=False,
        text_encoder_type="clip",
        pos_embed_max_len=32,
        mask_frames=True,
        pred_len=dataset.pred_len,
        context_len=dataset.context_len,
        cond_mode="latent",
        diffusion_steps=10,
        noise_schedule="cosine",
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_target_loc=0.0,
    )

    model, _ = create_model_and_diffusion(args, loader)
    output = model(motion, torch.zeros(motion.shape[0], dtype=torch.long), cond["y"])

    assert output.shape == motion.shape


def test_latent_conditioned_generate_writes_raw_chunk_results(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, num_rows=3, chunk_len=6, feature_dim=5, latent_dim=3)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path)
    loader = SimpleNamespace(dataset=dataset)
    args = SimpleNamespace(
        seed=10,
        cuda=False,
        device=-1,
        dataset="latent_tennis_g1",
        data_dir="",
        motion_filter="*.npz",
        latent_embeddings_path=str(path),
        save_dir=str(tmp_path),
        model_path=str(tmp_path / "model000000000.pt"),
        output_dir=str(tmp_path / "samples"),
        num_samples=2,
        num_repetitions=1,
        guidance_param=1.0,
        gen_guidance_param=1.0,
        motion_length=1.0,
        input_text="",
        text_prompt="",
        dynamic_text_path="",
        action_file="",
        action_name="",
        prefix_motion_filter="",
        prefix_file="",
        prefix_start=-1,
        autoregressive=False,
        autoregressive_include_prefix=False,
        autoregressive_init="data",
        use_ema=False,
        latent_dim=16,
        layers=1,
        cond_mask_prob=0.1,
        arch="trans_dec",
        emb_trans_dec=False,
        text_encoder_type="clip",
        pos_embed_max_len=32,
        mask_frames=True,
        pred_len=dataset.pred_len,
        context_len=dataset.context_len,
        cond_mode="latent",
        diffusion_steps=2,
        noise_schedule="cosine",
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_target_loc=0.0,
        unconstrained=False,
    )
    model, _ = create_model_and_diffusion(args, loader)
    torch.save(model.state_dict(), args.model_path)

    out_dir = generate(args)
    results = np.load(tmp_path / "samples" / "results.npy", allow_pickle=True).item()

    assert out_dir == args.output_dir
    assert results["motion_format"] == "latent_motion_chunk"
    assert results["motion"].shape == (args.num_samples, dataset.pred_len, dataset.feature_dim)
    assert results["lengths"].tolist() == [dataset.pred_len, dataset.pred_len]
    assert results["feature_dim"] == dataset.feature_dim
    assert results["latent_cond_dim"] == dataset.latent_cond_dim
    assert results["latent_embeddings_path"] == str(path)
    assert "latent_data_dir" in results


def test_latent_interpolation_uses_normalized_linear_space(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, num_rows=3, chunk_len=6, feature_dim=5, latent_dim=3)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path)

    latents = interpolate_normalized_latents(dataset, row_a=0, row_b=2, alphas=[0.0, 0.5, 1.0])
    latent_a = dataset.transform_latent(dataset.latents[0])
    latent_b = dataset.transform_latent(dataset.latents[2])

    assert np.allclose(latents[0], latent_a)
    assert np.allclose(latents[1], 0.5 * latent_a + 0.5 * latent_b)
    assert np.allclose(latents[2], latent_b)


def test_latent_interpolation_batch_shares_prefix_and_varies_latent(tmp_path):
    path = tmp_path / "embeddings.npz"
    _, _, chunks = _write_embedding_chunks(path, num_rows=4, chunk_len=6, feature_dim=5, latent_dim=3)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path)

    motion, cond = build_interpolation_batch(
        dataset,
        row_a=0,
        row_b=3,
        prefix_row=2,
        alphas=[0.0, 0.5, 1.0],
    )
    denorm_prefix = dataset.inv_transform(cond["y"]["prefix"].squeeze(2).permute(0, 2, 1).numpy())

    assert motion.shape == (3, dataset.feature_dim, 1, dataset.pred_len)
    assert cond["y"]["lengths"].tolist() == [dataset.pred_len] * 3
    assert np.allclose(denorm_prefix, np.repeat(chunks[2:3, :dataset.context_len], 3, axis=0))
    assert cond["y"]["db_key"] == [f"{path.stem}:2"] * 3
    assert not torch.allclose(cond["y"]["latent"][0], cond["y"]["latent"][-1])


def test_latent_interpolation_generate_writes_results_metadata(tmp_path):
    path = tmp_path / "embeddings.npz"
    _write_embedding_chunks(path, num_rows=4, chunk_len=6, feature_dim=5, latent_dim=3)
    dataset = LatentConditionedChunkDataset(latent_embeddings_path=path)
    loader = SimpleNamespace(dataset=dataset)
    args = SimpleNamespace(
        seed=10,
        cuda=False,
        device=-1,
        dataset="latent_tennis_g1",
        data_dir="",
        motion_filter="*.npz",
        latent_embeddings_path=str(path),
        save_dir=str(tmp_path),
        model_path=str(tmp_path / "model000000000.pt"),
        output_dir=str(tmp_path / "latent_interpolation"),
        row_a=0,
        row_b=3,
        prefix_row=2,
        alphas=[0.0, 0.5, 1.0],
        num_repetitions=1,
        guidance_param=1.0,
        fixed_noise=True,
        export_viser=False,
        skip_endpoint_references=False,
        latent_dim=16,
        layers=1,
        cond_mask_prob=0.1,
        arch="trans_dec",
        emb_trans_dec=False,
        text_encoder_type="clip",
        pos_embed_max_len=32,
        mask_frames=True,
        pred_len=dataset.pred_len,
        context_len=dataset.context_len,
        cond_mode="latent",
        diffusion_steps=2,
        noise_schedule="cosine",
        sigma_small=True,
        lambda_vel=0.0,
        lambda_rcxyz=0.0,
        lambda_fc=0.0,
        lambda_target_loc=0.0,
        unconstrained=False,
        use_ema=False,
    )
    model, _ = create_model_and_diffusion(args, loader)
    torch.save(model.state_dict(), args.model_path)

    out_dir = generate_latent_interpolation(args)
    results = np.load(tmp_path / "latent_interpolation" / "results.npy", allow_pickle=True).item()
    references = np.load(tmp_path / "latent_interpolation" / "endpoint_references.npy", allow_pickle=True).item()

    assert out_dir == args.output_dir
    assert results["motion_format"] == "latent_motion_chunk"
    assert results["motion"].shape == (3, dataset.pred_len, dataset.feature_dim)
    assert results["row_a"] == 0
    assert results["row_b"] == 3
    assert results["prefix_row"] == 2
    assert results["latent_interpolation_space"] == "normalized_linear"
    assert bool(results["fixed_noise"])
    assert results["alpha_values"].tolist() == [0.0, 0.5, 1.0]
    assert results["prefix_sources"] == [f"{path.stem}:2"] * 3
    assert references["motion"].shape == (2, dataset.pred_len, dataset.feature_dim)
    assert references["prefix_sources"] == [f"{path.stem}:0", f"{path.stem}:3"]


def test_latent_tennis_g1_uses_g1_vec_schema_from_nested_qpos_files(tmp_path):
    player_dir = tmp_path / "p1"
    player_dir.mkdir()
    _write_latent_tennis_motion(player_dir / "Random_001_Tennis 001.npz")
    _write_latent_tennis_motion(player_dir / "Random_002_Tennis 001.npz", yaw_offset=0.3)

    dataset = LatentTennisG1(data_dir=tmp_path, fixed_len=6)

    assert len(dataset.motion_paths) == 2
    assert dataset.dataname == "latent_tennis_g1"
    assert dataset.root_velocity_source == "finite_difference"
    assert dataset.num_actions == 1
    assert dataset.action_to_action_name(0) == "tennis"
    assert dataset.feature_dim == 39
    assert dataset.joint_names == JOINT_NAMES

    item = dataset[0]
    assert item["inp"].shape == (39, 1, 6)
    assert item["lengths"] == 6
    assert item["action_text"] == "tennis"
    assert not np.allclose(dataset.motions[0]["features"][:, :3], 0.0)

    dataset.save_metadata(tmp_path)
    metadata = json.loads((tmp_path / "latent_tennis_g1_metadata.json").read_text())
    assert metadata["feature_dim"] == 39
    assert metadata["velocity_frame_convention"].endswith("finite_difference")
    assert metadata["joint_names"] == JOINT_NAMES
    assert issubclass(LAFANG1, G1MotionDataset)
    assert issubclass(LatentTennisG1, G1MotionDataset)
