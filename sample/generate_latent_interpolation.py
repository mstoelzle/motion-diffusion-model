import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_loaders.get_data import resolve_dataset_defaults
from data_loaders.g1 import LatentConditionedChunkDataset
from data_loaders.tensors import collate
from sample.generate import load_dataset
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import (
    add_base_options,
    parse_and_load_from_model,
)
from utils.sampler_util import ClassifierFreeSampleModel


def latent_interpolation_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a controlled latent interpolation demo for latent-conditioned "
            "latent_tennis_g1 checkpoints."
        )
    )
    add_base_options(parser)
    group = parser.add_argument_group("latent_interpolation")
    group.add_argument("--model_path", required=True, type=str)
    group.add_argument("--output_dir", required=True, type=str)
    group.add_argument("--row_a", required=True, type=int)
    group.add_argument("--row_b", required=True, type=int)
    group.add_argument("--prefix_row", default=None, type=int)
    group.add_argument("--alphas", default="0,0.25,0.5,0.75,1", type=str)
    group.add_argument("--num_repetitions", default=1, type=int)
    group.add_argument("--guidance_param", default=1.0, type=float)
    group.add_argument("--fixed_noise", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--export_viser", action="store_true")
    group.add_argument("--skip_endpoint_references", action="store_true")
    return resolve_dataset_defaults(parse_and_load_from_model(parser))


def parse_alphas(alphas):
    if isinstance(alphas, str):
        values = [float(value.strip()) for value in alphas.split(",") if value.strip()]
    else:
        values = [float(value) for value in alphas]
    if not values:
        raise ValueError("At least one interpolation alpha must be provided")
    return values


def _validate_row(dataset, row, name):
    row = int(row)
    if row < 0 or row >= dataset.chunks.shape[0]:
        raise ValueError(f"{name}={row} is outside valid row range 0..{dataset.chunks.shape[0] - 1}")
    return row


def interpolate_normalized_latents(dataset, row_a, row_b, alphas):
    row_a = _validate_row(dataset, row_a, "row_a")
    row_b = _validate_row(dataset, row_b, "row_b")
    alphas = np.asarray(parse_alphas(alphas), dtype=np.float32)
    latent_a = dataset.transform_latent(dataset.latents[row_a])
    latent_b = dataset.transform_latent(dataset.latents[row_b])
    return ((1.0 - alphas[:, None]) * latent_a[None] + alphas[:, None] * latent_b[None]).astype(np.float32)


def build_interpolation_batch(dataset, row_a, row_b, prefix_row=None, alphas=None):
    if not isinstance(dataset, LatentConditionedChunkDataset):
        raise TypeError("Latent interpolation requires LatentConditionedChunkDataset")
    alphas = parse_alphas("0,0.25,0.5,0.75,1" if alphas is None else alphas)
    row_a = _validate_row(dataset, row_a, "row_a")
    row_b = _validate_row(dataset, row_b, "row_b")
    prefix_row = row_a if prefix_row is None else _validate_row(dataset, prefix_row, "prefix_row")

    prefix = dataset.transform(dataset.chunks[prefix_row, :dataset.context_len]).astype(np.float32)
    prefix = torch.from_numpy(prefix.T).float().unsqueeze(1)
    latents = interpolate_normalized_latents(dataset, row_a, row_b, alphas)
    zero_inp = torch.zeros(dataset.feature_dim, 1, dataset.pred_len, dtype=torch.float32)
    key = f"{dataset.path.stem}:{prefix_row}"
    batch = [
        {
            "prefix": prefix,
            "inp": zero_inp.clone(),
            "latent": torch.from_numpy(latent).float(),
            "lengths": dataset.pred_len,
            "orig_lengths": dataset.fixed_len,
            "key": key,
        }
        for latent in latents
    ]
    return collate(batch)


def _to_device(model_kwargs):
    model_kwargs["y"] = {
        key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs["y"].items()
    }
    return model_kwargs


def _latent_results_dict(dataset, motion, lengths, text, args, prefix_sources, extra=None):
    results = {
        "motion": motion,
        "text": text,
        "lengths": lengths,
        "num_samples": len(text) // int(args.num_repetitions),
        "num_repetitions": int(args.num_repetitions),
        "motion_format": "latent_motion_chunk",
        "feature_dim": dataset.feature_dim,
        "latent_cond_dim": dataset.latent_cond_dim,
        "context_len": dataset.context_len,
        "prefix_sources": prefix_sources,
        "latent_embeddings_path": str(dataset.path),
        "latent_data_dir": str(getattr(dataset, "source_data_dir", getattr(args, "data_dir", ""))),
    }
    if extra:
        results.update(extra)
    return results


def save_endpoint_references(dataset, args, output_dir):
    rows = [int(args.row_a), int(args.row_b)]
    motion = np.stack([
        dataset.chunks[row, dataset.context_len:dataset.context_len + dataset.pred_len]
        for row in rows
    ], axis=0).astype(np.float32)
    text = [f"reference_row_{row}" for row in rows]
    prefix_sources = [f"{dataset.path.stem}:{row}" for row in rows]
    results = _latent_results_dict(
        dataset,
        motion,
        np.full(len(rows), dataset.pred_len, dtype=np.int64),
        text,
        SimpleNamespace(num_repetitions=1, data_dir=getattr(args, "data_dir", "")),
        prefix_sources,
        extra={
            "reference_rows": np.asarray(rows, dtype=np.int64),
            "reference_note": "Dataset endpoint futures for latent interpolation comparison.",
        },
    )
    path = Path(output_dir) / "endpoint_references.npy"
    np.save(path, results)
    return path


def export_viser_results(results_path, output_dir):
    from visualize.export_lafan_g1_to_viser import export_all, load_results

    results = load_results(results_path)
    return export_all(results, output_dir)


def generate_latent_interpolation(args=None):
    if args is None:
        args = latent_interpolation_args()
    args = resolve_dataset_defaults(args)
    if args.dataset != "latent_tennis_g1" or args.cond_mode != "latent":
        raise ValueError("Latent interpolation generation requires --dataset latent_tennis_g1 and --cond_mode latent")

    alphas = parse_alphas(args.alphas)
    args.prefix_row = int(args.row_a) if getattr(args, "prefix_row", None) is None else int(args.prefix_row)
    args.num_samples = len(alphas)
    args.batch_size = len(alphas)
    args.num_repetitions = int(getattr(args, "num_repetitions", 1))

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print("Loading latent interpolation dataset...")
    data = load_dataset(args, args.context_len + args.pred_len, args.pred_len)
    dataset = data.dataset
    if not isinstance(dataset, LatentConditionedChunkDataset):
        raise TypeError("Expected LatentConditionedChunkDataset for latent interpolation")

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    use_cfg = args.guidance_param != 1 and model.cond_mode != "no_cond"
    if use_cfg:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    _motion, model_kwargs = build_interpolation_batch(
        dataset,
        row_a=args.row_a,
        row_b=args.row_b,
        prefix_row=args.prefix_row,
        alphas=alphas,
    )
    model_kwargs = _to_device(model_kwargs)
    if use_cfg:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

    motion_shape = (args.batch_size, model.njoints, model.nfeats, args.pred_len)
    all_motions = []
    all_lengths = []
    all_text = []
    all_prefix_sources = []
    all_alphas = []

    with torch.no_grad():
        for rep_i in range(args.num_repetitions):
            print(f"### Latent interpolation sampling [repetition #{rep_i}]")
            noise = None
            if args.fixed_noise:
                noise = torch.randn((1,) + motion_shape[1:], device=dist_util.dev()).repeat(args.batch_size, 1, 1, 1)
            sample = diffusion.p_sample_loop(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=noise,
                const_noise=args.fixed_noise,
            )
            sample_vec = sample.detach().cpu().squeeze(2).permute(0, 2, 1).numpy()
            sample_vec = dataset.inv_transform(sample_vec)
            all_motions.append(sample_vec)
            all_lengths.append(model_kwargs["y"]["lengths"].cpu().numpy())
            all_text.extend([f"alpha_{alpha:g}" for alpha in alphas])
            all_prefix_sources.extend(model_kwargs["y"]["db_key"])
            all_alphas.extend(alphas)

    motion = np.concatenate(all_motions, axis=0).astype(np.float32)
    lengths = np.concatenate(all_lengths, axis=0).astype(np.int64)
    results = _latent_results_dict(
        dataset,
        motion,
        lengths,
        all_text,
        args,
        all_prefix_sources,
        extra={
            "row_a": int(args.row_a),
            "row_b": int(args.row_b),
            "prefix_row": int(args.prefix_row),
            "alphas": np.asarray(all_alphas, dtype=np.float32),
            "alpha_values": np.asarray(alphas, dtype=np.float32),
            "latent_interpolation_space": "normalized_linear",
            "fixed_noise": bool(args.fixed_noise),
        },
    )
    results_path = output_dir / "results.npy"
    np.save(results_path, results)
    print(f"Saved latent interpolation results to [{results_path}]")

    reference_path = None
    if not getattr(args, "skip_endpoint_references", False):
        reference_path = save_endpoint_references(dataset, args, output_dir)
        print(f"Saved endpoint references to [{reference_path}]")

    if getattr(args, "export_viser", False):
        paths = export_viser_results(results_path, output_dir / "viser_npz")
        print(f"Exported {len(paths)} interpolation Viser files to [{output_dir / 'viser_npz'}]")
        if reference_path is not None:
            ref_paths = export_viser_results(reference_path, output_dir / "viser_npz_references")
            print(f"Exported {len(ref_paths)} reference Viser files to [{output_dir / 'viser_npz_references'}]")

    return str(output_dir)


if __name__ == "__main__":
    generate_latent_interpolation()
