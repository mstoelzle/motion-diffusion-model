import argparse
import re
from pathlib import Path

import numpy as np


def load_results(results_path):
    data = np.load(results_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError(f"Expected a .npy results file, got npz: {results_path}")
    results = data.item() if data.shape == () else data[None][0]
    motion = np.asarray(results["motion"])
    if motion.ndim != 3 or motion.shape[-1] != 36:
        raise ValueError(
            "Expected lafan_g1 generated motion with shape "
            f"(num_outputs, frames, 36), got {motion.shape}"
        )
    motion_format = results.get("motion_format", "")
    if motion_format and motion_format != "g1_qpos":
        raise ValueError(f"Expected motion_format='g1_qpos', got {motion_format!r}")
    return results


def sample_flat_index(results, sample_idx, rep_idx):
    num_samples = int(results["num_samples"])
    num_repetitions = int(results["num_repetitions"])
    if not 0 <= sample_idx < num_samples:
        raise IndexError(f"sample_idx {sample_idx} is outside [0, {num_samples})")
    if not 0 <= rep_idx < num_repetitions:
        raise IndexError(f"rep_idx {rep_idx} is outside [0, {num_repetitions})")
    return rep_idx * num_samples + sample_idx


def safe_label(label):
    label = str(label) if label is not None else "sample"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip())
    return label.strip("_") or "sample"


def write_viser_npz(output_path, qpos, fps, text=None, length=None, joint_names=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "qpos": np.asarray(qpos, dtype=np.float32),
        "fps": np.array(int(round(float(fps))), dtype=np.int64),
    }
    if text is not None:
        payload["text"] = np.array(str(text))
    if length is not None:
        payload["length"] = np.array(int(length), dtype=np.int64)
    if joint_names is not None:
        payload["joint_names"] = np.asarray(joint_names)
    np.savez(output_path, **payload)
    return output_path


def export_single(results, output_path, sample_idx=0, rep_idx=0):
    flat_idx = sample_flat_index(results, sample_idx, rep_idx)
    motion = np.asarray(results["motion"])
    lengths = np.asarray(results.get("lengths", [motion.shape[1]] * len(motion)))
    text = results.get("text", ["sample"] * len(motion))
    fps = results.get("fps", 50)
    joint_names = results.get("joint_names", None)

    length = int(lengths[flat_idx])
    qpos = motion[flat_idx, :length]
    return write_viser_npz(
        output_path,
        qpos=qpos,
        fps=fps,
        text=text[flat_idx] if len(text) > flat_idx else None,
        length=length,
        joint_names=joint_names,
    )


def export_all(results, output_dir):
    motion = np.asarray(results["motion"])
    num_samples = int(results["num_samples"])
    num_repetitions = int(results["num_repetitions"])
    text = results.get("text", ["sample"] * len(motion))
    paths = []
    for rep_idx in range(num_repetitions):
        for sample_idx in range(num_samples):
            flat_idx = sample_flat_index(results, sample_idx, rep_idx)
            label = safe_label(text[flat_idx] if len(text) > flat_idx else "sample")
            output_path = (
                Path(output_dir)
                / f"sample{sample_idx:02d}_rep{rep_idx:02d}_{label}.npz"
            )
            paths.append(export_single(results, output_path, sample_idx, rep_idx))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert lafan_g1 sample.generate results.npy files into qpos .npz "
            "files accepted by HoloSoma's viser_player.py."
        )
    )
    parser.add_argument("--results_path", required=True, type=Path)
    parser.add_argument("--output_path", default="", type=Path)
    parser.add_argument("--output_dir", default="", type=Path)
    parser.add_argument("--sample_idx", default=0, type=int)
    parser.add_argument("--rep_idx", default=0, type=int)
    parser.add_argument("--all", action="store_true", help="Export every sample/repetition.")
    args = parser.parse_args()

    results = load_results(args.results_path)
    if args.all:
        output_dir = args.output_dir
        if not str(output_dir):
            output_dir = args.results_path.parent / "viser_npz"
        paths = export_all(results, output_dir)
        print(f"Exported {len(paths)} files to {Path(output_dir).resolve()}")
        return

    output_path = args.output_path
    if not str(output_path):
        output_path = args.results_path.parent / (
            f"viser_sample{args.sample_idx:02d}_rep{args.rep_idx:02d}.npz"
        )
    path = export_single(results, output_path, args.sample_idx, args.rep_idx)
    print(f"Exported {path.resolve()}")


if __name__ == "__main__":
    main()
