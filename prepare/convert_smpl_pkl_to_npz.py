"""Convert legacy chumpy-based SMPL pickle files to plain NumPy archives."""

from __future__ import annotations

import argparse
import inspect
import pickle
from pathlib import Path

import numpy as np


def _patch_legacy_chumpy_runtime() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    for name, typ in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, typ)


def convert_smpl_pkl_to_npz(input_path: Path, output_path: Path) -> None:
    _patch_legacy_chumpy_runtime()

    with input_path.open("rb") as input_file:
        data = pickle.load(input_file, encoding="latin1")

    arrays = {
        key: np.asarray(data[key])
        for key in ["J", "f", "kintree_table", "posedirs", "v_template", "weights"]
    }
    arrays["shapedirs"] = np.asarray(data["shapedirs"].r)
    arrays["J_regressor"] = np.asarray(data["J_regressor"].todense())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    convert_smpl_pkl_to_npz(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
