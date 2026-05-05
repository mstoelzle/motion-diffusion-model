import numpy as np

from visualize.export_lafan_g1_to_viser import export_all, export_single, load_results


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
