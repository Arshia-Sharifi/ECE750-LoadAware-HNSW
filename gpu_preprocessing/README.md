# GPU preprocessing (run on Google Colab, T4)

This folder is self-contained. Copy it to Colab together with `configs/datasets.yaml`. The spec is in `../IMPLEMENTATION_PLAN.md`, section 7.

## Output per dataset

The large files stay on the Colab disk (`--data-root`). The small files go to `--out`.

| Path | Content |
|---|---|
| `<data-root>/artifacts/<id>/train.npy`, `test.npy`, `holdout.npy`, `meta.json` | Raw float32 vectors and metadata. Local disk only. |
| `<out>/<id>/gt/neighbors.npy`, `gt/distances.npy`, `gt/gt_meta.json` | Exact top-100 neighbours of the test queries. `gt_source` is `provided` or `generated`. |
| `<out>/<id>/calibration/samplings.bin`, `calib_meta.json` | Calibration queries and exact ground truth, in Ada-ef's binary format. |
| `<out>/<id>/stats/estimator.bin` | Mean and covariance, in Ada-ef's Estimator format. |
| `<out>/<id>/manifest.json` | Fingerprints that `scripts/prepare_dataset.sh` checks on the serving machine. |

The datasets are NumPy `.npy` files, not HDF5 (plan section 4.3). h5py is used only to read the ann-benchmarks-format downloads, which are HDF5 upstream.

## Colab steps

1. Runtime, Change runtime type, T4 GPU.
2. Open `colab_preprocess.ipynb`, mount Drive, and run all cells. To run one dataset by hand:
   ```bash
   pip install uv && uv pip install --system -r requirements.txt   # Colab keeps its preinstalled torch and CUDA
   python run_preprocess.py --dataset glove-100-angular \
       --steps download,convert,gt,calib,stats,manifest \
       --data-root /content/data --out /content/drive/MyDrive/laef_artifacts \
       --registry ../configs/datasets.yaml --device cuda --verify-provided 500
   ```
3. Repeat for the other dataset ids in `configs/datasets.yaml`.
4. Copy each `laef_artifacts/<id>/` folder to the serving machine, then run `bash scripts/prepare_dataset.sh <id> <path-to-folder>`.

Every step is idempotent: a finished step is skipped unless `--force` is given. The `gt`, `calib`, `stats` and `manifest` steps need a CUDA GPU and refuse to run without one. On the serving machine only `download,convert` runs (see `scripts/prepare_dataset.sh`). The unit tests call the functions directly on the CPU: `python -m pytest tests -q` from this folder.

Limits: about 16 GB VRAM, about 12.7 GB RAM and about 100 GB disk. Everything streams in chunks, so the raw download is the only large per-dataset requirement.
