<div align="center">

# Calibrated Weak Supervision for Post-Harvest<br>Burned-Cropland Mapping Under Label Scarcity

**Mapping crop-residue burning with scarce labels through optical change, fire context, and external calibration.**

[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](papers/burned_area_mapping_paper.pdf)
![Code](https://img.shields.io/badge/Code-Python-3776AB)
![Reproducibility](https://img.shields.io/badge/Reproduction%20scripts-Available-2ea44f)

**Raunak Bhagate · Maitri Polisetty · R I Minu**

SRM Institute of Science & Technology

[Paper (PDF)](papers/burned_area_mapping_paper.pdf) · [Contact](mailto:rb7314@srmist.edu.in) · [Overview](#overview) · [Setup](#setup) · [Data](#data) · [Reproduce](#reproduce) · [Citation](#citation)

</div>

---

## Overview

**Mapping small agricultural burns is challenging when reliable labels are scarce.** We study post-harvest burned-cropland mapping in Punjab, India, using Sentinel-2 spectral change and VIIRS active-fire context to construct weak labels.

**The framework connects weak supervision, spatial evaluation, and external calibration.** We compare pseudo-label recipes, feature representations, and classifier families using district-held-out evaluation, then calibrate a Google Earth Engine Random Forest map against MODIS MCD64A1 and district fire-count concordance.

The study supports **district-scale burden assessment and hotspot screening**. Internal scores measure agreement with weak labels; exact scar boundaries and stable year-to-year performance remain unestablished.

![Workflow from satellite inputs and weak-label construction through district-held-out learning, statewide inference, external calibration, and robustness assessment](assets/fig2_workflow.png)

**[Read the paper for the full method, results, and limitations →](papers/burned_area_mapping_paper.pdf)**

## Highlights

- **Weak supervision under label scarcity.** Three label recipes combine fire proximity and optical change, with direct NBR and dNBR predictors excluded from model inputs.
- **Spatial evaluation and map calibration.** District-held-out model comparisons and external deployment-threshold selection are treated as distinct stages.
- **Transfer and robustness.** Experiments examine neighboring-state transfer to Haryana, cross-year performance in Punjab, and Sentinel-1/Sentinel-2 feature fusion.

## Setup

Python 3.10 or newer. From this directory:

```bash
pip install -e .
```

`pip install -r requirements.txt` installs the same pins without the editable package. Earth Engine stages also need a Google Earth Engine project:

```bash
earthengine authenticate
```

Set `PROJECT` and `DRIVE_FOLDER` in [burnmap/config/datasets.py](burnmap/config/datasets.py). Both are empty in git.

Configuration is split under `burnmap/config/`:

- `datasets.py` — collection IDs, date windows, coverage gates
- `recipes.py` — `strict`, `moderate`, and `very_strict` weak-label rules
- `models.py` — sampling, classifier hyperparameters, evaluation gates
- `published.py` — the reported map: `very_strict` / `rf` / feature set A, threshold 0.60

## Data

Satellite inputs are public Earth Engine collections. The export scripts do not read a local satellite archive.

| Collection | Config name |
|---|---|
| `FAO/GAUL/2015/level2` | `GAUL_LVL2` |
| `ESA/WorldCover/v200` | `WORLDCOVER_V200` |
| `COPERNICUS/S2_SR_HARMONIZED` | `S2_SR` |
| `COPERNICUS/S2_CLOUD_PROBABILITY` | `S2_CLOUD_PROB` |
| `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | `CLOUD_SCORE_PLUS` |
| `COPERNICUS/S1_GRD` | `S1_GRD` |
| `NASA/LANCE/SNPP_VIIRS/C2` | `VIIRS_SNPP` |
| `NASA/LANCE/NOAA20_VIIRS/C2` | `VIIRS_NOAA20` |
| `MODIS/061/MCD64A1` | `MODIS_MCD64A1` |

Local files that are already in the repository:

- `data/samples/` — the sample tables the paper's learned comparisons use. Punjab 2024 has one file per recipe and seed (`punjab_2024_<recipe>_A_s<seed>.csv`) plus the dNBR baseline tables. Punjab 2023, Haryana 2024, and Sentinel-1 feature set E are included for the cross-year, zero-shot, and fusion results. Sets B–D are column subsets of the Punjab 2024 tables. A district-held-out MODIS classifier baseline is not part of the reported experiment. A fresh Earth Engine export may still use older Drive filenames; the loaders accept those names too. Re-exported samples will not be bit-identical, because stratified sampling is stochastic even with the fixed seeds.
- `data/external/fire_counts.csv` — district fire counts (`district`, `fire_count`) used for the Spearman concordance in the threshold sweep. The pipeline treats this table as the CAQM/IARI district count file.
- `data/fixtures/fig1_layers/` — boundaries, cropland, MODIS, and VIIRS density used by the study-area figure.

Generated metrics, figures, and map GeoTIFFs go to `outputs/`, which is gitignored. The three statewide probability GeoTIFFs are about 540 MB each, so they are produced by `scripts/export_maps.py` and downloaded from Drive rather than stored here. Map calibration and the final map figure need those GeoTIFFs under `outputs/<year>/phase5_maps/`.

## Reproduce

Run commands from the repository root.

Earth Engine exports, after `PROJECT` and `DRIVE_FOLDER` are set:

```bash
python scripts/assemble_data.py --year 2024
python scripts/check_labels.py --year 2024
python scripts/export_samples.py --year 2024
python scripts/export_transfer_samples.py --year 2024
python scripts/export_maps.py --year 2024
```

`export_transfer_samples.py` exports Punjab 2023, zero-shot Haryana 2024, and Sentinel-1. The statewide map is trained on every labeled Punjab cropland pixel. Download new sample CSVs into `data/samples/` and map GeoTIFFs into `outputs/2024/phase5_maps/`.

Local stages use the committed samples and do not import Earth Engine:

```bash
python scripts/train_eval.py --year 2024
python scripts/transfer.py --year 2024
python scripts/paper_tables.py --year 2024
```

`train_eval.py` fits random forest, gradient boosting, logistic regression, XGBoost, LightGBM, and the two MLPs, then merges the metric tables. The MLP stage is the slow one.

After the probability maps are on disk:

```bash
python scripts/calibrate.py --year 2024
python scripts/make_figures.py
```

`calibrate.py` reads `data/external/fire_counts.csv` unless `--caqm_csv` points elsewhere. `make_figures.py` writes images under `outputs/figures/`. The study-area and weak-label panels call Earth Engine; the study-area script also reads `data/fixtures/fig1_layers/`.

Single-stage flags that the chain scripts do not expose are available as modules, for example `python -m burnmap.modeling.cross_validation --classifiers rf`.

The numbered lab pipeline maps onto these commands as follows: phase 1 `assemble_data.py`, phase 2 `check_labels.py`, phase 3 export `export_samples.py`, phase 3 CV plus phase 4 `train_eval.py`, phase 5 export `export_maps.py`, phase 5 validation and calibration `calibrate.py`, phase 6 export `export_transfer_samples.py`, phase 6 CV `transfer.py`, phase 7 `paper_tables.py`.

## Release plan

| Item | Status |
|---|---|
| Paper (PDF) | ✅ [Available](papers/burned_area_mapping_paper.pdf) |
| Experiment code | ✅ `burnmap/` |
| Reproduction scripts | ✅ `scripts/` |
| Setup and data-access instructions | ✅ [Setup](#setup) and [Data](#data) |

## Citation

If you use this work, please cite the [paper](papers/burned_area_mapping_paper.pdf). Citation metadata is provided in [CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{bhagate2026calibrated,
  title     = {Calibrated Weak Supervision for Post-Harvest Burned-Cropland Mapping Under Label Scarcity},
  author    = {Bhagate, Raunak and Polisetty, Maitri and Minu, R I},
  booktitle = {Proceedings of the 47th Asian Conference on Remote Sensing (ACRS 2026)},
  year      = {2026}
}
```

## Contact

[Raunak Bhagate](mailto:rb7314@srmist.edu.in) · SRM Institute of Science & Technology
