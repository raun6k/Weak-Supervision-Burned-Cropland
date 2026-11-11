"""Sampling, classifier hyperparameters, and evaluation gates."""

# ── CV & sampling (Phases 2-3) ───────────────────────────────────────
SEEDS       = [13, 29, 71]
K_FOLDS     = 5
N_PER_CLASS = 3000          # up from 400 in iteration 1
INNER_K_FOLDS = 3
INNER_VAL_POINTS_PER_CLASS = 500   # up from 150

# Threshold sweep
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CHOOSE_MODE = "KAPPA"  # "F1" | "KAPPA" | "ACC"

# Random Forest (GEE-native)
RF_N_TREES      = 300
RF_BAG_FRAC     = 0.7
RF_MIN_LEAF_POP = 1
RF_VARS_PER_SPLIT = None  # None = sqrt(n_features)

# Gradient Boosted Trees (GEE-native)
GBT_N_TREES     = 300
GBT_SHRINKAGE   = 0.1
GBT_MAX_NODES   = 64

# ── Feature importance ───────────────────────────────────────────────
TOPK_IMPORTANCE = 15

# ── Baselines (Phase 3) ─────────────────────────────────────────────
FIXED_DNBR_THRESHOLD = 0.12   # universal dNBR baseline (no recipe tuning)

# ── Phase 3 gate thresholds ─────────────────────────────────────────
# CV metrics table must have 15 rows (5 folds × 3 seeds) per recipe × feature set
GATE_CV_ROWS_PER_CONFIG = 15

# ── Phase 4: Python classifiers ─────────────────────────────────────
# Logistic Regression
LR_C = 1.0
LR_MAX_ITER = 1000

# XGBoost
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_COLSAMPLE_BYTREE = 0.7

# LightGBM
LGBM_N_ESTIMATORS = 300
LGBM_NUM_LEAVES = 31
LGBM_LEARNING_RATE = 0.1

# MLP (PyTorch)
MLP_HIDDEN = [128, 64]
MLP_DROPOUT = 0.3
MLP_LR = 0.001
MLP_EPOCHS = 200
MLP_BATCH_SIZE = 256

# MLP-SCE (Symmetric Cross-Entropy)
SCE_ALPHA = 1.0
SCE_BETA = 1.0

# ── Phase 4 gate thresholds ─────────────────────────────────────────
GATE_BEST_KAPPA_STRICT  = 0.35
GATE_BEST_AUC           = 0.75
GATE_BEST_COHENS_D      = 0.80
GATE_WILCOXON_P         = 0.05
GATE_LR_RF_GAP          = 0.03
GATE_SWIR_ABLATION_DROP = 0.03

# Phase 5 gate thresholds
GATE_MODIS_DISTRICT_R2      = 0.50
GATE_BURNED_AREA_RATIO      = 2.0   # our area / MODIS area <= this
GATE_SPEARMAN_CAQM          = 0.60

# ── Phase 6: Generalizability ───────────────────────────────────────
GATE_MULTIYEAR_KAPPA_STD    = 0.05
GATE_HARYANA_ZEROSHOT_KAPPA = 0.15
GATE_S1_IMPROVEMENT         = 0.03
