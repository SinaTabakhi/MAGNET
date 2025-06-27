from yacs.config import CfgNode

# ---------------------------------------------------------
# Config definition
# ---------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
_C.DATASET = CfgNode()
_C.DATASET.ROOT = "data"
_C.DATASET.SPLITS = "split_data"
_C.DATASET.PROCESSED = "processed_data"
_C.DATASET.NAME = "BRCA"  # Options: "BLCA", "BRCA", "OV"
_C.DATASET.NUM_CLASSES = 5
_C.DATASET.CLASS_NAMES = ["Normal-like", "Luminal A", "Luminal B", "Basal-like", "HER2-enriched"]
_C.DATASET.SEEDS = [10, 20, 30, 40, 50]
_C.DATASET.OMICS = ["DNA", "mRNA", "miRNA"]
_C.DATASET.SPLIT_RATIO = [0.7, 0.1, 0.2] # Training, validation, test set ration
_C.DATASET.NUM_FEATURES = 1000 # Number of features to select using the feature selection method
_C.DATASET.SPARSITY_RATES = 0.80  # For constructing the patient similarity network

# ---------------------------------------------------------
# MLP encoder
# ---------------------------------------------------------
_C.ENCODER = CfgNode()
_C.ENCODER.HID_DIMS = 128
_C.ENCODER.DROPOUT_RATE = 0.0

# ---------------------------------------------------------
# Multi-head attention layer
# ---------------------------------------------------------
_C.ATTENTION = CfgNode()
_C.ATTENTION.NUM_HEADS = 4

# ---------------------------------------------------------
# GNN decoder
# ---------------------------------------------------------
_C.DECODER = CfgNode()
_C.DECODER.HID_DIMS = 128
_C.DECODER.NUM_LAYERS = 2
_C.DECODER.DROPOUT_RATE = 0.0
_C.DECODER.NEGATIVE_SLOPE = 0.2

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
_C.SOLVER = CfgNode()
_C.SOLVER.MAX_EPOCHS = 200
_C.SOLVER.LR = 1e-3  # Learning rate
_C.SOLVER.WD = 1e-3  # Weight decay

# ---------------------------------------------------------
# Comet logger setup
# ---------------------------------------------------------
_C.COMET = CfgNode()
_C.COMET.PROJECT_NAME = "magnet"
_C.COMET.WORKSPACE = "YOUR_WORKSPACE"
_C.COMET.LOG_DIR = "comet_logs"
_C.COMET.USE = False

# ---------------------------------------------------------
# CSV logger setup
# ---------------------------------------------------------
_C.CSV = CfgNode()
_C.CSV.LOG_DIR = "csv_logs"
_C.CSV.USE = True

# ---------------------------------------------------------
# Ray tune setup (hyperparameter optimization)
# ---------------------------------------------------------
_C.TUNE = CfgNode()
_C.TUNE.MIN_EPOCHS = 10
_C.TUNE.MAX_EPOCHS = 100
_C.TUNE.NUM_SAMPLES = 100  # Number of hyperparameter combinations to sample
_C.TUNE.METRIC = "val_acc" # Metric to optimize (options: "val_acc", "val_auroc", "val_auprc", "val_f1_macro")
_C.TUNE.LOG_DIR = "tune_logs"
_C.TUNE.USE = True

# ---------------------------------------------------------
# UMAP setup
# ---------------------------------------------------------
_C.UMAP = CfgNode()
_C.UMAP.N_NEIGHBORS = 15
_C.UMAP.MIN_DIST = 0.1

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
_C.RESULT = CfgNode()
_C.RESULT.OUTPUT_DIR = "output"
# setup model directory
_C.RESULT.SAVE_MODEL_DIR = "models"
_C.RESULT.SAVE_MODEL_TMPL = "{dataset_name}_model_{seed}.ckpt"
_C.RESULT.SAVE_MODEL = True
# setup output csv files
_C.RESULT.FILE_HEADER_CLF = ['metric', 'seed', 'value']
_C.RESULT.FILE_HEADER_TIME = ['seed', 'value']
_C.RESULT.SAVE_RESULT = True
# setup PyTorch Lightning log
_C.RESULT.LIGHTNING_LOG_DIR = "lightning_logs"
# setup plot output
_C.RESULT.SAVE_PLOT_DIR = "plots"
_C.RESULT.SAVE_PLOT_TMPL = "{mode}_{name}_{title}_[{seed}].{format}"
# setup cluster metric output
_C.RESULT.SAVE_CLUSTER_DIR = "cluster_metrics"
_C.RESULT.SAVE_CLUSTER_TMPL = "{mode}_cluster_metrics_[{seed}].csv"

def get_cfg_defaults():
    return _C.clone()
