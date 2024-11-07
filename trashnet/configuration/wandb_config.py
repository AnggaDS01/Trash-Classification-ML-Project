from trashnet.constant.training_pipeline import *

WANDB_CONFIG = {
    "project": WANDB_PROJECT_NAME,
    "config": {
        "learning_rate": LEARNING_RATE,
        "loss_function": LOSS_FUNCTION,
        "metrics": METRICS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "architecture": MODEL_NAME,
        "dataset": DATASET,
    }
}