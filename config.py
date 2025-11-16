# config.py
from pathlib import Path

class Config:
    # ------------------- data -------------------
    DATA_ROOT = Path("./data")
    IMG_SIZE = 28
    NUM_CHANNELS = 1
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # ------------------- model ------------------
    PATCH_SIZE = 7                              # 28 / 7 = 4 → 16 patches
    NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 16
    EMBEDDING_DIMS = 64
    ATTENTION_HEADS = 4
    TRANSFORMER_BLOCKS = 4
    MLP_HIDDEN = 128
    NUM_CLASSES = 10

    # ------------------- training ---------------
    LEARNING_RATE = 1e-3
    EPOCHS = 5
    SEED = 42
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # ------------------- paths ------------------
    CKPT_DIR = Path("./checkpoints")
    CKPT_DIR.mkdir(exist_ok=True)
    BEST_CKPT = CKPT_DIR / "vit_mnist_best.pth"
    LAST_CKPT = CKPT_DIR / "vit_mnist_last.pth"
    HISTORY_FILE = CKPT_DIR / "history.json"