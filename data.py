import numpy as np
import config
import mlx.core as mx

val_bin_arr = np.memmap(config.VAL_BIN, dtype=config.TOKEN_DTYPE, mode="r")
train_bin_arr = np.memmap(config.TRAIN_BIN, dtype=config.TOKEN_DTYPE, mode="r")

def reset_rngs(split: str = "both", seed: int = config.SEED):
    global train_rng, val_rng
    if split == "train":
        train_rng = np.random.default_rng(seed)
    elif split == "both":
        train_rng = np.random.default_rng(seed)
        val_rng = np.random.default_rng(seed + 100)
    else:
        val_rng = np.random.default_rng(seed + 100)

def get_batch(split: str):
    """
    Creates a random batch of SEQ_LEN + 1.
    """
    rng = train_rng if split == "train" else val_rng
    bin_arr = val_bin_arr if split == "val" else train_bin_arr

    starts = rng.integers(low=0, high=len(bin_arr) - config.SEQ_LEN, size=config.BATCH_SIZE)

    batch = [bin_arr[start: start + config.SEQ_LEN + 1] for start in starts]
    batch = np.stack(batch).astype(np.int32)  ### token dtype is uint16 anyways

    x_b = mx.array(batch[:, :-1])
    y_b = mx.array(batch[:, 1:])

    return (x_b, y_b)
