import mlx.core as mx
from data import reset_rngs, get_batch
import model
import config
import json
import math

with open(config.SPLIT_MANIFEST_PATH) as f:
    val_bpt = json.load(f)["val_bpt"]


def evaluate(lm: model.Transformer, loss_fn, eval_size: int):
    reset_rngs(split="val")
    was_training = lm.training

    lm.eval()

    val_losses = []
    for _ in range(eval_size):
        x_val_b, y_val_b = get_batch("val")
        val_loss = loss_fn(lm, x_val_b, y_val_b)
        val_losses.append(val_loss.item())
    
    val_losses_mean = sum(val_losses) / len(val_losses)

    ## bpb calculation
    val_bpb = val_losses_mean / (math.log(2) * val_bpt)

    if was_training:
        lm.train()

    return (val_losses_mean, val_bpb)