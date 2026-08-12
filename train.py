import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import time
from data import reset_rngs, get_batch
from model import Transformer, quantize_weights, loss_fn
import config
import wandb
import argparse
from pathlib import Path
import json
import os
from functools import partial
from eval_inference import evaluate


config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def count_params(module: nn.Module):
    return sum(p.size for _, p in tree_flatten(module.parameters()))


def make_schedule(peak: float, warmup_steps: int, max_steps: int):
    """
    dynamic lr scheduler, mlx feature, cleaner
    """

    warm = optim.linear_schedule(init=0, end=peak, steps=warmup_steps)
    cos = optim.cosine_decay(init=peak, decay_steps= max_steps - warmup_steps, end=0.1 * peak)
    return optim.join_schedules([warm, cos], [warmup_steps])


def save_model(model: Transformer, model_config: dict, run_dir: Path, arch: str):
    """
    saves model at the end of training. no resume capability.
    """
    weights = tree_flatten(model.parameters(), destination={})
    total_n_of_params = count_params(model) / 1e6
    save_path = run_dir / f"final_{arch}_{total_n_of_params:.0f}M_muon_m{model_config["MUON_LR"]:.1e}_a{model_config["ADAMW_LR"]:.1e}.safetensors"
    tmp_path = str(save_path) + ".tmp.safetensors"
    mx.save_safetensors(tmp_path, weights, metadata={"config": json.dumps(model_config)})
    os.replace(tmp_path, save_path)
    print(f"SAVE: Model saved to {save_path}")

def build_optimizers(muon_lr: float, adamw_lr: float, warmup_steps: int, max_steps: int):
    muon = optim.Muon(
        learning_rate=make_schedule(peak=muon_lr, warmup_steps=warmup_steps, max_steps=max_steps),
        momentum=0.95,
        weight_decay=0.1,
        nesterov=True,
        ns_steps=5
    )

    adamw_decay = optim.AdamW(
        learning_rate=make_schedule(peak=adamw_lr, warmup_steps=warmup_steps, max_steps=max_steps),
        betas=[0.90, 0.95],
        weight_decay=0.1,
        bias_correction=True
    )

    adamw_nodecay = optim.AdamW(
        learning_rate=make_schedule(peak=adamw_lr, warmup_steps=warmup_steps, max_steps=max_steps),
        betas=[0.90, 0.95],
        weight_decay=0.0,
        bias_correction=True
    )

    # this works like a router with predicates, read mlx docs
    # adamw gets embeddings, muon gets everything else >= 2-dims, adamw with no decay gets 1D stuff
    optimizers = optim.MultiOptimizer(
        [adamw_decay, muon, adamw_nodecay],
        [lambda p, w: p.startswith("embeddings"), lambda p, w: w.ndim >= 2])

    return optimizers

def train(K: int, D: int, H: int, token_budget: int, muon_lr: float, adamw_lr: float, seed: int, is_ternary: bool = False):

    mx.random.seed(seed)
    
    lm = Transformer(K=K, D=D, H=H, V=config.VOCAB_SIZE, ternary=is_ternary)
    mx.eval(lm.parameters())
    model_config = {"K":K ,"D":D, "H":H, "V": config.VOCAB_SIZE, "MUON_LR": muon_lr, "ADAMW_LR": adamw_lr, "seed":seed,  "IS_TERNARY": is_ternary}
    reset_rngs("both", seed)

    max_steps = token_budget // (config.BATCH_SIZE * config.SEQ_LEN)
    warmup_steps = max(1, int(0.01 * max_steps))

    optimizers = build_optimizers(muon_lr=muon_lr, adamw_lr=adamw_lr, warmup_steps=warmup_steps, max_steps=max_steps)
    optimizers.init(lm.trainable_parameters())
    mx.eval(optimizers.state)

    n_of_params = count_params(lm)
    n_of_params_in_millions = n_of_params / 1e6
    architecture_type = "fp" if not is_ternary else "ternary"

    tokens_trained = 0
    lm.train()

    # init wandb run
    run = wandb.init(project="avenue-mlx",
                        name = (f"{architecture_type}_{n_of_params_in_millions:.0f}M_muon"
                                    f"_m{muon_lr:.1e}_a{adamw_lr:.1e}_s{seed}"),
                        config={
                            "muon_lr": muon_lr,
                            "adamw_lr": adamw_lr,
                            "min_lr_frac": 0.1,
                            "optimizer": "muon+adamw",
                            "warmup_steps": warmup_steps,
                            "max_steps": max_steps,
                            "batch_size": config.BATCH_SIZE,
                            "seq_len": config.SEQ_LEN,
                            "K": K, "D": D, "H": H,
                            "vocab_size": config.VOCAB_SIZE,
                            "seed": seed,
                            "weight_decay": 0.1,
                            "grad_clip": 1.0,
                            "n_params": n_of_params_in_millions
                        },)

    RUN_DIR = config.CHECKPOINT_DIR / f"run_{run.id}_{run.name}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("TRAIN.PY - IN PROGRESS: Starting model training loop")
    print(f"TRAIN.PY: Will train for {max_steps} steps.")
    print(f"TRAIN.PY - Model config: {model_config}")
    print(f"TRAIN.PY - Number of parameters: {n_of_params}")

    # prepare for flip rate calculation
    if is_ternary:
        previous_weight = None
        current_weight = None

    loss_and_grad = nn.value_and_grad(lm, loss_fn)


    states = [lm.state, optimizers.state, mx.random.state]
    @partial(mx.compile, inputs=states, outputs=states)
    def step_fn(x, y):
        loss, grads = loss_and_grad(lm, x, y)
        grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
        optimizers.update(lm, grads)
        return loss, grad_norm

    start = time.perf_counter()

    for step in range(0, max_steps):
        x_b, y_b = get_batch("train")
        loss, grad_norm = step_fn(x_b, y_b)
        mx.eval(states, loss, grad_norm)
        tokens_trained += y_b.size

        if step % 50 == 0:
            metrics = {"train_loss": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "adamw_lr": optimizers.optimizers[0].learning_rate.item(),
                        "muon_lr": optimizers.optimizers[1].learning_rate.item(),
                        "tokens_trained": tokens_trained,}
            
            ## measure flip rate if ternary
            if is_ternary:
                layer = lm.main.layers[0].Q_layer
                current_weight = mx.sign(quantize_weights(layer.weight))
                mx.eval(current_weight)
                if previous_weight is not None:
                    metrics["flip_rate_50"] = (current_weight != previous_weight).astype(mx.float32).mean().item()
                    metrics["frac_zero"] = (current_weight == 0).astype(mx.float32).mean().item()
                    metrics["latent_absmax"] = layer.weight.abs().max().item()
                    metrics["latent_absmean"] = layer.weight.abs().mean().item()
                    print(
                        f"Step: {step}, train loss: {loss.item():.3f} time elapsed: {time.perf_counter() - start:.2f}s,",
                        f"tokens_trained: {tokens_trained}, tok/s:{tokens_trained /(time.perf_counter() - start):.2f}, grad_norm: {grad_norm.item():.2f},",
                        f"MUON LR: {optimizers.optimizers[1].learning_rate.item():.6f}, ADAMW LR: {optimizers.optimizers[0].learning_rate.item():.6f}",
                        f"flip_rate_50: {(current_weight != previous_weight).astype(mx.float32).mean().item()}, frac_zero: {(current_weight == 0).astype(mx.float32).mean().item()},",
                        f"latent_absmax: {layer.weight.abs().max().item()}, latent_absmean: {layer.weight.abs().mean().item()}"
                        )
                previous_weight = current_weight
            else:
                print(
                    f"TRAINING: Step: {step}, train loss: {loss.item():.3f} time elapsed: {time.perf_counter() - start:.2f}s,",
                    f"tokens_trained: {tokens_trained}, tok/s:{tokens_trained /(time.perf_counter() - start):.2f}, grad_norm: {grad_norm.item():.2f},",
                    f"MUON LR: {optimizers.optimizers[1].learning_rate.item():.6f}, ADAMW LR: {optimizers.optimizers[0].learning_rate.item():.6f}"
                    )
            wandb.log(metrics, step=step)

        # eval
        if step % 300 == 0:
            val_loss, val_bpb = evaluate(lm, loss_fn, 10)

            # log to wandb
            wandb.log({
                        "val_loss": val_loss,
                        "val_bpb": val_bpb},
                        step=step)
            print(f"EVAL: Step: {step}, train loss: {loss.item():.3f}, val_loss: {val_loss:.3f}")

    # final eval
    final_val_loss, final_val_bpb = evaluate(lm, loss_fn, 50)
    wandb.log({"final_val_loss": final_val_loss, "final_val_bpb": final_val_bpb}, step=step)

    save_model(lm, model_config, RUN_DIR, architecture_type)

    # finish wandb logging
    run.finish()
    
    # clear memory
    del lm, optimizers
    mx.clear_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--muon-lr", type=float, required=True)
    parser.add_argument("--adamw-lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ternary", action="store_true", help="To train in ternary weights or not")
    args = parser.parse_args()
    train(K=args.K, D=args.D, H=args.H, token_budget=args.token_budget, muon_lr=args.muon_lr, adamw_lr=args.adamw_lr, seed=args.seed, is_ternary=args.ternary)