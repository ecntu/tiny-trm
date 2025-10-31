# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "einops",
#     "torch",
#     "datasets",
#     "ema-pytorch",
#     "tqdm",
#     "wandb",
#     "accelerate",
# ]
# ///

# A single-file implementation of the Tiny Recursive Model (TRM) trained on Sudoku.
# Paper: https://arxiv.org/abs/2305.10445
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange
from einops.layers.torch import Reduce

from accelerate import Accelerator
from ema_pytorch import EMA
from datasets import load_dataset

import os
import argparse
from functools import partial
from tqdm import tqdm


class TRM(nn.Module):
    def __init__(
        self,
        net,
        output_head,
        Q_head,
        input_embedding,
        init_y,
        init_z,
        halt_loss_weight=0.5,
    ):
        super().__init__()
        self.net = net
        self.output_head = output_head
        self.Q_head = Q_head
        self.input_embedding = input_embedding
        self.init_y = init_y
        self.init_z = init_z
        self.halt_loss_weight = halt_loss_weight

    def latent_recursion(self, x, y, z, n=6):
        for i in range(n):  # latent reasoning
            z = self.net(y, z, x)
        y = self.net(y, z)  # refine output answer
        return y, z

    def deep_recursion(self, x, y, z, n=6, T=3):
        # recursing T-1 times to improve y and z (no gradients needed)
        with torch.no_grad():
            for j in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)
        # recursing once to improve y and z
        y, z = self.latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), self.output_head(y), self.Q_head(y)

    @torch.no_grad()
    def predict(self, x_input, N_supervision=16, n=6, T=3):
        y_hats = []
        y, z = self.init_y, self.init_z
        x = self.input_embedding(x_input)
        for step in range(N_supervision):
            (y, z), y_hat, _ = self.deep_recursion(x, y, z, n=n, T=T)
            y_hats.append(y_hat)
        return rearrange(y_hats, "n b l c -> n b l c")

    def forward(self, x_input, y, z, y_true, n=6, T=3):
        x = self.input_embedding(x_input)
        (y, z), y_hat, q_hat = self.deep_recursion(x, y, z, n=n, T=T)

        rec_loss = F.cross_entropy(
            rearrange(y_hat, "b l c -> (b l) c"), rearrange(y_true, "b l -> (b l)")
        )
        halt_loss = F.binary_cross_entropy_with_logits(
            q_hat,
            (y_hat.argmax(dim=-1) == y_true).float().mean(dim=-1, keepdim=True),
        )
        loss = rec_loss + self.halt_loss_weight * halt_loss

        return (y, z), y_hat, q_hat, loss, (rec_loss, halt_loss)


class SwiGLU(nn.Module):
    """SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)"""

    def __init__(self, d_model, factor=4):
        super().__init__()
        d_ff = int(d_model * factor)
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


class MixerBlock(nn.Module):
    def __init__(self, seq_len, h_dim, factor=4):
        super().__init__()
        self.l_mixer = SwiGLU(seq_len, factor=factor)
        self.d_mixer = SwiGLU(h_dim, factor=factor)
        self.l_norm = nn.LayerNorm(h_dim)
        self.d_norm = nn.LayerNorm(h_dim)

    def forward(self, h):
        o = self.l_norm(h)
        o = rearrange(o, "b l d -> b d l")
        o = self.l_mixer(o)
        o = rearrange(o, "b d l -> b l d")

        h = o + h

        o = self.d_norm(h)
        o = self.d_mixer(o)
        return o + h


class Net(nn.Module):
    def __init__(self, seq_len, h_dim, n_layers=2, factor=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MixerBlock(seq_len=seq_len, h_dim=h_dim, factor=factor)
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(h_dim)

    def forward(self, y, z, x=None):
        h = (x + y + z) if x is not None else (y + z)
        for block in self.blocks:
            h = block(h)
        return self.out_norm(h)


def train_batch(
    accelerator,
    model,
    batch,
    opt,
    scheduler,
    logger,
    N_supervision,
    n,
    T,
    halt_prob_thresh=0.5,
    max_grad_norm=1.0,
):
    model.train()
    x_input, y_true = batch["inputs"], batch["labels"]
    z, y = model.init_z, model.init_y

    for step in range(N_supervision):
        with accelerator.autocast():
            (y, z), y_hat, q_hat, loss, (rec_loss, halt_loss) = model(
                x_input, y, z, y_true, n=n, T=T
            )

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        scheduler.step()

        halt_probs = q_hat.sigmoid()
        if halt_probs.gt(halt_prob_thresh).all():
            break

    logger(
        {
            "train/loss": loss.detach().item(),
            "train/rec_loss": rec_loss.detach().item(),
            "train/halt_loss": halt_loss.detach().item(),
            "train/halt_prob_mean": halt_probs.mean().item(),
            "train/halt_prob_std": halt_probs.std().item(),
            "train/batch_steps": step + 1,
            "train/lr": opt.param_groups[0]["lr"],
            "train/token_corr_y": float(token_corr(y)),
            "train/token_corr_z": float(token_corr(z)),
        }
    )


@torch.inference_mode()
def evaluate(accelerator, model, data_loader, N_supervision=16, n=6, T=3):
    model.eval()
    total_cells, correct = 0, 0
    total_puzzles, solved = 0, 0

    it = (
        tqdm(data_loader, desc="Evaluating")
        if accelerator.is_main_process
        else data_loader
    )

    for batch in it:
        x_input, y_true = batch["inputs"], batch["labels"]

        with accelerator.autocast():
            y_hats_logits = model.predict(
                x_input, N_supervision=N_supervision, n=n, T=T
            )

        pred_cells = y_hats_logits[-1].argmax(-1)  # use only the last sup step

        # gather across processes for global metrics
        pred_cells = accelerator.gather_for_metrics(pred_cells)
        y_true = accelerator.gather_for_metrics(y_true)

        correct += (pred_cells == y_true).sum().item()
        solved += (pred_cells == y_true).all(dim=-1).sum().item()
        total_cells += y_true.numel()
        total_puzzles += y_true.shape[0]

        if accelerator.is_main_process:
            it.set_postfix(
                {"sol": solved / total_puzzles, "acc": correct / total_cells}
            )
    return solved / total_puzzles, correct / total_cells


def token_corr(h, eps=1e-8):
    # To guard against representation collapse
    b, l, d = h.shape

    x = h - h.mean(dim=2, keepdim=True)
    x = x / x.var(dim=2, unbiased=False, keepdim=True).sqrt().clamp_min(eps)

    corr = x @ x.transpose(1, 2) / d  # (b, l, l)
    eye = torch.eye(l, device=h.device, dtype=h.dtype).unsqueeze(0)  # (1, l, l)
    offdiag_sum = (corr * (1 - eye)).sum(dim=(1, 2)) / (l * l - l)  # (b,)
    return offdiag_sum.mean()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--mlp_factor", type=int, default=4)

    parser.add_argument("--N_supervision", type=int, default=16)
    parser.add_argument("--N_supervision_eval", type=int, default=None)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--halt_loss_weight", type=float, default=0.5)
    parser.add_argument("--halt_prob_thresh", type=float, default=0.95)

    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--lr_warmup_iters", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--ema_beta", type=float, default=0.999)
    parser.add_argument("--epochs", type=int, default=60_000)
    parser.add_argument("--steps", type=int, default=None)

    parser.add_argument("--val_every", type=int, default=50)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--log_wandb", action="store_true")

    args = parser.parse_args()

    vocab_len, seq_len = 10, 81

    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using: {device}, {accelerator.mixed_precision}")

    lr = args.lr / (args.batch_size * args.N_supervision)
    accelerator.print(f"Effective learning rate: {lr}")

    model = TRM(
        net=Net(
            seq_len=seq_len,
            h_dim=args.h_dim,
            n_layers=args.n_layers,
            factor=args.mlp_factor,
        ),
        output_head=nn.Linear(args.h_dim, vocab_len),
        Q_head=nn.Sequential(Reduce("b l h -> b h", "mean"), nn.Linear(args.h_dim, 1)),
        input_embedding=nn.Embedding(vocab_len, args.h_dim),
        init_y=nn.Buffer(torch.randn(args.h_dim)),
        init_z=nn.Buffer(torch.randn(args.h_dim)),
        halt_loss_weight=args.halt_loss_weight,
    )

    model = torch.compile(model)

    # Note: checkpoints are not really meant for cont. training here, just for eval-only runs.
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path))
        accelerator.print(f"Loaded checkpoint from {args.checkpoint_path}")

    accelerator.print(model)
    accelerator.print("No. of parameters:", sum(p.numel() for p in model.parameters()))

    ds_path = "emiliocantuc/sudoku-extreme-1k-aug-1000"
    train_ds = load_dataset(ds_path, split="train")
    val_ds = load_dataset(ds_path, split="test[:1024]")
    test_ds = load_dataset(ds_path, split="test")
    for ds in (train_ds, val_ds, test_ds):
        ds.set_format(type="torch", columns=["inputs", "labels"])

    get_loader = partial(
        DataLoader,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=device.type == "cuda",
        num_workers=4 if (device.type == "cuda") else 0,
        persistent_workers=device.type == "cuda",
    )
    train_loader = get_loader(train_ds, shuffle=True)
    val_loader = get_loader(val_ds, shuffle=False)
    test_loader = get_loader(test_ds, shuffle=False)

    model, train_loader, val_loader, test_loader = accelerator.prepare(
        model, train_loader, val_loader, test_loader
    )

    ema = EMA(
        accelerator.unwrap_model(model),
        beta=args.ema_beta,
        forward_method_names=("predict",),
    ).to(device)

    if args.log_wandb:
        import wandb

        wandb.init(
            project="trm-sudoku",
            config=vars(args),
            settings=wandb.Settings(code_dir="."),
        )
        wandb.watch(model, log="all", log_freq=10)

    def logger(data, step=None):
        if not accelerator.is_main_process:
            return
        if "train/loss" in data:
            accelerator.print(
                f"step {step} | loss: {data['train/loss']:.4f} | batch steps: {data['train/batch_steps']}"
            )
        if args.log_wandb:
            wandb.log(data, step=step)

    if not args.eval_only:
        opt = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=args.lr_warmup_iters
        )

        opt, scheduler = accelerator.prepare(opt, scheduler)

        n_steps = args.steps or (args.epochs * len(train_loader))
        best_acc = 0.0
        for step, batch in enumerate(cycle(train_loader), start=1):
            train_batch(
                accelerator=accelerator,
                model=model,
                batch=batch,
                opt=opt,
                scheduler=scheduler,
                logger=partial(logger, step=step),
                N_supervision=args.N_supervision,
                n=args.n,
                T=args.T,
                halt_prob_thresh=args.halt_prob_thresh,
            )

            ema.update()

            if step >= n_steps:
                break

            if step % args.val_every == 0:
                solve_rate, cell_acc = evaluate(
                    accelerator,
                    ema,
                    val_loader,
                    N_supervision=args.N_supervision_eval or args.N_supervision,
                    n=args.n,
                    T=args.T,
                )
                logger(
                    {
                        "val/solve_rate": float(solve_rate),
                        "val/cell_accuracy": float(cell_acc),
                    },
                    step=step,
                )

                if (
                    accelerator.is_main_process
                    and args.checkpoint_path
                    and cell_acc > best_acc
                ):
                    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
                    accelerator.save(ema.ema_model.state_dict(), args.checkpoint_path)
                    accelerator.print(f"Checkpoint saved to {args.checkpoint_path}")
                    best_acc = cell_acc

        ema.copy_params_from_ema_to_model()

    solve_rate, cell_acc = evaluate(
        accelerator,
        model,
        test_loader,
        N_supervision=args.N_supervision_eval or args.N_supervision,
        n=args.n,
        T=args.T,
    )
    logger(
        {"test/solve_rate": float(solve_rate), "test/cell_accuracy": float(cell_acc)}
    )
