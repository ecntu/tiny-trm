# /// script
# requires-python = "==3.12"
# dependencies = [
#     "einops",
#     "torch",
#     "datasets",
#     "ema-pytorch",
#     "tqdm",
#     "accelerate",
#     "wandb",
# ]
# ///

# A single-file implementation of the Tiny Recursive Model (TRM) trained on Sudoku.
# Paper: https://arxiv.org/abs/2305.10445
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Reduce

from accelerate import Accelerator
from ema_pytorch import EMA
from datasets import load_dataset

import os
import shutil
import argparse
from collections import defaultdict
from functools import partial
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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
            z = self.net(x, y, z)
        y = self.net(torch.zeros_like(x), y, z)  # refine output answer
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
        y_hats, ys, zs = [], [], []
        y, z = self.init_y(), self.init_z()
        x = self.input_embedding(x_input)
        for step in range(N_supervision):
            (y, z), y_hat, _ = self.deep_recursion(x, y, z, n=n, T=T)
            y_hats.append(y_hat)
            ys.append(y)
            zs.append(z)
        return (
            rearrange(y_hats, "n b l c -> n b l c"),
            rearrange(ys, "n b l d -> n b l d"),
            rearrange(zs, "n b l d -> n b l d"),
        )

    def forward(self, x_input, y, z, alive, y_true, n=6, T=3):
        x = self.input_embedding(x_input)
        if y is None:
            y, z = self.init_y(), self.init_z()

        (y, z), y_hat, q_hat = self.deep_recursion(x, y, z, n=n, T=T)

        b, l = y_true.shape
        total_alive = alive.float().sum().clamp_min(1.0)

        rec_loss = (
            rearrange(
                F.cross_entropy(
                    rearrange(y_hat.float(), "b l c -> (b l) c"),
                    rearrange(y_true, "b l -> (b l)"),
                    reduction="none",
                ),
                "(b l) -> b l",
                b=b,
            )
            * (alive).float()
        ).sum() / (total_alive * l)

        # or fractional: (y_hat.argmax(dim=-1) == y_true).float().mean(dim=-1, keepdim=True)
        halt_target = (y_hat.argmax(dim=-1) == y_true).all(dim=-1, keepdim=True).float()
        halt_loss = (
            F.binary_cross_entropy_with_logits(
                q_hat.float(),
                halt_target.float(),
                reduction="none",
            )
            * (alive).float()
        ).sum() / total_alive

        loss = rec_loss + self.halt_loss_weight * halt_loss
        return (y, z), y_hat, q_hat, loss, (rec_loss, halt_loss)


def _find_multiple(a, b):
    return (-(a // -b)) * b


class SwiGLU(nn.Module):
    """SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)"""

    def __init__(self, d_model, expansion):
        super().__init__()
        d_inter = _find_multiple(round(d_model * expansion * 2 / 3), 256)
        self.W1 = nn.Linear(d_model, d_inter, bias=False)
        self.W3 = nn.Linear(d_model, d_inter, bias=False)
        self.W2 = nn.Linear(d_inter, d_model, bias=False)

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


def rms_norm(x, eps=1e-5):
    input_dtype = x.dtype
    x = x.float()
    variance = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x.to(input_dtype)


class MixerBlock(nn.Module):
    def __init__(self, seq_len, h_dim, expansion):
        super().__init__()
        self.l_mixer = SwiGLU(seq_len, expansion)
        self.d_mixer = SwiGLU(h_dim, expansion)

    def forward(self, h):
        o = rms_norm(h)
        o = rearrange(o, "b l d -> b d l")
        o = self.l_mixer(o)
        o = rearrange(o, "b d l -> b l d")

        h = o + h

        o = rms_norm(h)
        o = self.d_mixer(o)
        return o + h


class Net(nn.Module):
    def __init__(self, seq_len, h_dim, expansion, n_layers=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                MixerBlock(seq_len=seq_len, h_dim=h_dim, expansion=expansion)
                for _ in range(n_layers)
            )
        )

    def forward(self, x, y, z):
        return rms_norm(self.blocks(x + y + z))


class InitState(nn.Module):
    def __init__(self, h_dim, mode, device):
        super().__init__()

        self.get_state = partial(torch.randn, h_dim, device=device)
        self.state = nn.Buffer(self.get_state()) if mode == "buffer" else None
        self.mode = mode

    def forward(self):
        return self.state if self.mode == "buffer" else self.get_state()


def train_batch(
    accelerator,
    model,
    batch,
    opt,
    scheduler,
    N_supervision,
    n,
    T,
    halt_prob_thresh=0.5,
    max_grad_norm=1.0,
    logger=None,
):
    model.train()

    x_input, y_true = batch["inputs"], batch["labels"]

    alive = torch.ones(x_input.shape[0], 1, device=x_input.device, dtype=torch.bool)
    y, z = None, None

    for step in range(N_supervision):
        with accelerator.autocast():
            (y_new, z_new), y_hat, q_hat, loss, (rec_loss, halt_loss) = model(
                x_input, y, z, alive, y_true, n=n, T=T
            )

        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        last_grad_norm = float(grad_norm) if grad_norm is not None else 0.0
        opt.step()
        opt.zero_grad(set_to_none=True)

        should_halt = q_hat.sigmoid() >= halt_prob_thresh
        alive = alive & ~should_halt

        if step > 0 and halt_prob_thresh <= 1.0:
            _a = rearrange(alive, "b 1 -> b 1 1")
            y, z = (torch.where(_a, y_new, y), torch.where(_a, z_new, z))
        else:
            y, z = y_new, z_new

        if (~alive).all():
            break

    scheduler.step()

    if logger is not None and accelerator.is_main_process:
        with torch.no_grad():
            preds = y_hat.argmax(dim=-1)
            correct = (preds == y_true).float().mean()
            solved = (preds == y_true).all(dim=-1).float().mean()
            logger(
                {
                    "train/loss": loss,
                    "train/rec_loss": rec_loss,
                    "train/halt_loss": halt_loss,
                    "train/prop_alive": alive.float().mean(),
                    "train/batch_steps": step + 1,
                    "train/lr": opt.param_groups[0]["lr"],
                    "train/token_corr_y": float(token_corr(y)),
                    "train/token_corr_z": float(token_corr(z)),
                    "train/logit_norm": float(y_hat.norm(dim=-1).mean()),
                    "train/grad_norm": last_grad_norm,
                    "train/acc": correct,
                    "train/solved": solved,
                }
            )


@torch.inference_mode()
def evaluate(
    accelerator, model, data_loader, N_supervisions=[16], n=6, T=3, k_passes=1
):
    model.eval()
    it = (
        tqdm(data_loader, desc="Evaluating")
        if accelerator.is_main_process
        else data_loader
    )

    _Ns = torch.as_tensor(sorted(N_supervisions)) - 1  # zero-indexed
    metrics = defaultdict(int)
    total_cells, total_puzzles = 0, 0

    for batch in it:
        x_input, y_true = batch["inputs"], batch["labels"]

        pred_cells = []
        for _ in range(k_passes):
            with accelerator.autocast():
                y_hats_logits, _, _ = model.predict(
                    x_input, N_supervision=max(N_supervisions), n=n, T=T
                )  # (n, b, l, c)

            pred_cells.append(y_hats_logits[_Ns].argmax(dim=-1))

        pred_cells = rearrange(pred_cells, "k n b l -> k n b l").mode(dim=0).values

        # gather across processes for global metrics
        pred_cells = accelerator.gather_for_metrics(pred_cells)
        y_true = accelerator.gather_for_metrics(y_true)

        correct_cells = reduce((pred_cells == y_true), "n b l -> n", "sum")
        solved = reduce((pred_cells == y_true).all(dim=-1), "n b -> n", "sum")
        for N, c, s in zip(_Ns, correct_cells, solved):
            metrics[f"acc_N{N.item() + 1}"] += c
            metrics[f"solved_N{N.item() + 1}"] += s

        total_cells += y_true.numel()
        total_puzzles += y_true.shape[0]

        if accelerator.is_main_process:
            it.set_postfix(
                {
                    "sol": solved[-1].item() / total_puzzles,
                    "acc": correct_cells[-1].item() / total_cells,
                }
            )

    metrics = {
        k: v / (total_cells if "acc" in k else total_puzzles)
        for k, v in metrics.items()
    }
    last_acc = metrics[f"acc_N{_Ns[-1].item() + 1}"]
    return metrics, last_acc


# https://arxiv.org/abs/2211.09961
@torch.no_grad()
def asymptotic_alignment_score(
    accelerator, model, loader, N_supervisions=[16], T=3, n=6, n_batches=None
):
    _Ns = torch.as_tensor(sorted(N_supervisions)) - 1  # zero-indexed

    def aa_score_batch(x_input):
        with accelerator.autocast():
            _, ys, zs = model.predict(
                x_input, N_supervision=max(N_supervisions), n=n, T=T
            )
            _, ys2, zs2 = model.predict(
                x_input, N_supervision=max(N_supervisions), n=n, T=T
            )
            ys, zs, ys2, zs2 = (ys[_Ns], zs[_Ns], ys2[_Ns], zs2[_Ns])

        aa_score = lambda a, b: F.cosine_similarity(
            rearrange(a, "n b l d -> n b (l d)"),
            rearrange(b, "n b l d -> n b (l d)"),
            dim=-1,
        ).sum(dim=1)

        return aa_score(ys, ys2), aa_score(zs, zs2)

    n_batches = n_batches or len(loader)
    n_examples = 0
    y_aa_score = torch.zeros(len(_Ns), device=accelerator.device)
    z_aa_score = torch.zeros(len(_Ns), device=accelerator.device)

    for _, batch in zip(range(n_batches), loader):
        x_input = batch["inputs"].to(device)

        y_aa, z_aa = aa_score_batch(x_input)
        y_aa_score += y_aa
        z_aa_score += z_aa
        n_examples += x_input.size(0)

    scores = {}
    for N, y_s, z_s in zip(_Ns, y_aa_score, z_aa_score):
        scores[f"y_aa_N{N.item() + 1}"] = y_s.item() / n_examples
        scores[f"z_aa_N{N.item() + 1}"] = z_s.item() / n_examples

    return scores


def token_corr(h, eps=1e-8):
    # To guard against representation collapse
    h = h.detach().float()
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


def model_factory(args, device, compile):
    vocab_len, seq_len = 10, 81

    model = TRM(
        net=Net(
            seq_len=seq_len,
            h_dim=args.h_dim,
            n_layers=args.n_layers,
            expansion=args.mlp_factor,
        ),
        output_head=nn.Linear(args.h_dim, vocab_len),
        Q_head=nn.Sequential(Reduce("b l h -> b h", "mean"), nn.Linear(args.h_dim, 1)),
        input_embedding=nn.Embedding(vocab_len, args.h_dim),
        init_y=InitState(args.h_dim, mode=args.init_state, device=device),
        init_z=InitState(args.h_dim, mode=args.init_state, device=device),
        halt_loss_weight=args.halt_loss_weight,
    )
    if compile:
        model = torch.compile(model)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--mlp_factor", type=int, default=4)
    parser.add_argument(
        "--init_state", type=str, default="buffer", choices=["buffer", "random"]
    )

    parser.add_argument("--N_supervision", type=int, default=16)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--halt_loss_weight", type=float, default=0.5)
    parser.add_argument("--halt_prob_thresh", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=2000 // 16)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--ema_beta", type=float, default=0.999**16)
    parser.add_argument("--epochs", type=int, default=60_000 // 16)
    parser.add_argument("--steps", type=int, default=None)

    parser.add_argument(
        "--mixed_precision", default="no", choices=["bf16", "fp16", "no"]
    )
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--k_passes", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=50)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--log_with", type=str, default=None)

    args = parser.parse_args()

    accelerator = Accelerator(
        log_with=args.log_with, mixed_precision=args.mixed_precision
    )
    accelerator.init_trackers(
        "trm-sudoku", config=vars(args), init_kwargs={"wandb": {"save_code": True}}
    )
    if args.checkpoint and accelerator.is_main_process:  # add script to checkpoint
        accelerator.register_save_state_pre_hook(
            lambda _, __, dir: os.makedirs(dir, exist_ok=True)
            or shutil.copy(__file__, os.path.join(dir, "main.py"))
        )

    device = accelerator.device
    gpu = device.type == "cuda"
    accelerator.print(f"Using: {device}, {accelerator.mixed_precision}")

    model = model_factory(args, device, not args.no_compile)
    accelerator.print(model)
    accelerator.print("No. of parameters:", sum(p.numel() for p in model.parameters()))

    ds_path = "emiliocantuc/sudoku-extreme-1k-aug-1000"
    train_ds = load_dataset(ds_path, split="train")
    val_ds = load_dataset(ds_path, split="test[:1024]")
    test_ds = load_dataset(ds_path, split="test")
    for ds in (train_ds, val_ds, test_ds):
        ds.set_format(type="torch", columns=["inputs", "labels"])

    workers = min(8, os.cpu_count() or 4)
    get_loader = partial(
        DataLoader,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=gpu,
        num_workers=workers if gpu else 0,
        persistent_workers=workers > 0 and gpu,
        prefetch_factor=4 if workers > 0 and gpu else None,
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
        update_every=1,
    ).to(device)  # TODO only on main?
    accelerator.register_for_checkpointing(ema)

    if "wandb" in accelerator.trackers and accelerator.is_main_process:
        accelerator.get_tracker("wandb", unwrap=True).watch(
            model, log="all", log_freq=10
        )

    def logger(data, step=None, print_every=1):
        accelerator.log(data, step=step)
        if step and step % print_every == 0 and "train/loss" in data:
            accelerator.print(
                f"step {step} {' | '.join([f'{m}: {data[f"train/{m}"]:.4f}' for m in ('loss', 'grad_norm', 'acc')])}"
            )

    if not args.eval_only:
        n_steps = args.steps or (args.epochs * len(train_loader))

        opt = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=gpu,
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1e-8,
            total_iters=args.lr_warmup_steps,
        )
        opt, scheduler = accelerator.prepare(opt, scheduler)

        if args.checkpoint and os.path.exists(args.checkpoint):
            accelerator.load_state(args.checkpoint)
            accelerator.print(f"Loaded checkpoint from {args.checkpoint} for training")

        best_acc = 0.0
        for step, batch in tqdm(
            enumerate(cycle(train_loader), start=1),
            total=n_steps,
            disable=not accelerator.is_main_process,
            desc="Training",
        ):
            train_batch(
                accelerator=accelerator,
                model=model,
                batch=batch,
                opt=opt,
                scheduler=scheduler,
                N_supervision=args.N_supervision,
                n=args.n,
                T=args.T,
                halt_prob_thresh=args.halt_prob_thresh,
                logger=partial(logger, step=step, print_every=10),
            )

            ema.update()

            if step >= n_steps:
                break

            if step % args.val_every == 0:
                metrics, last_acc = evaluate(
                    accelerator,
                    ema,
                    val_loader,
                    N_supervisions=[
                        args.N_supervision // 2,
                        args.N_supervision,
                        args.N_supervision * 2,
                    ],
                    n=args.n,
                    T=args.T,
                )

                aa_scores = asymptotic_alignment_score(
                    accelerator,
                    model,
                    val_loader,
                    N_supervisions=[
                        args.N_supervision // 2,
                        args.N_supervision,
                        args.N_supervision * 2,
                    ],
                    n=args.n,
                    T=args.T,
                    n_batches=8,
                )
                metrics.update(aa_scores)
                logger(
                    {f"val/{k}": v for k, v in metrics.items()},
                    step=step,
                )

                if args.checkpoint and last_acc > best_acc:
                    accelerator.save_state(args.checkpoint)
                    accelerator.print(f"Checkpoint saved to {args.checkpoint}")
                    best_acc = last_acc

    if args.skip_eval:
        accelerator.end_training()
        exit(0)

    if args.checkpoint and os.path.exists(args.checkpoint):
        accelerator.load_state(args.checkpoint)
        accelerator.print(f"Loaded checkpoint from {args.checkpoint} for evaluation")

    ema.copy_params_from_ema_to_model()  # always evaluate EMA model
    solve_rate, cell_acc = evaluate(
        accelerator,
        model,
        test_loader,
        N_supervisions=[args.N_supervision],
        n=args.n,
        T=args.T,
        k_passes=args.k_passes,
    )
    logger({"test/solve_rate": solve_rate, "test/cell_accuracy": cell_acc})
    accelerator.end_training()
