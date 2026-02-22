# /// script
# requires-python = "==3.12"
# dependencies = [
#     "einops",
#     "torch",
#     "datasets",
#     "ema-pytorch",
#     "simple-parsing",
#     "tqdm",
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

from einops import rearrange
from einops.layers.torch import Reduce

from ema_pytorch import EMA
from datasets import load_dataset

import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import Literal
from tqdm import tqdm
import simple_parsing
import wandb


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


def rms_norm(x):
    return F.rms_norm(x, (x.shape[-1],))


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
        self.x_norm, self.y_norm, self.z_norm = (
            nn.RMSNorm(h_dim),
            nn.RMSNorm(h_dim),
            nn.RMSNorm(h_dim),
        )

    def forward(self, x, y, z):
        h = self.x_norm(x) + self.y_norm(y) + self.z_norm(z)
        return rms_norm(self.blocks(h))


class InitState(nn.Module):
    def __init__(self, h_dim, mode, device):
        super().__init__()

        self.get_state = partial(torch.randn, h_dim, device=device)
        self.state = nn.Buffer(self.get_state()) if mode == "buffer" else None
        self.mode = mode

    def forward(self):
        return self.state if self.mode == "buffer" else self.get_state()


def train_batch(model, batch, opt, scheduler, cfg, logger=None):
    model.train()

    x_input, y_true = batch["inputs"], batch["labels"]
    b, device = x_input.shape[0], x_input.device
    N_sup = cfg.N_supervision

    alive = torch.ones(b, 1, device=device, dtype=torch.bool)
    y, z = None, None

    min_steps = (
        torch.rand(b, 1, device=device) <= cfg.halt_exploration_prob
    ) * torch.randint(low=2, high=N_sup + 1, size=(b, 1), device=device)

    corruption_std = torch.rand(()).item() * cfg.corruption_std

    if cfg.randomize_N_supervision:  # TODO better distribution
        N_sup = torch.randint(N_sup // 2, N_sup + 1, (1,), device=device).item()

    for step in range(N_sup):
        (y_new, z_new), y_hat, q_hat, loss, (rec_loss, halt_loss) = model(
            x_input, y, z, alive, y_true, n=cfg.n, T=cfg.T
        )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )
        last_grad_norm = float(grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

        keep_alive = q_hat.sigmoid() < cfg.halt_prob_thresh
        alive = alive & (keep_alive | (step < min_steps))

        if step > 0 and cfg.halt_prob_thresh <= 1.0:
            _a = rearrange(alive, "b 1 -> b 1 1")
            y, z = (torch.where(_a, y_new, y), torch.where(_a, z_new, z))
        else:
            y, z = y_new, z_new

        if corruption_std > 0.0:
            y = y + torch.randn_like(y) * corruption_std
            z = z + torch.randn_like(z) * corruption_std

        if (~alive).all():
            break

    scheduler.step()

    if logger is not None:
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
                    "train/logit_norm": float(y_hat.norm(dim=-1).mean()),
                    "train/grad_norm": last_grad_norm,
                    "train/acc": correct,
                    "train/solved": solved,
                }
            )


@torch.inference_mode()
def evaluate(model, data_loader, N_sup, cfg):
    model.eval()

    device = next(model.parameters()).device
    accs, solves = torch.zeros(N_sup, device=device), torch.zeros(N_sup, device=device)
    n_cells, n_puzzles = 0, 0

    it = tqdm(data_loader, desc="Evaluating")
    for batch in it:
        x_input, y_true = batch["inputs"], batch["labels"]

        pred_cells = []
        for _ in range(cfg.k_passes):
            y_hats_logits, _, _ = model.predict(
                x_input, N_supervision=N_sup, n=cfg.n, T=cfg.T
            )
            pred_cells.append(y_hats_logits.argmax(dim=-1))
        preds = rearrange(pred_cells, "k n b l -> k n b l").mode(dim=0).values

        for i in range(N_sup):
            accs[i] += (preds[i] == y_true).sum()
            solves[i] += (preds[i] == y_true).all(dim=-1).sum()

        n_cells += y_true.numel()
        n_puzzles += y_true.shape[0]

        it.set_postfix(
            acc=accs[-1].item() / n_cells, solved=solves[-1].item() / n_puzzles
        )

    accs, solves = accs / n_cells, solves / n_puzzles
    metrics = {
        "acc": accs[-1],
        "solved": solves[-1],
        "acc_delta_mid": accs[-1] - accs[N_sup // 2],
        "solved_delta_mid": solves[-1] - solves[N_sup // 2],
        "acc_delta_first": accs[-1] - accs[0],
        "solved_delta_first": solves[-1] - solves[0],
    }
    return metrics, accs, solves


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def model_factory(cfg, device):
    vocab_len, seq_len = 10, 81

    model = TRM(
        net=Net(
            seq_len=seq_len,
            h_dim=cfg.h_dim,
            n_layers=cfg.n_layers,
            expansion=cfg.mlp_factor,
        ),
        output_head=nn.Linear(cfg.h_dim, vocab_len),
        Q_head=nn.Sequential(Reduce("b l h -> b h", "mean"), nn.Linear(cfg.h_dim, 1)),
        input_embedding=nn.Embedding(vocab_len, cfg.h_dim, device=device),
        init_y=InitState(cfg.h_dim, mode=cfg.init_state, device=device),
        init_z=InitState(cfg.h_dim, mode=cfg.init_state, device=device),
        halt_loss_weight=cfg.halt_loss_weight,
    )
    return torch.compile(model) if not cfg.no_compile else model


@dataclass
class Config:
    # Model
    n_layers: int = 2
    h_dim: int = 512
    mlp_factor: int = 4
    init_state: Literal["buffer", "random"] = "buffer"

    # TRM recursion
    N_supervision: int = 16
    N_supervision_test: int | None = None
    n: int = 6
    T: int = 3
    halt_loss_weight: float = 0.5
    halt_prob_thresh: float = 0.5
    halt_exploration_prob: float = 0.1
    randomize_N_supervision: bool = False
    corruption_std: float = 0.0

    # Training
    batch_size: int = 768
    lr: float = 1e-4
    lr_warmup_steps: int = 2000 // 16
    weight_decay: float = 1.0
    ema_beta: float = 0.999**16
    max_grad_norm: float = 1.0
    epochs: int = 60_000 // 16
    steps: int | None = None
    data_seed: int = 42
    test_size: int | None = None
    n_augs: int = 1000

    # Infra
    no_compile: bool = False
    k_passes: int = 1
    val_every: int = 250
    eval_only: bool = False
    skip_eval: bool = False
    checkpoint: str | None = None
    wandb_project: str = "trm-sudoku"


if __name__ == "__main__":
    cfg = simple_parsing.parse(Config)
    cfg.N_supervision_test = cfg.N_supervision_test or cfg.N_supervision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = device.type == "cuda"
    if gpu:
        torch.set_float32_matmul_precision("high")
    print(f"Using: {device}")

    wandb.init(project=cfg.wandb_project, config=asdict(cfg), save_code=True)

    model = model_factory(cfg, device).to(device)
    print(model)
    print("No. of parameters:", sum(p.numel() for p in model.parameters()))

    ds = load_dataset(f"emiliocantuc/sudoku-extreme-1k-aug-{cfg.n_augs}")
    val_test = ds["test"].train_test_split(train_size=2048, seed=cfg.data_seed)
    train_ds, val_ds, test_ds = ds["train"], val_test["train"], val_test["test"]
    if cfg.test_size is not None:
        test_ds = test_ds.shuffle(seed=cfg.data_seed).select(range(cfg.test_size))

    for ds in (train_ds, val_ds, test_ds):
        ds.set_format(type="torch", columns=["inputs", "labels"])

    def collate_fn(batch):
        return {k: torch.stack([b[k] for b in batch]).to(device) for k in batch[0]}

    get_loader = partial(
        DataLoader, batch_size=cfg.batch_size, drop_last=True, collate_fn=collate_fn
    )
    train_loader = get_loader(train_ds, shuffle=True)
    val_loader = get_loader(val_ds, shuffle=False)
    test_loader = get_loader(test_ds, shuffle=False)

    ema = EMA(
        model,
        beta=cfg.ema_beta,
        forward_method_names=("predict",),
        update_every=1,
    ).to(device)

    def logger(data, step=None, print_every=1):
        wandb.log(data, step=step)
        if step and step % print_every == 0 and "train/loss" in data:
            print(
                f"step {step} {' | '.join([f'{m}: {data[f"train/{m}"]:.4f}' for m in ('loss', 'grad_norm', 'acc')])}"
            )

    if not cfg.eval_only:
        n_steps = cfg.steps or (cfg.epochs * len(train_loader))

        opt = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.95),
            weight_decay=cfg.weight_decay,
            fused=gpu,
        )

        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1e-8,
            total_iters=cfg.lr_warmup_steps,
        )
        if cfg.checkpoint and os.path.exists(cfg.checkpoint):
            ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            opt.load_state_dict(ckpt["opt"])
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"Loaded checkpoint from {cfg.checkpoint} for training")

        wandb.watch(model, log="all", log_freq=10)

        best_acc = 0.0
        for step, batch in tqdm(
            enumerate(cycle(train_loader), start=1),
            total=n_steps,
            desc="Training",
        ):
            train_batch(
                model=model,
                batch=batch,
                opt=opt,
                scheduler=scheduler,
                cfg=cfg,
                logger=partial(logger, step=step, print_every=10),
            )

            ema.update()

            if step >= n_steps:
                break

            if step % cfg.val_every == 0:
                metrics, _, _ = evaluate(ema, val_loader, cfg.N_supervision, cfg)
                logger({f"val/{k}": v for k, v in metrics.items()}, step=step)

                last_acc = metrics["acc"]
                if cfg.checkpoint and last_acc > best_acc:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        },
                        cfg.checkpoint,
                    )
                    print(f"Checkpoint saved to {cfg.checkpoint}")
                    best_acc = last_acc

    if cfg.skip_eval:
        wandb.finish()
        exit(0)

    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        ckpt = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        print(f"Loaded checkpoint from {cfg.checkpoint} for evaluation")

    ema.copy_params_from_ema_to_model()  # always evaluate EMA model
    metrics, accs, solves = evaluate(
        model, test_loader, N_sup=cfg.N_supervision_test, cfg=cfg
    )

    d = {f"test/{k}": v for k, v in metrics.items()}
    t = wandb.Table(
        data=list(zip(range(1, len(accs) + 1), accs.tolist(), solves.tolist())),
        columns=["N_sup", "acc", "solved"],
    )
    d["test/acc-N"] = wandb.plot.line(t, "N_sup", "acc", title="Test Acc vs N")
    d["test/solved-N"] = wandb.plot.line(t, "N_sup", "solved", title="Test Solved vs N")
    logger(d)
    wandb.finish()
