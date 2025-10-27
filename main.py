# /// script
# requires-python = "==3.12"
# dependencies = [
#     "einops",
#     "torch",
#     "datasets",
#     "ema-pytorch",
# ]
# ///

# A single-file implementation of the Tiny Recursive Model (TRM) trained on Sudoku.
# Paper: https://arxiv.org/abs/2305.10445

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange
from einops.layers.torch import Rearrange

from ema_pytorch import EMA
from datasets import load_dataset


class TRM(nn.Module):
    def __init__(self, net, output_head, Q_head, input_embedding, init_z, init_y):
        super().__init__()
        self.net = net
        self.output_head = output_head
        self.Q_head = Q_head
        self.input_embedding = input_embedding
        self.init_z = init_z
        self.init_y = init_y

    def latent_recursion(self, x, y, z, n=6):
        for i in range(n):  # latent reasoning
            z = self.net(x, y, z)
        y = self.net(y, z)  # refine output answer
        return y, z

    def deep_recursion(self, x, y, z, n=6, T=3):
        # recursing T-1 times to improve y and z (no gradients needed)
        with torch.no_grad():
            for j in range(T - 1):
                y, z = self.latent_recursion(x, y, z, n)
        # recursing once to improve y and z
        y, z = self.latent_recursion(x, y, z, n)
        return (y.detach(), z.detach()), self.output_head(z), self.Q_head(z)

    @torch.no_grad()
    def predict(self, x_input, N_supervision=16, n=6, T=3):
        y_hats = []
        z, y = self.init_z, self.init_y
        for step in range(N_supervision):
            x = self.input_embedding(x_input)
            (y, z), y_hat, _ = self.deep_recursion(x, y, z, n=n, T=T)
            y_hats.append(y_hat)
        return rearrange(y_hats, "n b l c -> n b l c")

    def forward(self, x_input, y, z, y_true, n=6, T=3):
        x = self.input_embedding(x_input)
        (y, z), y_hat, q_hat = model.deep_recursion(x, y, z, n=n, T=T)

        loss = F.cross_entropy(
            rearrange(y_hat, "b l c -> (b l) c"), rearrange(y_true, "b l -> (b l)")
        )  # TODO why does the paper reduce 'b ... -> b' with mean and then sum over b?
        loss += F.binary_cross_entropy_with_logits(
            q_hat,
            (y_hat.argmax(dim=-1) == y_true)
            .all(dim=1, keepdim=True)
            .float(),  # TODO try partial credit (mean)?
        )

        return (y, z), y_hat, q_hat, loss


class SwiGLU(nn.Module):
    """SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)"""

    def __init__(self, d_model, factor=4, device=None, dtype=None):
        super().__init__()
        d_ff = int(d_model * factor)
        self.W1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


def rms_norm(x, eps=1e-8):
    dtype = x.dtype  # TODO will x ever not be float32?
    x = x.to(torch.float32)
    var = x.square().mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x.to(dtype)


class Net(nn.Module):
    """MLP-Mixer style with post-norms"""

    def __init__(self, seq_len, h_dim, factor=4, device=None, dtype=None):
        super().__init__()
        self.l_mixer = SwiGLU(seq_len, factor=factor, device=device, dtype=dtype)
        self.d_mixer = SwiGLU(h_dim, factor=factor, device=device, dtype=dtype)

    def forward(self, y, z, x=None):
        h = (x + y + z) if x is not None else (y + z)

        o = rearrange(h, "b l d -> b d l")
        o = self.l_mixer(o)

        o = rearrange(o, "b d l -> b l d")
        h = o + h
        h = rms_norm(h)

        o = self.d_mixer(o)
        return rms_norm(o + h)


class InputEmbedding(nn.Module):
    def __init__(self, vocab_len, seq_len, h_dim):
        super().__init__()
        self.vocab_emb = nn.Embedding(vocab_len, h_dim)
        self.pos_emb = nn.Embedding(seq_len, h_dim)

    def forward(self, x_input):
        return self.vocab_emb(x_input) + self.pos_emb.weight


class Select(nn.Module):
    def __init__(self, index: int, dim: int):
        super().__init__()
        self.dim, self.index = dim, index

    def forward(self, x):
        return x.select(self.dim, self.index)


def paper_model_factory(
    vocab_size, seq_len, h_dim, h_factor=4, device=None, dtype=None
):
    input_embedding = InputEmbedding(vocab_size, seq_len, h_dim)
    net = Net(seq_len, h_dim, factor=h_factor)
    output_head = nn.Linear(h_dim, vocab_size)
    Q_head = nn.Sequential(
        Rearrange("b l h -> b l h"),
        Select(0, dim=1),
        nn.Linear(h_dim, 1),
    )
    init_z = nn.Parameter(torch.randn(h_dim) * 1e-2)
    init_y = nn.Parameter(torch.randn(h_dim) * 1e-2)
    return TRM(net, output_head, Q_head, input_embedding, init_z, init_y)


def train_batch(
    batch, opt, scheduler, N_supervision, n, T, halt_prob_thresh=0.5, device=None
):
    model.train()
    x_input, y_true = batch["inputs"].to(device), batch["labels"].to(device)
    z, y = model.init_z, model.init_y

    for step in range(N_supervision):
        (y, z), y_hat, q_hat, loss = model(x_input, y, z, y_true, n=n, T=T)

        print(loss.item())

        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        if q_hat.sigmoid().gt(halt_prob_thresh).all():
            print("halted")
            break


@torch.no_grad()
def evaluate(model, data_loader, N_supervision=16, n=6, T=3, device=None):
    model.eval()
    total, correct = 0, 0
    for batch in data_loader:
        x_input, y_true = batch["inputs"].to(device), batch["labels"].to(device)
        y_hats_logits = model.predict(x_input, N_supervision=N_supervision, n=n, T=T)
        total += y_true.numel()
        correct += (y_hats_logits[-1].argmax(-1) == y_true).sum().item()
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_len", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=81)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--N_supervision", type=int, default=16)
    parser.add_argument("--N_supervision_eval", type=int, default=None)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--T", type=int, default=3)
    parser.add_argument("--halt_prob_thresh", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_iters", type=int, default=2000)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--ema_beta", type=float, default=0.999)
    parser.add_argument("--epochs", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("Using device:", device)

    model = paper_model_factory(
        vocab_size=args.vocab_len,
        seq_len=args.seq_len,
        h_dim=args.h_dim,
        h_factor=4,
        device=device,
        dtype=dtype,
    ).to(device)
    ema = EMA(model, beta=args.ema_beta)

    print(model)

    ds_path = "emiliocantuc/sudoku-extreme-1k-aug-1000"
    train_ds = load_dataset(ds_path, split="train").shuffle(seed=42)
    train_ds.set_format(type="torch", columns=["inputs", "labels"])

    val_ds = load_dataset(ds_path, split="test[:1024]")
    val_ds.set_format(type="torch", columns=["inputs", "labels"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    opt = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=args.lr_warmup_iters
    )

    for epoch in range(args.epochs):
        for i, batch in enumerate(train_loader):
            print(f"Epoch {epoch} | Batch {i}")
            train_batch(
                batch,
                opt=opt,
                scheduler=scheduler,
                N_supervision=args.N_supervision,
                n=args.n,
                T=args.T,
                halt_prob_thresh=args.halt_prob_thresh,
                device=device,
            )
            ema.update()

        acc = evaluate(
            model,
            val_loader,
            N_supervision=args.N_supervision_eval or args.N_supervision,
            n=args.n,
            T=args.T,
            device=device,
        )

        print(f"Epoch {epoch} | Val Accuracy: {acc:.4f}")
