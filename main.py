# /// script
# requires-python = "==3.12"
# dependencies = [
#     "einops",
#     "torch",
#     "datasets",
#     "ema-pytorch",
#     "tqdm",
#     "wandb",
# ]
# ///

# A single-file implementation of the Tiny Recursive Model (TRM) trained on Sudoku.
# Paper: https://arxiv.org/abs/2305.10445

from functools import partial
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange
from einops.layers.torch import Reduce

from ema_pytorch import EMA
from datasets import load_dataset
from tqdm import tqdm


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
        return (y.detach(), z.detach()), self.output_head(z), self.Q_head(z)

    @torch.no_grad()
    def predict(self, x_input, N_supervision=16, n=6, T=3):
        y_hats = []
        z, y = self.init_z, self.init_y
        x = self.input_embedding(x_input)
        for step in range(N_supervision):
            (y, z), y_hat, _ = self.deep_recursion(x, y, z, n=n, T=T)
            y_hats.append(y_hat)
        return rearrange(y_hats, "n b l c -> n b l c")

    def forward(self, x_input, y, z, y_true, n=6, T=3):
        x = self.input_embedding(x_input)
        (y, z), y_hat, q_hat = self.deep_recursion(x, y, z, n=n, T=T)

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

    def __init__(self, d_model, factor=4):
        super().__init__()
        d_ff = int(d_model * factor)
        self.W1 = nn.Linear(d_model, d_ff)
        self.W3 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(F.silu(self.W1(x)) * self.W3(x))


class Net(nn.Module):
    """MLP-Mixer style with post-norms"""

    def __init__(self, seq_len, h_dim, factor=4):
        super().__init__()
        self.l_mixer = SwiGLU(seq_len, factor=factor)
        self.d_mixer = SwiGLU(h_dim, factor=factor)
        self.l_norm = nn.LayerNorm(h_dim)
        self.d_norm = nn.LayerNorm(h_dim)

    def forward(self, y, z, x=None):
        h = (x + y + z) if x is not None else (y + z)

        o = rearrange(h, "b l d -> b d l")
        o = self.l_mixer(o)

        o = rearrange(o, "b d l -> b l d")
        h = o + h
        h = self.l_norm(h)

        o = self.d_mixer(h)
        return self.d_norm(o + h)


class InputEmbedding(nn.Module):
    def __init__(self, vocab_len, seq_len, h_dim):
        super().__init__()
        self.vocab_emb = nn.Embedding(vocab_len, h_dim)
        self.pos_emb = nn.Embedding(seq_len, h_dim)

    def forward(self, x_input):
        return self.vocab_emb(x_input) + self.pos_emb.weight


def train_batch(
    model, batch, opt, scheduler, N_supervision, n, T, halt_prob_thresh=0.5, device=None
):
    model.train()
    x_input, y_true = batch["inputs"].to(device), batch["labels"].to(device)
    z, y = model.init_z, model.init_y

    for step in range(N_supervision):
        (y, z), y_hat, q_hat, loss = model(x_input, y, z, y_true, n=n, T=T)

        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        halt_probs = q_hat.sigmoid()
        if halt_probs.gt(halt_prob_thresh).all():
            break

    return loss.detach().item(), halt_probs.detach(), step + 1


@torch.inference_mode()
def evaluate(model, data_loader, N_supervision=16, n=6, T=3, device=None):
    model.eval()
    total, correct = 0, 0
    with tqdm(data_loader, desc="Evaluating", leave=True) as pbar:
        for batch in pbar:
            x_input, y_true = batch["inputs"].to(device), batch["labels"].to(device)
            y_hats_logits = model.predict(
                x_input, N_supervision=N_supervision, n=n, T=T
            )
            total += y_true.numel()
            correct += (y_hats_logits[-1].argmax(-1) == y_true).sum().item()
            pbar.set_postfix({"acc": correct / total})
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--yz_init_std", type=float, default=1e-2)

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
    parser.add_argument("--steps", type=int, default=None)

    parser.add_argument("--val_every", type=int, default=50)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default="./tmp.pt")
    parser.add_argument("--log_wandb", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("Using device:", device)

    vocab_len, seq_len = 10, 81

    model = TRM(
        net=Net(seq_len=seq_len, h_dim=args.h_dim, factor=4),
        output_head=nn.Linear(args.h_dim, vocab_len),
        Q_head=nn.Sequential(Reduce("b l h -> b h", "mean"), nn.Linear(args.h_dim, 1)),
        input_embedding=InputEmbedding(
            vocab_len=vocab_len, seq_len=seq_len, h_dim=args.h_dim
        ),
        init_z=nn.Parameter(torch.randn(args.h_dim) * args.yz_init_std),
        init_y=nn.Parameter(torch.randn(args.h_dim) * args.yz_init_std),
    ).to(device=device, dtype=dtype)

    model = torch.compile(model)
    ema = EMA(
        model,
        beta=args.ema_beta,
        forward_method_names=("predict",),
    )

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
    )
    train_loader = get_loader(train_ds, shuffle=True)
    val_loader = get_loader(val_ds, shuffle=False)
    test_loader = get_loader(test_ds, shuffle=False)

    # Note: checkpoints are not really meant for cont. training here, just for eval-only runs.
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        model.load_state_dict(torch.load(args.checkpoint_path))
        print(f"Loaded checkpoint from {args.checkpoint_path}")

    if not args.eval_only:
        opt = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=args.lr_warmup_iters
        )

        if args.log_wandb:
            import wandb

            wandb.init(
                project="trm-sudoku",
                config=vars(args),
                settings=wandb.Settings(code_dir="."),
            )
            wandb.watch(model, log="all")

        def cycle(loader):
            while True:
                for batch in loader:
                    yield batch

        n_steps = args.steps or (args.epochs * len(train_loader))

        # train loop
        for step, batch in enumerate(cycle(train_loader), 1):
            loss, halt_probs, batch_steps = train_batch(
                model,
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

            print(
                f"Loss: {loss:.3f} | Steps: {batch_steps} | Halt Probs: {halt_probs.mean().item():.3f}"
            )

            if step >= n_steps:
                break

            if step % args.val_every == 0:
                acc = evaluate(
                    ema,
                    val_loader,
                    N_supervision=args.N_supervision_eval or args.N_supervision,
                    n=args.n,
                    T=args.T,
                    device=device,
                )
                if args.log_wandb:
                    wandb.log(
                        {
                            "train/step": step,
                            "train/epoch": step // len(train_loader),
                            "train/loss": float(loss),
                            "train/halt_prob_mean": float(halt_probs.mean().item()),
                            "train/halt_prob_std": float(halt_probs.std().item()),
                            "train/batch_steps": int(batch_steps),
                            "train/lr": float(opt.param_groups[0]["lr"]),
                            "val/accuracy": float(acc),
                        }
                    )

        ema.copy_params_from_ema_to_model()

        if args.checkpoint_path:
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"Checkpoint saved to {args.checkpoint_path}")

    acc = evaluate(
        model,
        test_loader,
        N_supervision=args.N_supervision_eval or args.N_supervision,
        n=args.n,
        T=args.T,
        device=device,
    )
    print(f"Eval Accuracy: {acc:.4f}")
    if args.log_wandb:
        wandb.log({"test/accuracy": float(acc)})
