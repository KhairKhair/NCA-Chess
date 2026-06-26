import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataset import NUM_RETURN_BUCKETS


def seed(x_vis, total_chn):
    n, cv, h, w = x_vis.shape
    if cv == total_chn:
        return x_vis
    pad = torch.zeros(n, total_chn - cv, h, w, device=x_vis.device)
    return torch.cat([x_vis, pad], dim=1)


def rollout(model, x_vis, steps, update_rate):
    state = seed(x_vis, model.chn + model.out_chn)
    for _ in range(steps):
        state = model(state, update_rate)
    return state

def train(
    model,
    train_data,
    test_data,
    epochs=200,
    bs=512,
    lr=1e-3,
    eval_every=10,
    eval_n=500,
    device=None,
    min_steps=32,
    max_steps=64,
    lr_step=2000,        # StepLR step_size, in OPTIMIZER STEPS (see note)
    lr_gamma=0.3,        # reference value
    grad_norm=True,      # per-parameter gradient normalization (Growing-NCA trick)
    update_rate=0.5,
):
    X, Y, fens_tr = train_data
    Xte, Yte, fens_te = test_data

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)               # AdamW
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=lr_step, gamma=lr_gamma                       # StepLR
    )
    N = X.shape[0]

    history = {"epoch": [], "loss": [], "test_loss": [],
               "square_acc": [], "move_acc": [], "lr": []}

    pbar = tqdm(range(epochs))
    for ep in pbar:
        model.train()
        perm = torch.randperm(N)
        epoch_losses = []

        for idx in perm.split(bs):
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            steps = torch.randint(min_steps, max_steps + 1, (1,)).item()
            pred = rollout(model, xb, steps, update_rate)
            logits = pred[:, -NUM_RETURN_BUCKETS:].mean(dim=[-2, -1])
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad()
            loss.backward()

            if grad_norm:
                # normalize each parameter's gradient to unit norm
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad /= (p.grad.norm() + 1e-8)

            opt.step()
            scheduler.step()        # stepped per optimizer step, like the reference
            epoch_losses.append(loss.item())

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        history["epoch"].append(ep)
        history["loss"].append(avg_train_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        test_loss = square_acc = move_acc = None
        if (ep + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                n_te = min(eval_n, Xte.shape[0])
                idx_te = torch.randperm(Xte.shape[0])[:n_te]
                xte = Xte[idx_te].to(device)
                yte = Yte[idx_te].to(device)
                steps_te = (min_steps + max_steps) // 2
                pred_te = rollout(model, xte, steps_te)
                logits_te = pred_te[:, -NUM_RETURN_BUCKETS:].mean(dim=[-2, -1])
                test_loss = F.cross_entropy(logits_te, yte).item()
                preds = logits_te.argmax(dim=1)
                square_acc = (preds == yte).float().mean().item()
                top5 = logits_te.topk(5, dim=1).indices
                move_acc = (top5 == yte.unsqueeze(1)).any(dim=1).float().mean().item()

        history["test_loss"].append(test_loss)
        history["square_acc"].append(square_acc)
        history["move_acc"].append(move_acc)

        if test_loss is not None:
            pbar.set_postfix(loss=f"{avg_train_loss:.4f}",
                             test_loss=f"{test_loss:.4f}",
                             top1=f"{square_acc:.2%}", top5=f"{move_acc:.2%}")
        else:
            pbar.set_postfix(loss=f"{avg_train_loss:.4f}")

    return history
