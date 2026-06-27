from tqdm import tqdm
import chess
import torch
import torch.nn.functional as F

 
def rollout(model, x, steps):
    y = x
    for _ in range(steps):
        y = model(y)
    return y


@torch.no_grad()
def eval_mse(model, Xte, Yte, device, eval_n=None, min_steps=32, max_steps=64):
    model.eval()

    if eval_n is not None and eval_n < Xte.shape[0]:
        idx = torch.randperm(Xte.shape[0])[:eval_n]
        xb = Xte[idx].to(device)
        yb = Yte[idx].to(device)
    else:
        xb = Xte.to(device)
        yb = Yte.to(device)

    steps = torch.randint(min_steps, max_steps + 1, (1,)).item()
    pred = rollout(model, xb, steps)

    loss = F.mse_loss(pred, yb)
    return loss.item()

def train(
    model,
    train_data,
    test_data,
    epochs=200,
    bs=512,
    lr=1e-3,
    eval_every=5,
    eval_n=500,
    device=None,
    min_steps=32,
    max_steps=64,
):
    X, Y, fens_tr = train_data
    Xte, Yte, fens_te = test_data

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on: {device}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    N = X.shape[0]

    history = {
        "epoch": [],
        "loss": [],
        "test_loss": [],
    }

    pbar = tqdm(range(epochs))

    for ep in pbar:
        model.train()

        perm = torch.randperm(N)
        epoch_losses = []

        for idx in perm.split(bs):
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            steps = torch.randint(min_steps, max_steps + 1, (1,)).item()

            pred = rollout(model, xb, steps)
            loss = F.mse_loss(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_losses.append(loss.item())

        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        history["epoch"].append(ep)
        history["loss"].append(avg_train_loss)

        # Only test every eval_every epochs
        if ep % eval_every == 0 or ep == epochs - 1:
            test_loss = eval_mse(
                model,
                Xte,
                Yte,
                device=device,
                eval_n=eval_n,
                min_steps=min_steps,
                max_steps=max_steps,
            )
        else:
            test_loss = None

        history["test_loss"].append(test_loss)

        if test_loss is not None:
            pbar.set_postfix(
                loss=f"{avg_train_loss:.4f}",
                test_loss=f"{test_loss:.4f}",
            )
        else:
            pbar.set_postfix(
                loss=f"{avg_train_loss:.4f}",
            )

    return history