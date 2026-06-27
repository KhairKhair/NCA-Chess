# CA_policy.py
# Policy-head chess NCA: factored from/to logit maps, cross-entropy training,
# legality-aware joint scoring at eval, mirror augmentation on train only.
from tqdm import tqdm
import chess
import torch
from helper import sq_to_canonical
from model import policy_loss
 
@torch.no_grad()
def move_accuracy(model, X, fens, fl, tl, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    fr, to = model(X.to(device).float())
    fr, to = fr.cpu(), to.cpu()
    hits = 0
    for i in range(X.shape[0]):
        board = chess.Board(fens[i]); turn = board.turn
        best, best_sc = None, -1e30
        for mv in board.legal_moves:
            f = sq_to_canonical(mv.from_square, turn)
            t = sq_to_canonical(mv.to_square, turn)
            sc = fr[i, f].item() + to[i, t].item()
            if sc > best_sc:
                best_sc, best = sc, (f, t)
        if best == (int(fl[i]), int(tl[i])):
            hits += 1
    return hits / X.shape[0]
 
def train(model, train_data, test_data, epochs=200, bs=512, lr=1e-3,
          eval_every=5, eval_n=500, device=None):
    X, fl, tl, fens_tr = train_data
    Xte, flte, tlte, fens_te = test_data
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = X.shape[0]
    history = {"loss": [], "epoch": [], "train_acc": [], "test_acc": []}
    pbar = tqdm(range(epochs))
    for ep in pbar:
        model.train()
        for idx in torch.randperm(N).split(bs):
            xb = X[idx].to(device).float(); fb = fl[idx].to(device); tb = tl[idx].to(device)
            frl, tol = model(xb)
            loss = policy_loss(frl, tol, fb, tb)
            opt.zero_grad(); loss.backward(); opt.step()
        history["loss"].append(loss.item())
        if ep % eval_every == 0 or ep == epochs - 1:
            tr_acc = move_accuracy(model, X[:eval_n], fens_tr[:eval_n],
                                   fl[:eval_n], tl[:eval_n], device)
            te_acc = move_accuracy(model, Xte[:eval_n], fens_te[:eval_n],
                                   flte[:eval_n], tlte[:eval_n], device)
            history["epoch"].append(ep)
            history["train_acc"].append(tr_acc)
            history["test_acc"].append(te_acc)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             tr=f"{tr_acc:.3f}", te=f"{te_acc:.3f}")
    return history
 