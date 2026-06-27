import chess
import torch
from datasets import load_dataset
from helper import canonical_tensor, sq_to_canonical
from tqdm import tqdm


def games_to_policy_dataset(n_positions=20000, skip_opening=6, max_per_game=12,
                            eval_cap=20000, cache="humanpolicy.pt"):
    """Stream games -> (X uint8, fl, tl, fens).

    X is stored as uint8 (planes are 0/1) -> 4x smaller than float32, lossless.
    fens are kept only for the first `eval_cap` positions (training never uses
    them; only legality-aware move_accuracy does, and it sees << eval_cap).
    """
    ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)

    X = None  # lazily allocated once we know the plane shape
    FL, TL, fens = [], [], []
    count = 0
    pbar = tqdm(total=n_positions, desc="positions")

    for game in ds:
        ucis = game.get("moves_uci") or []
        if not ucis:
            continue
        board = chess.Board()
        taken = 0
        for i, uci in enumerate(ucis):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            if move not in board.legal_moves:
                break
            if i >= skip_opening and taken < max_per_game:
                # skip opening book, cap positions per game for diversity
                turn = board.turn
                t = canonical_tensor(board).to(torch.uint8)
                if X is None:
                    X = torch.empty(n_positions, *t.shape, dtype=torch.uint8)
                X[count] = t
                FL.append(sq_to_canonical(move.from_square, turn))
                TL.append(sq_to_canonical(move.to_square, turn))
                if count < eval_cap:
                    fens.append(board.fen())
                count += 1
                taken += 1
                pbar.update(1)
            board.push(move)
            if count >= n_positions:
                break
        if count >= n_positions:
            break

    pbar.close()
    X = X[:count]
    fl = torch.tensor(FL, dtype=torch.long)
    tl = torch.tensor(TL, dtype=torch.long)
    torch.save({"X": X, "fl": fl, "tl": tl, "fens": fens}, cache)
    print(f"built {X.shape[0]} positions, {len(fens)} fens -> {cache}")
    return X, fl, tl, fens


def load_split(cache="humanpolicy.pt", test_frac=0.1, seed=0):
    """Train/test split. Test set and the train-eval prefix are drawn ONLY from
    fen'd positions, so move_accuracy (which needs a board) always has its FEN.
    X stays uint8 here; cast to float per-batch in the training loop."""
    d = torch.load(cache)
    X, fl, tl, fens = d["X"], d["fl"], d["tl"], d["fens"]
    n = X.shape[0]
    n_fen = len(fens)  # positions that carry a FEN

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_fen, generator=g).tolist()
    n_test = int(n_fen * test_frac)
    te_i = perm[:n_test]
    tr_fen_i = perm[n_test:]               # fen'd train positions (for train-acc eval)
    rest_i = list(range(n_fen, n))         # non-fen'd, train-only

    tr_i = tr_fen_i + rest_i               # fen'd FIRST so X_tr[:eval_n] all have fens
    fens_tr = [fens[i] for i in tr_fen_i] + [None] * len(rest_i)

    train = (X[tr_i], fl[tr_i], tl[tr_i], fens_tr)
    test = (X[te_i], fl[te_i], tl[te_i], [fens[i] for i in te_i])
    return train, test


if __name__ == "__main__":
    games_to_policy_dataset(n_positions=1_000_000, skip_opening=3,
                            max_per_game=80, eval_cap=20000)