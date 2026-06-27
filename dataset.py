import chess
import torch
from datasets import load_dataset
from helper import canonical_tensor, sq_to_canonical

def games_to_policy_dataset(n_positions=20000, skip_opening=6, max_per_game=12,
                            cache="humanpolicy.pt"):
    
    # load dataset from https://github.com/angeluriot/Chess_games
    ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)
    
    # Xs is board state tensor, FL/TL are from/to move labels, fens is for eval/debug
    Xs, FL, TL, fens = [], [], [], []
    for game in ds:
        ucis = game.get("moves_uci") or []
        if not ucis: continue
        board = chess.Board(); taken = 0
        for i, uci in enumerate(ucis):
            try: move = chess.Move.from_uci(uci)
            except ValueError: break
            if move not in board.legal_moves: break
            if i >= skip_opening and taken < max_per_game:
                # skip first few moves (opening book) and limit positions per game to increase diversity
                turn = board.turn
                Xs.append(canonical_tensor(board))
                FL.append(sq_to_canonical(move.from_square, turn))
                TL.append(sq_to_canonical(move.to_square, turn))
                fens.append(board.fen()); taken += 1
            board.push(move)
            if len(Xs) >= n_positions: break
        if len(Xs) >= n_positions: break
    X = torch.stack(Xs)
    fl = torch.tensor(FL, dtype=torch.long)
    tl = torch.tensor(TL, dtype=torch.long)
    torch.save({"X": X, "fl": fl, "tl": tl, "fens": fens}, cache)
    print(f"built {X.shape[0]} positions -> {cache}")
    return X, fl, tl, fens
 
def load_split(cache="humanpolicy.pt", test_frac=0.1, seed=0):
    # builds train/test split
    d = torch.load(cache)
    X, fl, tl, fens = d["X"], d["fl"], d["tl"], d["fens"]
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_test = int(n * test_frac)
    te_i, tr_i = perm[:n_test], perm[n_test:]
    X_tr, fl_tr, tl_tr = X[tr_i], fl[tr_i], tl[tr_i]
    fens_tr = [fens[i] for i in tr_i]               
    train = (X_tr, fl_tr, tl_tr, fens_tr)
    test  = (X[te_i], fl[te_i], tl[te_i], [fens[i] for i in te_i])
    return train, test
 
if __name__ == "__main__":
    games_to_policy_dataset(n_positions=5000, skip_opening=6, max_per_game=30)