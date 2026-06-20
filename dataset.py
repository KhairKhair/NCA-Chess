import chess
import torch
from datasets import load_dataset
from helper import canonical_tensor


def games_to_policy_dataset(
    n_positions=20000,
    skip_opening=6,
    max_per_game=12,
    cache="humanpolicy.pt",
):
    # load dataset from https://github.com/angeluriot/Chess_games
    ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)
    # Xs = board before move
    # Ys = board after move
    # fens = before-move FENs for eval/debug
    Xs, Ys, fens = [], [], []

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
                # board before move
                Xs.append(canonical_tensor(board))
                fens.append(board.fen())
                # make move, then save board after move
                board.push(move)
                Ys.append(canonical_tensor(board))
                taken += 1
            else:
                board.push(move)

            if len(Xs) >= n_positions:
                break

        if len(Xs) >= n_positions:
            break

    X = torch.stack(Xs)  
    Y = torch.stack(Ys)
    torch.save(
        {
            "X": X,
            "Y": Y,
            "fens": fens,
        },
        cache,
    )
    print(f"built {X.shape[0]} positions -> {cache}")
    return X, Y, fens


def load_split(cache="humanpolicy.pt", test_frac=0.1, seed=0):
    # builds train/test split
    d = torch.load(cache)
    X, Y, fens = d["X"], d["Y"], d["fens"]
    n = X.shape[0]

    perm = torch.randperm(
        n,
        generator=torch.Generator().manual_seed(seed),
    ).tolist()

    n_test = int(n * test_frac)
    te_i, tr_i = perm[:n_test], perm[n_test:]

    train = (
        X[tr_i],
        Y[tr_i],
        [fens[i] for i in tr_i],
    )
    test = (
        X[te_i],
        Y[te_i],
        [fens[i] for i in te_i],
    )
    return train, test


if __name__ == "__main__":
    games_to_policy_dataset(
        n_positions=1000,
        skip_opening=6,
        max_per_game=1,
    )