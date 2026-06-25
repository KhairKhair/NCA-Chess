import chess
import torch
from datasets import load_dataset
from helper import canonical_tensor_as
import chess.engine as engine
from dm_utils import (
    tokenize,
    centipawns_to_win_probability,
    get_uniform_buckets_edges_values,
    compute_return_buckets_from_returns,
)
import numpy as np

NUM_RETURN_BUCKETS = 30
_BUCKET_EDGES, _ = get_uniform_buckets_edges_values(NUM_RETURN_BUCKETS)


def games_to_policy_dataset(
    engine,
    n_positions=20000,
    skip_opening=6,
    max_per_game=12,
    cache="humanpolicy.pt",
):
    # load dataset from https://github.com/angeluriot/Chess_games
    ds = load_dataset("angeluriot/chess_games", split="train", streaming=True)
    # Xs = tokenized FEN, Ys = return bucket integer, fens = raw FEN strings
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
                mover = board.turn          
                fens.append(board.fen())
                result = engine.analyse(board, chess.engine.Limit(time=0.05))["score"]
                centipawns = result.relative.score(mate_score=10000)
                win_prob = centipawns_to_win_probability(centipawns)
                bucket = int(compute_return_buckets_from_returns(
                    np.asarray([win_prob]), _BUCKET_EDGES
                )[0])
                Xs.append(canonical_tensor_as(board, mover))
                Ys.append(bucket)
                fens.append(board.fen())
                board.push(move)
                taken += 1
            else:
                board.push(move)

            if len(Xs) >= n_positions:
                break

        if len(Xs) >= n_positions:
            break

    X = torch.stack(Xs)
    Y = torch.tensor(Ys, dtype=torch.long)
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
    sf = engine.SimpleEngine.popen_uci("./stockfish")
    n_positions = 400
    games_to_policy_dataset(
        engine=sf,
        n_positions=n_positions,
        skip_opening=6,
        max_per_game=1,cache=f"humanpolicy_{n_positions}.pt",
    )
