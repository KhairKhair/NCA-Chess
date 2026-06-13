# server.py
import json

import chess
import numpy as np
import torch
from flask import Flask, jsonify, request, send_file

from model import PolicyChessCA
from helper import canonical_tensor, sq_to_canonical

RUN_JSON = "results_policy-mid/run_000.json"
WEIGHTS_PT = "results_policy-mid/model_000.pt"

# Load model config and weights
with open(RUN_JSON, "r", encoding="utf-8") as f:
    cfg = json.load(f)["cfg"]
model = PolicyChessCA(
    channels=cfg["channels"],
    steps=cfg["steps"],
    hidden_act=cfg["hidden_act"],
    bound_act=cfg["bound_act"],
    fire_rate=cfg.get("fire_rate", 1.0),
)
model.load_state_dict(torch.load(WEIGHTS_PT, map_location="cpu"))
model.eval()

# Force deterministic, full updates at inference
model.fire_rate = 1.0
print(f"loaded model: {cfg}")
app = Flask(__name__)


def encode(board: chess.Board) -> torch.Tensor:
    # Encode a chess.Board into the model's canonical tensor format, with batch dim.
    return canonical_tensor(board).unsqueeze(0)


def canonical_rank_to_absolute(rank: int, was_black: bool) -> int:
    # Convert canonical rank (0-7 from mover's perspective) to absolute rank (0-7 from white's perspective).
    return 7 - rank if was_black else rank


def canon_idx_to_square(idx: int, was_black: bool) -> int:
    rank_canon, file_ = divmod(idx, 8)
    rank_abs = canonical_rank_to_absolute(rank_canon, was_black)
    return chess.square(file_, rank_abs)

GAMMA = 0.6
def to_frames(maps, was_black):
    gmax = max(1e-9, max(float(f.max()) for f in maps))
    out = []
    for frame in maps:
        norm = (frame / gmax) ** GAMMA
        if was_black:
            norm = norm[::-1, :]
        out.append(norm.round(3).tolist())
    return out


@torch.no_grad()
def rollout(board: chess.Board):
    x = encode(board)
    was_black = board.turn == chess.BLACK
    s = model._init_state(x)

    # per-step decision readout (canonical frame). apply the policy head to the
    # intermediate state at EVERY step and softmax each head, then take the
    # per-square max of the from- and to-probabilities. 

    decision_canon = []                   
    from_canon, to_canon = [], []        
    from_logits = to_logits = None
    for _ in range(model.steps):
        s = model._step(s, x)
        logits_t = model.head(s)[0]               
        from_logits = logits_t[0].reshape(64)
        to_logits = logits_t[1].reshape(64)
        fp = torch.softmax(from_logits, dim=0).reshape(8, 8).cpu().numpy()
        tp = torch.softmax(to_logits, dim=0).reshape(8, 8).cpu().numpy()
        decision_canon.append(np.maximum(fp, tp))
        from_canon.append(fp)
        to_canon.append(tp)

    raw_from_c = int(torch.argmax(from_logits))
    raw_to_c = int(torch.argmax(to_logits))

    raw = chess.Move(
        canon_idx_to_square(raw_from_c, was_black),
        canon_idx_to_square(raw_to_c, was_black),
    )
    raw_queen = chess.Move(raw.from_square, raw.to_square, promotion=chess.QUEEN)

    # use the raw argmax move if legal, else project onto legal moves by
    # scoring each legal (f, t) pair as from_logit[f] + to_logit[t].
    if raw in board.legal_moves:
        chosen = raw
    elif raw_queen in board.legal_moves:
        chosen = raw_queen
    else:
        chosen, best_score = None, -float("inf")
        for move in board.legal_moves:
            f = sq_to_canonical(move.from_square, board.turn)
            t = sq_to_canonical(move.to_square, board.turn)
            score = from_logits[f].item() + to_logits[t].item()
            if score > best_score:
                best_score, chosen = score, move

    activity_frames_abs = to_frames(decision_canon, was_black)
    from_frames_abs = to_frames(from_canon, was_black)
    to_frames_abs = to_frames(to_canon, was_black)

    # from/to probability heatmaps (softmax), absolute coords, for the UI.
    from_prob = torch.softmax(from_logits, dim=0).reshape(8, 8).cpu().numpy()
    to_prob = torch.softmax(to_logits, dim=0).reshape(8, 8).cpu().numpy()
    if was_black:
        from_prob = from_prob[::-1, :]
        to_prob = to_prob[::-1, :]
    from_heat = from_prob.round(4).tolist()
    to_heat = to_prob.round(4).tolist()

    chosen_uci = chosen.uci() if chosen is not None else None
    raw_uci = chess.square_name(raw.from_square) + chess.square_name(raw.to_square)

    return {
        "move": chosen_uci,
        "raw": raw_uci,
        "frames": activity_frames_abs,
        "from_heat": from_heat,
        "to_heat": to_heat,
        "from_frames": from_frames_abs,
        "to_frames": to_frames_abs,
    }


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    fen = payload.get("fen")

    if not fen:
        return jsonify({"error": "Missing FEN."}), 400

    try:
        board = chess.Board(fen)
    except ValueError:
        return jsonify({"error": "Invalid FEN."}), 400

    if board.is_game_over():
        return jsonify(
            {
                "game_over": True,
                "result": board.result(),
                "move": None,
            }
        )

    out = rollout(board)

    if out["move"] is None:
        return jsonify({"error": "No legal move could be selected."}), 500

    move = chess.Move.from_uci(out["move"])

    # SAN must be generated before pushing the move.
    move_san = board.san(move)
    board.push(move)

    return jsonify(
        {
            **out,
            "move_san": move_san,
            "fen_after": board.fen(),
            "game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
        }
    )


@app.route("/legal", methods=["POST"])
def legal():
    """Validate a human move and return the resulting FEN."""
    payload = request.get_json(silent=True) or {}

    try:
        board = chess.Board(payload["fen"])
        move = chess.Move.from_uci(payload["uci"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False})

    if move not in board.legal_moves:
        move_queen = chess.Move(
            move.from_square,
            move.to_square,
            promotion=chess.QUEEN,
        )
        if move_queen in board.legal_moves:
            move = move_queen
        else:
            return jsonify({"ok": False})

    board.push(move)

    return jsonify(
        {
            "ok": True,
            "fen": board.fen(),
            "game_over": board.is_game_over(),
            "result": board.result() if board.is_game_over() else None,
        }
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
