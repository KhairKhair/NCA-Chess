import torch
import chess

NUM_PIECE_PLANES = 12        
EMPTY_PLANE = 12
VISIBLE_CHANNELS = 13       
_PIECE_LETTERS = ["P", "N", "B", "R", "Q", "K"]   


def board_to_tensor(board):
    t = torch.zeros(VISIBLE_CHANNELS, 8, 8, dtype=torch.float32)
    t[EMPTY_PLANE] = 1.0   
    for sq, piece in board.piece_map().items():
        r, c = chess.square_rank(sq), chess.square_file(sq)
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        t[plane, r, c] = 1.0
        t[EMPTY_PLANE, r, c] = 0.0
    return t


def canonical_tensor_as(board, perspective):
    x = board_to_tensor(board)
    if perspective == chess.WHITE:
        return x
    x = torch.flip(x, dims=[1])        
    white, black, empty = x[0:6], x[6:12], x[12:13]
    return torch.cat([black, white, empty], dim=0)   


def canonical_tensor(board):
    return canonical_tensor_as(board, board.turn)


def sq_to_canonical(sq, turn):
    r, c = chess.square_rank(sq), chess.square_file(sq)
    if turn == chess.BLACK:
        r = 7 - r
    return r * 8 + c


def decode_planes(planes, min_conf=0.3, min_margin=0.1):
    p = planes[:VISIBLE_CHANNELS]
    top2 = p.topk(2, dim=0)
    best, runner = top2.values[0], top2.values[1]
    idx = top2.indices[0]
    margin = best - runner

    out = [["" for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for c in range(8):
            k = idx[r, c].item()
            if best[r, c] < min_conf or margin[r, c] < min_margin:
                out[r][c] = "?"
            elif k == EMPTY_PLANE:
                out[r][c] = ""
            else:
                letter = _PIECE_LETTERS[k % 6]
                out[r][c] = letter if k < 6 else letter.lower()
    return out