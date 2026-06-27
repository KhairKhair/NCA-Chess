import torch
import chess

# Material values. Color is encoded by sign (white +, black -),
# so a single channel carries both piece identity and side.
PIECE_VALUES = {
    chess.PAWN:   1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK:   5.0,
    chess.QUEEN:  9.0,
    chess.KING:   100.0, # no material value — sentinel so the king square isn't "empty"
}

def board_to_tensor(board):
    t = torch.zeros(1, 8, 8, dtype=torch.float32)
    for sq, piece in board.piece_map().items():
        r, c = chess.square_rank(sq), chess.square_file(sq)
        val = PIECE_VALUES[piece.piece_type]
        t[0, r, c] = val if piece.color == chess.WHITE else -val
    return t


def canonical_tensor_as(board, perspective):
    x = board_to_tensor(board)
    if perspective == chess.WHITE:
        return x
    x = torch.flip(x, dims=[1])
    return -x

def canonical_tensor(board):
    return canonical_tensor_as(board, board.turn)