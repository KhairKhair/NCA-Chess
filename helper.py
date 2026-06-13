
import torch
import chess

def board_to_tensor(board):
    # From chess.Board to 12x8x8 tensor: 6 piece types * 2 colors, with binary occupancy
    t = torch.zeros(12, 8, 8, dtype=torch.float32)
    for sq, piece in board.piece_map().items():
        r, c = chess.square_rank(sq), chess.square_file(sq)
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        t[plane, r, c] = 1.0
    return t
 
def canonical_tensor(board):
    # Flips board to mover's perspective: white pieces in planes 0-5, black pieces in 6-11
    # Model can learn from white's perspective only, and black is just mirrored white
    x = board_to_tensor(board)
    if board.turn == chess.WHITE:
        return x
    x = torch.flip(x, dims=[1])          
    return torch.cat([x[6:], x[:6]], dim=0) 
 
def sq_to_canonical(sq, turn):
    # Maps chess square index (0-63) to canonical 0-63 from mover's perspective, with file flip for black 
    r, c = chess.square_rank(sq), chess.square_file(sq)
    if turn == chess.BLACK:
        r = 7 - r
    return r * 8 + c