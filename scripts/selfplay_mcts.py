"""Self-play policy iteration (AlphaZero-style): generate visit-distribution
targets by playing MCTS against a random opponent, and store the search's
soft policy (pi) at every model position.

The model's strong value head (95% ranking acc) guides the MCTS; the visit
distribution pi encodes the search's preference, which is a much richer and
more outcome-aware target than the teacher's single move. These (position, pi)
pairs are used to retrain the policy head.
"""
import argparse, random, time, torch
import numpy as np
from mzinga.core.board import Board
from mzinga.core.enums import GameType, PlayerColor
from ghive_diffusion.context_builder import HiveContextBuilder
from ghive_diffusion.inference import MCTSPlayer
from ghive_diffusion.tokenizer import build_default_tokenizer
from ghive_diffusion.eval.runner import RandomPlayer
from ghive_diffusion_lite import build_lite_model

def _play(board, mv):
    """Robustly play mv; returns True on success."""
    try:
        ms = board.get_move_string(mv)
    except Exception:
        ms = board.try_get_move_string(mv)
        if ms is None:
            return False
    board.trusted_play(mv, ms)
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--sims", type=int, default=16)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=2)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_lite_model()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(dev).eval()

    tk = build_default_tokenizer(model.cfg)
    builder = HiveContextBuilder(tk)
    player = MCTSPlayer(model, tk, num_simulations=a.sims, progressive_temperature=False)
    rnd = RandomPlayer()

    samples = []
    t0 = time.time()
    wins = 0
    for gi in range(a.games):
        board = Board(GameType.Base)
        model_is_white = (gi % 2 == 0)
        for _ply in range(a.max_plies):
            if board.game_is_over:
                break
            is_model_turn = (board.current_color == PlayerColor.White) == model_is_white
            if not is_model_turn:
                mv = rnd.play(board)
                if not _play(board, mv):
                    break
                continue
            # model turn: run MCTS and collect the root visit distribution
            move = player.search(board)
            children = player._tree.get(board.zobrist_key, {})
            legal_moves = list(board.get_valid_moves())
            legal_strs = builder._legal_moves(board)
            str_to_idx = {s: i for i, s in enumerate(legal_strs)}
            total_N = sum(c[0] for c in children.values()) or 1.0
            pi = np.zeros(len(legal_strs), dtype=np.float32)
            for act, (N, _W, _P) in children.items():
                if act < len(legal_moves):
                    try:
                        s = board.get_move_string(legal_moves[act])
                    except Exception:
                        continue
                    if s in str_to_idx:
                        pi[str_to_idx[s]] = N / total_N
            ssum = float(pi.sum())
            if ssum > 0:
                pi /= ssum
            else:
                pi[:] = 1.0 / len(pi)
            ctx = builder.encode(board, target_move=None)
            legal_ids = [tk.encode_move(s) for s in legal_strs]
            samples.append({
                "context_ids": torch.tensor(ctx, dtype=torch.long),
                "legal_ids": legal_ids,
                "pi": torch.tensor(pi, dtype=torch.float32),
            })
            if not _play(board, move):
                break
        st = board.board_state
        if board.game_is_over and st.name in ("WhiteWins", "BlackWins"):
            wins += 1
        if gi % a.log_every == 0 or gi == a.games - 1:
            dt = time.time() - t0
            print(f"game {gi+1}/{a.games} samples={len(samples)} wins={wins} {dt:.0f}s "
                  f"({dt/max(1,gi+1):.1f}s/game)", flush=True)

    torch.save(samples, a.out)
    print(f"DONE {len(samples)} samples -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
