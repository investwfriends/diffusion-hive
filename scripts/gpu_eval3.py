import argparse, torch
from ghive_diffusion_lite import build_lite_model
from ghive_diffusion.tokenizer import build_default_tokenizer
from ghive_diffusion.inference import FastPlayer, ValuePlayer, MCTSPlayer
from ghive_diffusion.eval.runner import RandomPlayer, run_eval, EvalConfig, FastPlayerAdapter, ValuePlayerAdapter, MCTSPlayerAdapter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--player", default="mcts", choices=["fast0","fast3","value","mcts"])
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--sims", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_lite_model()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(dev).eval()
    tk = build_default_tokenizer(model.cfg)
    if a.player == "value":
        player = ValuePlayerAdapter(ValuePlayer(model, tk, deterministic=True))
    elif a.player == "mcts":
        player = MCTSPlayerAdapter(MCTSPlayer(model, tk, num_simulations=a.sims, progressive_temperature=False))
    else:
        lk = 0 if a.player == "fast0" else 3
        player = FastPlayerAdapter(FastPlayer(model, tk, deterministic=True, lookahead_k=lk, lookahead_weight=0.5, diffusion_candidates=False))
    res = run_eval(player, RandomPlayer(), EvalConfig(n_games=a.games, max_plies=a.max_plies, swap_sides=True, seed=a.seed))
    print(f"RESULT device={dev} player={a.player} games={res.n_games} wins={res.wins} losses={res.losses} draws={res.draws} win_rate={res.win_rate:.3f}")
    print(res.to_markdown())

if __name__ == "__main__":
    main()
