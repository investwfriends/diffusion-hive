"""Retrain the policy head on soft MCTS visit-distribution targets.

Freezes the encoder + value head (the strong critic), trains only the decoder
+ policy head to match the visit distribution pi via soft cross-entropy
(KL divergence to pi). This is the policy-improvement step of policy iteration.
"""
import argparse, random, time, torch, math
import torch.nn.functional as F
from ghive_diffusion_lite import build_lite_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", required=True)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = build_lite_model()
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(dev)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("decoder.") or name.startswith("policy_score_head.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(x.numel() for x in trainable)} / {sum(x.numel() for x in model.parameters())}", flush=True)

    data = torch.load(a.data, map_location="cpu", weights_only=False)
    n = len(data)
    print(f"data: {n} samples", flush=True)

    opt = torch.optim.AdamW(trainable, lr=a.lr)
    def lr_at(step):
        prog = min(1.0, step / max(1, a.steps))
        return a.lr * (0.5 + 0.5 * math.cos(math.pi * prog))

    model.train()
    t0 = time.time()
    loss_ema = None
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step - 1)
        idxs = random.sample(range(n), min(a.batch, n))
        opt.zero_grad()
        total = torch.zeros((), device=dev)
        agree = 0
        for i in idxs:
            d = data[i]
            ctx = d["context_ids"].unsqueeze(0).to(dev)
            legal = d["legal_ids"]
            pi = d["pi"].to(dev)
            with torch.no_grad():
                _, enc_kv = model.forward_encoder(ctx, use_cache=True)
            scores, _ = model.score_legal_moves(ctx, legal, use_value_head=False, encoder_kv=enc_kv, move_chunk_size=32)
            logp = F.log_softmax(scores, dim=-1)
            loss = -torch.sum(pi * logp)
            total = total + loss / len(idxs)
            if int(scores.argmax().item()) == int(pi.argmax().item()):
                agree += 1
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        opt.step()
        loss_ema = float(total.item()) if loss_ema is None else 0.9 * loss_ema + 0.1 * float(total.item())
        if step % a.log_every == 0 or step == a.steps:
            dt = time.time() - t0
            print(f"step {step}/{a.steps} kl={total.item():.4f} ema={loss_ema:.4f} "
                  f"top1_vs_pi={agree}/{len(idxs)} lr={lr_at(step-1):.1e} {dt:.0f}s", flush=True)
        if step % a.save_every == 0:
            torch.save({"model": model.state_dict(), "step": step}, a.out + f".step{step}.pt")
            print(f"  saved {a.out}.step{step}.pt", flush=True)

    torch.save({"model": model.state_dict(), "step": a.steps, "kl_ema": loss_ema}, a.out)
    print(f"DONE {a.out} kl_ema={loss_ema:.4f}", flush=True)

if __name__ == "__main__":
    main()
