---
name: quickpod
description: Rent and drive GPU/CPU compute pods on QuickPod (quickpod.org) without the dsh-quickpod plugin — search offers, deploy pods from templates, SSH in, monitor, control lifecycle, and clean up. Includes the DiffusionHive project handoff (model, data, results, next steps).
whenToUse: Use for any task that needs rented GPU/CPU compute from QuickPod, or to resume the DiffusionHive project (find pods, reconnect via SSH, run jobs, destroy when done). Also use when a prior session used the quickpod_* tools but this session has no plugin.
metadata:
  account: "Invest With Friends — investwfriends@gmail.com — QuickPod user id 44024"
---

# QuickPod GPU/CPU compute — plugin-free usage guide

This skill is a drop-in replacement for the **dsh-quickpod** DeepSeek Harness
plugin (source lives in this repo under `dsh-quickpod/`). If your session has
the `quickpod_*` tools, prefer them — they wrap exactly the API below. If the
plugin is absent, drive QuickPod directly with HTTP (curl / python requests)
using this document.

## 1. Authentication

- API key: `qpk_...` from the QuickPod console (Settings → API Keys).
- Set `QUICKPOD_API_KEY` (recommended) or send two headers on every request:
  - `X-API-Key: <key>`
  - `Authorization: ApiKey <key>`
- Base URL: `https://api.quickpod.org`
- Verify credentials: `GET /update/auth/me` → `{"user": {...}}` (this account's user id is 44024). A 401 means a bad/expired key.

## 2. Core REST endpoints

| Purpose | Method + path | Key params |
|---|---|---|
| Search rentable GPUs | `GET /rentable` | `kind=gpu`, `gpu_type` (e.g. "RTX 4090"), `location`, `min_hourly`, `max_hourly`, `min_count`, `min_reliability`, `verified_only`, `limit`, `sort=hourly_cost`, `desc` |
| Search rentable CPUs | `GET /rentable_cpu` | same filters |
| Deploy GPU pod | `POST /update/createpod` | `template_uuid`, `offers_id`, `disk_size_gb`, `name`, `coupon_code` |
| Deploy CPU pod | `POST /update/createpod_cpu` | same |
| List my pods | `GET /mypods` (GPU), `GET /mypods_cpu` (CPU) | — |
| Pod status (full detail incl. SSH key) | `GET /mypods` then find by uuid | pod uuid |
| Pod logs (training output) | `GET /update/podlogs?pod_uuid=<uuid>` | — |
| Control pod | `GET /update/startpod|stoppod|restartpod|destroypod?pod_uuid=<uuid>` | action in path |
| Public templates | `GET /public_templates` | — |

The quickpod_* plugin tools map 1:1: search→/rentable, templates→/public_templates,
deploy→/update/createpod, pods→/mypods, status→/mypods lookup, logs→/update/podlogs,
control→/update/destroypod etc., wait→poll /mypods until target state.

## 3. Choosing a template and an offer

- **Base templates (public)**: "Jupyter Lab CUDA 12.6" (uuid
  `c5ba2322-8786-42f8-929f-5404df5ea8d8`, image
  `quickpod/jupyterlab:4.3.1-py3.10.12-cuda12.6-cudnn9.5-ubuntu22.04`) — light
  (~10 GB image, pulls fast), Python 3.10, CUDA 12.6, **torch not included**
  (pip install it). "Pytorch Latest" (`pytorch/pytorch:latest`) has torch
  preinstalled but is larger. "Fast AnimeV3" is generic PyTorch CUDA but very
  large/slow to pull — avoid unless needed.
- **Cheap verified US offers seen (2026-08)**: RTX 3070 $0.08/hr, RTX 4070
  $0.09/hr, RTX 4070 SUPER $0.09/hr, RTX A4000 $0.10/hr, RTX 3090 $0.18/hr.
- Filter `min_reliability >= 80` and sort by `hourly_cost`; request
  `disk_size_gb` 40-80 (torch + repo + data). **Never exceed $0.20/hr** — this
  is a hard project rule.
- Deploy response contains: `pod_uuid`, `public_ipaddress`,
  `open_port_start`..`open_port_end`, `ssh_private_key`, `ssh_public_keys`,
  `hourly_cost`.

## 4. SSH access (critical quirks learned the hard way)

- SSH user is the **pod UUID** (NOT root): `ssh -p <open_port_start> <pod_uuid>@<ip>`.
- Port = `open_port_start` (e.g. 40800, 15147).
- Key: save `pod.ssh_private_key` from status to a file with mode 0600; its
  public half is in `ssh_public_keys` (comment `dsh-training`).
- **Host quirk (learned 2026-08):** on some hosts (e.g. 96.28.88.208) the
  container sshd can start rejecting even registered keys after ~15 min
  ("Permission denied (publickey)") while the status still shows the key as
  authorized. Recovery: destroy the pod and redeploy on a different host/offer.
  The account's always-registered key is the `dsh-training` private key at
  `~/.ssh_qp` (its public key is pre-registered in `ssh_public_keys`) — try it
  first: `ssh -i ~/.ssh_qp -p <port> <uuid>@<ip>`.
- **Never hold an SSH session open for long jobs.** Launch with
  `nohup cmd > log 2>&1 < /dev/null &` so SSH returns immediately; some hosts
  block additional SSH sessions while one is active.
- Fresh pod timing: large images pull in 10-20+ min. Poll status until
  `State=running` and `command_successful=true` before SSHing.

## 5. Pod bring-up recipe (proven for this project)

```bash
# after SSH works:
pip install --no-cache-dir torch numpy gymnasium tqdm   # torch ~2GB download
# upload files binary-safe (no scp needed):
cat localfile | ssh -i ~/.ssh_qp -p <port> <uuid>@<ip> 'cat > /workspace/localfile'
# download files binary-safe:
ssh -i ~/.ssh_qp -p <port> <uuid>@<ip> 'cat /workspace/remotefile' > localfile
# run a long job detached:
ssh ... 'cd /workspace; export PYTHONPATH=/workspace:/workspace/Mzinga/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; nohup python3 script.py > run.log 2>&1 < /dev/null &'
# GPU memory: use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; on OOM reduce
# batch size or pass move_chunk_size to score_legal_moves.
# The Mzinga C# teacher engine needs: export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
```

## 6. Project handoff: DiffusionHive (where we left off)

Repo: this repository (DiffusionHive). Full story in `paper/paper.md` and the
figures in `paper/figures/`. Data files are attached to the GitHub release
(v1.0.0): `dataset_plan.pt`, `dataset_vsr.pt`, `sp_data.pt`.

- **Game**: Hive (win by fully surrounding the enemy queen). Teacher: native
  Mzinga C# engine (negamax depth 4, driven over UHP) — binary in
  `mzinga_uhp/` (Linux binary auto-downloaded by `setup_cloud.sh`).
- **Model**: `ghive_diffusion_lite.HiveLiteModel` — 3,120,540-param
  block-diffusion transformer (hidden 128, 6 layers, dense FFN, policy MLP head,
  value head, aux heads). Best checkpoint: `runs/plan_run/final_model.pt`.
- **Data** (release assets): `dataset_plan.pt` (44,410 teacher-vs-random plan
  samples), `dataset_vsr.pt` (28,783 vs-random samples), `sp_data.pt`
  (2,402 self-play MCTS visit-distribution samples).
- **Results vs random** (200-ply cap, swap sides): model ~11-17.5% win /
  ~0-5% loss / rest draws; random 0% win / 100% draw; **teacher 37.5% win /
  62.5% draw**. The model **beats random** (positive win rate, ~zero losses).
- **Bottleneck**: policy head (15.7% top-1 vs teacher); value head is strong
  (95.1% ranking accuracy, separation 1.01). The surround is a ~134-ply chase
  that even the strong teacher only converts 37.5% of the time.
- **Negative results (8 interventions)**: scaling width/depth, MLP policy head,
  value lookahead, plan-conditioned diffusion, MCTS (4-16 sims),
  diffusion-conditioned decoding, policy-only fine-tune, and full self-play
  policy iteration (soft visit-distribution targets, 2,402 samples → regressed
  fast3 to ~10% vs ~12.5-17.5% baseline). All plateau at 12-17% win.
- **Next steps (untried/uncertain)**: (a) outcome-driven policy RL (policy
  gradient with the strong value head as critic), or (b) a batched/compiled MCTS
  (C++ or vectorized) with hundreds of simulations — the current Python MCTS is
  CPU-bound and pathologically slow above ~16 sims (~12 min/game at 64 sims).
- **Eval harness**: `PYTHONPATH=/workspace:/workspace/Mzinga/src python3
  gpu_eval3.py --ckpt <model.pt> --player {fast3,value,mcts} --games N`
  (also `ghive_diffusion.eval.runner.run_eval`).

## 7. Cleanup

Billing continues while a pod runs — **always destroy pods when done**
(`GET /update/destroypod?pod_uuid=<uuid>` or quickpod_control destroy). Save
artifacts locally first (the `cat | ssh` download pattern above is binary-safe).
