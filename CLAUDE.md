# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent Reinforcement Learning (MARL) air combat simulation for a Master's thesis. Two Blue agents (MAPPO-trained) fight two Red agents (heuristic baseline) in a 3D dogfighting environment with realistic 6-DoF F-16 aircraft dynamics. The project follows a 5-phase curriculum learning plan; phases 0–1 (baseline MAPPO) are currently implemented.

## Commands

### Training
```bash
python training/train_mappo.py
python training/train_mappo.py --config configs/config.yaml --seed 42 --device cuda
python training/train_mappo.py --resume checkpoints/mappo_ep5000.pt
```
Outputs: `checkpoints/mappo_ep{N}.pt`, `checkpoints/mappo_final.pt`, `logs/train_log.csv`, `logs/train_log.json`

### Evaluation
```bash
python evaluation/eval.py --checkpoint checkpoints/mappo_final.pt
python evaluation/eval.py --checkpoint checkpoints/mappo_ep1000.pt --episodes 50 --deterministic
```
Outputs: console summary + `logs/eval_results.json`

### Tests
```bash
# Her test dosyası bağımsız çalışır — proje kökünden:
python tests/test_aircraft_model.py
python tests/test_env_step.py
python tests/test_train_mappo.py
python tests/test_opponent_pool.py
# ... (11 test dosyası toplam, her modül için bir tane)
```
**Önemli:** Windows'ta Unicode sorunu çıkarsa `-X utf8` ekle: `python -X utf8 tests/test_*.py`

### Dependencies
```bash
pip install torch numpy pyyaml scipy
```
No `requirements.txt` exists; these four packages cover all imports.

## Architecture

### Layered Structure

```
configs/config.yaml        ← single source of truth for all hyperparameters
envs/                      ← physics + environment
  aircraft_model.py        ← 6-DoF RK4 dynamics, state[18] + action[5]
  dogfight_env.py          ← Gym-like multi-agent loop (reset/step/obs/reward)
  reward_model.py          ← 8-term weighted reward with role/aggression system
  weapons_model.py         ← WEZ cone hit detection + damage
  geometry_utils.py        ← pure-numpy ENU coordinate helpers
  trim_solver.py           ← equilibrium solver for stable episode spawns
agents/
  heuristic_agent.py       ← rule-based opponent (CRITICAL → EVASION → PURSUIT)
  opponent_pool.py         ← fictitious self-play: checkpoint ring buffer, Red policy
training/
  train_mappo.py           ← MAPPO trainer: Actor/Critic MLPs, GAE, PPO update
  rollout_buffer.py        ← n_steps trajectory storage + GAE computation
utils/
  normalization.py         ← state → obs normalization (49D for 2v2 phases 0–1)
evaluation/
  eval.py                  ← checkpoint evaluation vs heuristic baselines
tests/                     ← 11 test dosyası, her modül için ayrı (616 test toplam)
```

### Data Flow per Environment Step

1. `Normalizer` converts raw aircraft state (18D) → observation vector (49D = 16 ego + 9 teammate + 12×2 enemies)
2. `MAPPOActor` maps observation → 5D action (aileron, elevator, rudder, throttle, fire)
3. `DogfightEnv.step()` applies actions through `AircraftModel` (RK4 physics, dt=0.05s)
4. `WeaponsModel` checks WEZ cone (±30°, 300–8000m), applies damage on hit
5. `RewardModel` computes 8-term reward scaled by agent role/aggression
6. `RolloutBuffer` stores (obs, action, log_prob, value, reward, done)

### MAPPO Training Loop

- Collect `n_steps=512` steps → compute GAE (γ=0.99, λ=0.95) → PPO minibatch updates
- Red opponent: **`OpponentPool`** (fictitious self-play) — heuristic fallback pool boşken
- Opponent pipeline: random (0–500k steps) → `OpponentPool` (500k+ steps)
- Pool her `pool_update_interval=200` episode'da bir actor snapshot alır (`checkpoints/pool_actor_ep{N}.pt`)
- Phase 1→2 early stopping: win_rate ≥ 40%, kills/episode ≥ 0.8, OOB rate ≤ 5%

### Active Hyperparameters

| Parametre | Değer |
|-----------|-------|
| `entropy_coeff` | 0.05 |
| `n_steps` | 512 |
| `w_survival` | 0.01 |
| `w_kill` | 15 |
| `w_wez` | 5 |

### OpponentPool — Fictitious Self-Play

`agents/opponent_pool.py` — `OpponentPool` sınıfı:
- Ring buffer (max 20 checkpoint); `add_checkpoint(path)` ile doldurulur
- `reset()`: episode başında pool'dan rastgele checkpoint seçer, actor yükler; pool boşsa heuristic fallback
- `act(obs_dict, state_dict)`: neural policy veya fallback ile Red aksiyonları üretir
- Red obs `DogfightEnv._build_obs_dict()` tarafından Red perspektifinden üretilir — ayrıca işlem gerekmez
- `MAPPOTrainer._save_pool_snapshot()`: actor-only checkpoint kaydeder (optimizer olmadan)

**Konfigürasyon** (`configs/config.yaml` → `opponent_pool` bölümü):
```yaml
opponent_pool:
  max_pool_size:         20
  start_step:       500_000
  pool_update_interval: 200
```

## Kritik Sorun: Policy Collapse (İzleniyor)

**Gözlemlenen pattern (3 denemede tekrarlandı):** Red heuristic tam güce geçince (1M adım sonrası) Blue win rate sıfıra düşüyor.

**Kök neden:** Exploration collapse + MAPPO hiç kill alamıyor. Sabit deterministic rakibe maruz kalan Blue policy dar sub-optimal davranışa sıkışıyor.

**Uygulanan çözüm:** Fictitious self-play — `OpponentPool` ile Red, Blue'nun kendi geçmiş checkpoint'lerine karşı oynuyor. Sabit heuristic'e kilitlenme ortadan kalkıyor.

**Mevcut durum (2026-04-12):** Faz 1 eğitimi devam ediyor, ~2.34M adım (ep 600). Reward dalgalı ama çöküş yok. Öncelik: collapse izleme → stabil reward artışı → fine-tune.

**Collapse kriterleri (otomatik durdurma):** 2 ardışık log'da reward < -3000 VE win rate = 0.00.

### Bilinen Sorunlar
- `pool_start_step` (config: 500k) kodda kullanılmıyor — pool ep 200'den itibaren aktif (erken, ama sorunsuz çalışıyor)
- OOB (sınır dışı) şu an LOSE sayılmıyor — düşük öncelik, collapse sonrası ele alınacak
- `train_mappo.py` başına `sys.stdout` line-buffered açma eklendi (flush sorunu çözüldü)

### Reward Role System

`RewardModel` supports an `aggression` float ∈ [0.0, 1.0]:
- `0.0` = defensive (high survival weight, low kill weight)
- `1.0` = aggressive (high kill weight, low survival weight)
- Phases 0–1: aggression=None (all scales=1.0); phases 2+: per-agent role assignment

### Key Configuration (configs/config.yaml)

All physics constants, reward weights, training hyperparameters, and curriculum thresholds are centralized here. Modifying behavior almost always means editing this file rather than source code. Notable sections: `aircraft`, `aero_coeffs`, `weapons`, `reward`, `roles`, `training`, `curriculum`.

### Observation Dimensions

- Phases 0–1 (2v2, no GAT): 49D = 16 (ego) + 9 (teammate) + 12×2 (enemies)
- Phases 2+ (planned): add 2D role embedding + 16D GAT message per teammate

### Code Language

Docstrings and inline comments are written in Turkish (Türkçe).
