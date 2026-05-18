# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent Reinforcement Learning (MARL) air combat simulation for a Master's thesis. Two Blue agents (MAPPO-trained) fight two Red agents (heuristic baseline) in a 3D dogfighting environment with realistic 6-DoF F-16 aircraft dynamics. The project uses a 4-phase curriculum learning plan.

**Mevcut durum (2026-05-12):** MAPPO GAT Faz-3 tamamlandı. Final MAPPO checkpoint: `mappo_gat_ep44000.pt`. QMIX implementasyonu tamamlandı (`models/qmix_net.py`, `training/train_qmix.py`), tam eğitim devam ediyor. MAPPO için doğrulanmış en iyi checkpoint: `mappo_gat_ep37000.pt` (heuristic eval: win=69.6%, kill/ep=1.51).

## Commands

### Training
**Kural: Her run ayrı log dosyasına yazmalı — asla aynı dosyaya append etme.**

```powershell
# Windows — terminalde görünür + log dosyasına yaz (PowerShell)
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/train_$timestamp.txt"
python -u -X utf8 training/train_mappo.py | Tee-Object -FilePath $log

# Checkpoint'ten devam
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/train_$timestamp.txt"
python -u -X utf8 training/train_mappo.py --resume checkpoints/mappo_ep11000.pt --start-phase 1 | Tee-Object -FilePath $log

# Faz-2: GAT iletişim, transfer learning (Faz-1 tamamlanmış checkpoint'ten)
python -u -X utf8 training/train_mappo.py --phase2 checkpoints/mappo_final.pt --freeze-steps 0 | Tee-Object -FilePath $log

# GAT Faz-3 (2v2) devam — mevcut GAT checkpoint'ten, direkt Faz-3'e (bash)
python -u -X utf8 training/train_mappo.py --phase2 checkpoints/mappo_gat_ep37000.pt --resume checkpoints/mappo_gat_ep37000.pt --start-phase 4 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).txt
```
**ÖNEMLİ:** `--phase2 X --resume X` kombinezonunu kullan (her ikisi aynı GAT checkpoint). `--phase2` tek başına Faz-1→GAT transferi içindir; GAT checkpoint'ten devam için her ikisi gerekli.

**CurriculumManager checkpoint'te state saklamıyor:** Resume edince her zaman Faz-1'den başlar. GAT eğitimini Faz-3'te devam ettirmek için `--start-phase 4` zorunlu.

**ÖNEMLİ:** Her zaman `-u -X utf8` kullan — `-u` stdout buffering'i devre dışı bırakır, `-X utf8` Türkçe karakterlerde UnicodeEncodeError'ı önler.

**Duplicate process kontrolü:** Her train başlatmadan önce:
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like '*train_mappo*' } | Select-Object ProcessId, CommandLine
# Temizlemek için:
Stop-Process -Id <PID> -Force
```
PyTorch her zaman 2 process açar (1 main venv + 1 idle system python worker) — normaldir.

**Config hot-reload yok:** `pool_update_interval`, `phase1_win_threshold`, `ammo_miss_penalty` ve tüm diğer değerler startup'ta bir kez okunur. Config değişikliği için restart gerekir.

Outputs: `checkpoints/mappo_ep{N}.pt`, `checkpoints/mappo_final.pt`, `logs/train_log.csv`

### QMIX Training
```bash
# Tam eğitim (bash)
python -u -X utf8 training/train_qmix.py 2>&1 | tee logs/train_qmix_$(date +%Y%m%d_%H%M%S).txt

# Test modu (10 episode)
python -u -X utf8 training/train_qmix.py --test

# Checkpoint'ten devam
python -u -X utf8 training/train_qmix.py --resume checkpoints/qmix_ep1000.pt 2>&1 | tee logs/train_qmix_$(date +%Y%m%d_%H%M%S).txt
```

Outputs: `checkpoints/qmix_ep{N}.pt`, `checkpoints/qmix_final.pt`, `logs/qmix_log.csv`

**Duplicate process kontrolü (QMIX):**
```powershell
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like '*train_qmix*' } | Select-Object ProcessId, CommandLine
```

### Evaluation
```bash
python -X utf8 evaluation/eval.py --checkpoint checkpoints/mappo_final.pt
python -X utf8 evaluation/eval.py --checkpoint checkpoints/mappo_ep1000.pt --episodes 50 --deterministic
```

### Tests
```bash
python -X utf8 tests/test_aircraft_model.py
python -X utf8 tests/test_env_step.py
python -X utf8 tests/test_train_mappo.py
python -X utf8 tests/test_opponent_pool.py
```

### Dependencies
```bash
pip install torch numpy pyyaml scipy
```

## Architecture

### Layered Structure

```
configs/config.yaml        ← single source of truth for all hyperparameters
envs/                      ← physics + environment
  aircraft_model.py        ← 6-DoF RK4 dynamics, state[18] + action[5]
  dogfight_env.py          ← Gym-like multi-agent loop; set_curriculum_phase() ile faz kontrolü
                              _critical_recovery_blue() — Blue OOB/zemin kurtarma override
  reward_model.py          ← 9-term weighted reward + r_fire_ready + r_pursuit + r_smooth
  weapons_model.py         ← WEZ cone hit detection + damage (cooldown=0.5s)
  geometry_utils.py        ← pure-numpy ENU coordinate helpers
  trim_solver.py           ← equilibrium solver for stable episode spawns
agents/
  heuristic_agent.py       ← rule-based opponent (CRITICAL → EVASION → PURSUIT)
  opponent_pool.py         ← adaptive fictitious self-play (win_rate-driven checkpoint selection)
models/
  qmix_net.py              ← ActionMapper (162 ayrık), AgentQNetwork (MLP), QMixNet (hypernetwork mixing)
training/
  train_mappo.py           ← MAPPO trainer + CurriculumManager
  train_qmix.py            ← QMIX trainer (off-policy, ε-greedy, ReplayBuffer 50k)
  rollout_buffer.py        ← n_steps trajectory storage + GAE computation
utils/
  normalization.py         ← state → obs normalization (50D)
evaluation/
  eval.py                  ← checkpoint evaluation vs heuristic baselines
tests/                     ← 11 test dosyası
scripts/
  diagnose_heading.py      ← heading diagnostic (checkpoint yükle, 5 ep, 50 adımda bir delta logla)
```

### Data Flow per Environment Step

1. `Normalizer` converts raw aircraft state (18D) → observation vector (**50D** = 17 ego + 9 teammate + 12×2 enemies)
   - Index 16 of ego: `cooldown_norm` (kalan cooldown / max_cooldown)
2. `DogfightEnv.step()` Blue agent action'ından önce `_critical_recovery_blue()` kontrol eder (OOB/zemin override)
3. `MAPPOActor` maps observation → 5D action (aileron, elevator, rudder, throttle, fire)
   - **squash:** tanh yerine `clamp(-1,1)` (aileron/elevator/rudder), `clamp(0,1)` (throttle) — satürasyon yok
4. `DogfightEnv.step()` applies actions through `AircraftModel` (RK4 physics, dt=0.05s)
5. `WeaponsModel` checks WEZ cone (±30°, 300–8000m), applies damage on hit
6. `RewardModel` computes 9-term reward + `r_fire_ready` + `r_pursuit` + `r_smooth`
7. Team kill bonus: tüm Red'ler ölünce her hayatta Blue'ya `+team_kill_bonus` eklenir
8. `RolloutBuffer` stores (obs, action, log_prob, value, reward, done)

### MAPPO Training Loop

- Collect `n_steps=512` steps → compute GAE (γ=0.99, λ=0.95) → PPO minibatch updates
- Red opponent: **`OpponentPool`** (adaptive fictitious self-play) — heuristic fallback pool boşken
- Pool adaptif seçim: global win_rate < 0.2 → eski checkpoint, > 0.6 → yeni checkpoint
- `MIN_MATCHES=5`: global_wr hesabına yalnızca ≥5 maç görmüş checkpoint'ler dahil edilir
- Buffer her zaman max (2v2) boyutunda; 1v1 fazlarda mevcut olmayan ajan slotları sıfırla doldurulur
- **mean_penalty:** action mean'e L2 ceza (`mean_pen_coeff=0.01`) — raw output'u ±3 aralığında tutar
- **gat_comm optimizer:** ayrı opt_gat yok; gat_comm + fc1_new parametreleri opt_actor içinde

### Active Hyperparameters

| Parametre | Değer |
|-----------|-------|
| `entropy_coeff` | 0.15 |
| `mean_penalty_coeff` | 0.01 |
| `n_steps` | 512 |
| `max_steps` | 1000 |
| `w_survival` | 0.0 |
| `w_kill` | 25 |
| `w_wez` | 5 |
| `w_tracking` | 6.0 |
| `w_pursuit` | 2.5 |
| `pursuit_norm_dist` | 10000m |
| `w_smooth_ctrl` | 0.05 |
| `w_smooth_throttle` | 0.003 |
| `w_throttle_ctx` | 0.8 |
| `ammo_miss_penalty` | 0.5 |
| `team_kill_bonus` | 15.0 |
| `fire_cooldown` | 0.5s |
| `freeze_steps` | 0 |
| `obs_dim` | 50 (GAT: 68D) |
| `pool_update_interval` | 200 |
| `phase1_win_threshold` | 0.40 |
| `total_timesteps` | 40_000_000 |

### Reward: r_smooth (Split Penalty)

`RewardModel._smooth_reward(prev_action)` — `dogfight_env.step()` tarafından çağrılır:
- `action_delta = action_t - action_{t-1}` (episode başında None → 0)
- `r_smooth_ctrl = -w_smooth_ctrl × sum(action_delta[:3]^2)` (aileron/elevator/rudder)
- `r_smooth_throttle = -w_smooth_throttle × action_delta[3]^2` (throttle — daha gevşek)
- Ağırlıklar: `w_smooth_ctrl=0.05`, `w_smooth_throttle=0.003`
- Amaç: throttle'ı ayrı cezalandır — uçuş kontrolleri kadar jitter beklenmez

### Reward: r_throttle_ctx (Bağlam-bazlı Throttle)

`RewardModel._throttle_context_reward(agent_state, enemy_states, action)`:
- ATA (aspect angle to attacker) ve mesafe bazlı yönlendirme
- Düşmana yakın + tehdit altında → tam gaz ödüllendirilir; uzakta boşta tam gaz cezalanır
- Ağırlık: `w_throttle_ctx=0.8`

### Reward: r_pursuit

`RewardModel._pursuit_reward(agent_state, enemy_states)`:
- En yakın hayatta düşmana mesafe `d` hesaplanır
- `r_pursuit = max(0, 1.0 - d / pursuit_norm_dist) × 0.3`
- Ağırlık: `w_pursuit = 2.5`, `pursuit_norm_dist = 10000m`

### Reward: ammo_miss_penalty

`RewardModel._resource_reward()`:
- WEZ dışı ateş: `-ammo_miss_penalty` (0.5)
- WEZ içi ıskalama: `-1.0`
- WEZ içi isabet: pozitif (mevcut)

### Reward: team_kill_bonus

`dogfight_env.step()` — done kontrolü sonrasında:
- Tüm Red'ler ölünce (`winner == BLUE`) her hayatta Blue ajana `+15.0` eklenir

### OOB CRITICAL Recovery — Blue MAPPO Agent

`DogfightEnv._critical_recovery_blue(state)` — `envs/dogfight_env.py`:
- Blue MAPPO agent'ın ağ çıktısı normalize edilmeden önce override edilir
- **Zemin (h < 300m):** burun kaldır (elevator pull ∝ mesafe), tam gaz, kanat düzelt
- **Harita sınırı (`_CRIT_MAP_MARGIN=5000m`'den az):** merkeze doğru bearing_angle hesapla, sert dönüş + rudder + tam gaz
  - `action[DA] = clip(bear_err × 1.5, -1, 1)` — aileron
  - `action[DR] = clip(bear_err × 0.5, -1, 1)` — koordineli dönüş için rudder
  - `action[DE] = 0.2`, `action[DT] = 1.0`
- Red heuristic zaten kendi `_critical_recovery()` metoduna sahipti — artık eşit koşul
- **NOT:** Self-play rakipler Blue'yu evasion manevrasına zorladığında OOB artar (heuristic-only=0.07, self-play=0.34-0.42) — bu OOB'nin birincil kaynağı throttle değil evasion

### MAPPOActor.squash() — Clamp vs Tanh

**Eski (bug):** `tanh` kullanıyordu → action mean ±17-26'ya büyüyünce `tanh(17)≈1.0` — tüm outputlar sabit, GAT sinyali etkisiz.

**Yeni:** `clamp(-1,1)` aileron/elevator/rudder, `clamp(0,1)` throttle. [-1,1] aralığında gradient=1, dışarıda gradient=0 ama mean bu kadar büyümez çünkü `mean_penalty` bunu önler.

### Curriculum — 4-Fazlı Plan (curriculum_v2)

`CurriculumManager` (training/train_mappo.py) ve `DogfightEnv.set_curriculum_phase()` birlikte çalışır.

| Dahili Faz | İsim | Senaryo | Geçiş Kriteri |
|---|---|---|---|
| 1 | Faz-1 | 1v1, WEZ-yakın spawn (500-1500m) | kill/ep≥0.30 AND win≥0.40, min 100 ep |
| 2 | Faz-1.5 | 1v1, dinamik mesafe (4000→16000m, +1000m/100ep) | kill/ep≥0.15, min 1200 ep |
| 3 | Faz-2 | 1v1, normal spawn (6000-12000m) | kill/ep≥0.15, min 300 ep |
| 4 | Faz-3 | 2v2, normal spawn (4000-12000m) | son faz |

**Faz-1.5 geri çekme:** Her 100 ep'de kill/ep < 0.10 ise spawn mesafesi -1000m geri çekilir (min 4000m).

**Spawn heading hizalaması (TÜM FAZLAR):** `_spawn_agents_wez_close()`, `_spawn_agents_dynamic_dist()`, `_spawn_agents_normal()` — hepsinde Blue ve Red, karşılıklı `arctan2` tabanlı ±45° heading hizalaması uygulanır.

### Observation Dimensions

- **50D** (tüm fazlar): 17 (ego + cooldown) + 9 (teammate) + 12×2 (enemies)
- 1v1 fazlarda eksik teammate/enemy slotları dummy dead-state ile doldurulur → ağ boyutu sabit

### Spawn: _spawn_agents_normal()

`DogfightEnv._spawn_agents_normal()` (envs/dogfight_env.py):
- Blue: harita içinde rastgele pozisyon (±40% range)
- Red: Blue'dan `rng.uniform(normal_spawn_dist, normal_spawn_dist_max)` uzakta rastgele yönde
- Heading hizalaması: `psi_b = arctan2(y_r-y_b, x_r-x_b) + uniform(-π/4, π/4)`
- 2v2: wingmen ±1000m x-offset ile aynı heading hizalamasına tabi

### OpponentPool — Adaptive Fictitious Self-Play

`agents/opponent_pool.py` — `OpponentPool` sınıfı:
- Ring buffer (max 20 checkpoint); her `pool_update_interval=200` episode'da snapshot alınır
- `record_outcome(is_win)` → her checkpoint için son 20 ep win_rate takibi
- Adaptif seçim: global_wr < 0.2 → en eski checkpoint, > 0.6 → en yeni, diğer → ağırlıklı random
- `MIN_MATCHES=5`: az maç görmüş checkpoint'ler global_wr hesabından hariç tutulur
- Yeni checkpoint default win_rate: `0.0` (güçlü gibi seçilmesini engeller)
- **UYARI:** `pool_update_interval: 999999` yapılırsa pool tamamen durur, sadece heuristic oynanır — bunu yapma

### Faz-2 GAT İletişim Mimarisi

`models/gat_comm.py` — `GATComm` sınıfı:
- Graf: 2 düğüm (mavi ajanlar), kenar özellikleri: [distance_norm, bearing_norm, threat_score]
- 4 attention head × 4 boyut = 16D çıkış mesajı
- ~1.2K parametre (hafif modül)

`GATMAPPOActor` / `GATMAPPOCritic` (training/train_mappo.py):
- Split first layer: `fc1_old` (eski 50D, checkpoint'ten) + `fc1_new` (18D, sıfır init)
- `forward`: `relu(fc1_old(obs[:50]) + fc1_new(obs[50:]))`
- **gat_comm + fc1_new parametreleri opt_actor içinde** — ayrı opt_gat yok (eski bug: opt_gat hiç step() çağrılmıyordu)

**Transfer learning stratejisi (freeze_steps=0 ile devre dışı):**
- `freeze_steps > 0` ise: `fc1_old` dondurulmuş → GATComm + fc1_new + tail öğrenir
- `freeze_steps = 0` (default): tüm ağ baştan fine-tune

**Obs boyutu değişimi:**
- Faz-1: 50D (17 ego + 9 tm + 12×2 enemy)
- Faz-2: 68D (50 base + 2 role_emb + 16 GAT mesajı)
- Critic: 100D → 136D (2 ajan × 68D)

**Başlatma (Faz-1 tamamlanınca):**
```powershell
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = "logs/train_$timestamp.txt"
python -u -X utf8 training/train_mappo.py --phase2 checkpoints/mappo_final.pt | Tee-Object -FilePath $log
```

### QMIX Mimarisi

`models/qmix_net.py`:
- `ActionMapper`: 162 ayrık indeks → 5D sürekli aksiyon lookup tablosu (itertools.product)
- `AgentQNetwork`: obs(50D) → Linear(128) → ReLU → Linear(128) → ReLU → Q(162), paylaşımlı ağ
- `QMixNet`: [Q_1, Q_2] + global_state(100D) → Q_total; hypernetwork ağırlıkları abs() ile pozitif → ∂Q_tot/∂Q_i ≥ 0

`training/train_qmix.py` — `QMIXTrainer`:
- `ReplayBuffer`: individual transitions, capacity=50k, numpy arrays
- `_select_actions()`: ε-greedy (1.0→0.05, 100k adım lineer)
- `_red_actions()`: `MultiHeuristicPolicy.act(states)` → red_ids filtrelenmiş
- `_update()`: Sample batch → online Q → mixer → target Q → MSE loss; `r_team = rewards.mean(dim=1)`
- Target sync: her `target_update_interval=200` episode hard copy
- Grad clip: 10.0

**QMIX Hiperparametreler** (`configs/config.yaml` → `qmix:` bölümü):
| Parametre | Değer |
|-----------|-------|
| `lr` | 1e-4 |
| `gamma` | 0.99 |
| `batch_size` | 64 |
| `buffer_capacity` | 50_000 |
| `eps_start / eps_end` | 1.0 / 0.05 |
| `eps_decay_steps` | 100_000 |
| `target_update_interval` | 200 ep |
| `train_freq` | 4 adım |
| `total_episodes` | 30_000 |
| `qmix_hidden` | 64 |
| `agent_hidden` | 128 |

## Curriculum Öğrenme Geçmişi ve Bulgular

### Policy Collapse Sorunu (Çözüldü)
**Gözlemlenen pattern:** Red heuristic tam güce geçince Blue win rate sıfıra düşüyordu (3 denemede tekrarlandı).
**Çözüm:** Adaptive OpponentPool ile fictitious self-play — sabit heuristic'e kilitlenme ortadan kalktı.

### WEZ Firing Sorunu (Çözüldü)
**Sorun:** Agent WEZ içinde bile ateş etmiyordu — cooldown 2s çok uzun, cooldown obs'da yoktu.
**Çözümler:** fire_cooldown 2s→0.5s, cooldown_norm obs'a eklendi (index 16), r_fire_ready reward shaping.

### Heading / Navigation Sorunu (Çözüldü)
**Sorun:** Normal spawn (5000-20000m) koşullarında heading diagnostic ortalaması 56-106° delta gösterdi.
**Çözüm:** r_pursuit reward + spawn heading hizalaması (±45°) tüm fazlara uygulandı + Faz-1.5 kademeli mesafe.
**Sonuç:** kill/ep ep 200'de 0.75-0.83'e yükseldi.

### Faz-3 Timeout Sorunu (Çözüldü, 2026-04-18)
**Sorun:** max_steps=2000, w_survival=0.001 ile ep'lerin %92-98'i timeout.
**Çözüm:** max_steps=1500, w_survival=0.0, w_tracking=3.0, team_kill_bonus=5.0, spawn_dist=4000m.
**Sonuç:** kill/ep 0.03→0.32, draw %98→88%.

### Tanh Satürasyon Sorunu (Çözüldü, 2026-04-28)
**Sorun:** GAT fc1_new öğrense de (norm 5.45), action mean ±17-26'ya büyümüş — `tanh(17)≈1.0` yüzünden ablation farkı sıfır.
**Çözüm:** `squash()` tanh→clamp, `mean_penalty_coeff=0.01`, `entropy_coeff` 0.10→0.15.

### opt_gat Optimizer Bug (Çözüldü, 2026-04-28)
**Sorun:** `opt_gat` oluşturulmuş ama PPO döngüsünde `.zero_grad()` ve `.step()` hiç çağrılmıyordu → fc1_new norm ep 3000'de hâlâ 0.0.
**Çözüm:** opt_gat tamamen kaldırıldı; gat_comm + fc1_new parametreleri opt_actor'a taşındı.

### OOB Sorunu (Çözüldü, 2026-04-28)
**Sorun:** Blue MAPPO agent harita dışına çıkınca ölmüyor (is_out_of_bounds() hiç çağrılmıyordu) → OOB episode timeout olarak bitiyordu (~%10-12).
**Çözüm:** `DogfightEnv._critical_recovery_blue()` — harita kenarı (3000m) ve zemin (300m) override. Red heuristic'teki CRITICAL mantığının aynısı.

### Pool Dondurulunca Faz-1 Platoya Giriyor (Öğrenildi, 2026-04-28)
**Gözlem:** `pool_update_interval=999999` ile 11300 episode boyunca win_rate ~0.32 — 0.55 eşiği aşılamadı.
**Sebep:** Self-play olmadan sabit heuristic'e karşı training plateau yapıyor. Önceki başarılı run'da pool devreye girince 4.3x artış olmuştu.
**Sonuç:** Pool her zaman aktif olmalı (pool_update_interval≤200).

### Mevcut Durum (2026-05-12)

**MAPPO GAT tamamlandı:**
- Final checkpoint: `mappo_gat_ep44000.pt` (Faz-3, 2v2)
- En iyi eval checkpoint: `mappo_gat_ep37000.pt` — heuristic'e karşı win=69.6%, kill/ep=1.51

**Final Eval — 500 ep, seed=42, 2v2, heuristic:**
| Model | Checkpoint | Win% | 95% CI | Kill/ep | 2nd Kill% | Draw% |
|-------|-----------|------|--------|---------|-----------|-------|
| MAPPO baseline | `mappo_final.pt` | 52.8% | — | 1.18 | — | 45.0% |
| GAT-MAPPO | `mappo_gat_ep37000.pt` | 69.6% | — | 1.51 | — | 29.2% |
| GAT-MAPPO+Role | `mappo_gat_ep44000.pt` | 62.4% | [58.1–66.5%] | 1.33 | 62.4% | 36.6% |
| **QMIX best** | `qmix_ep2000.pt` | **70.2%** | [66.0–74.0%] | **1.62** | **70.2%** | 27.8% |
| QMIX ep2500 | `qmix_ep2500.pt` | 53.0% | [48.6–57.3%] | 1.47 | 53.0% | 45.0% |

- QMIX ep2000 = GAT-MAPPO ep37000 (CI örtüşüyor), ama 844k vs 29.3M step → ~35× sample efficiency
- Role/OM sistemi zararlı (-7.2pp)
- QMIX ep2500 overfit: rolling W=0.92 ama eval W=0.53 — peak ep2000'deydi

### Key Configuration (configs/config.yaml)

Tüm fizik sabitleri, reward ağırlıkları ve curriculum parametreleri burada. `curriculum_v2` bölümü 4-fazlı curriculum'u tanımlar.

Kritik parametreler:
- `phase15_dist_start: 4000.0`, `phase15_dist_max: 16000.0`, `phase15_dist_step: 1000.0`, `phase15_step_episodes: 100`
- `phase15_pullback_thresh: 0.10`, `phase15_min_episodes: 1200`
- `phase2_kill_threshold: 0.15`, `phase2_min_episodes: 300`
- `phase3_spawn_dist: 2500.0` (geçici), `phase3_spawn_dist_max: 12000.0`
- `w_pursuit: 2.5`, `pursuit_norm_dist: 10000.0`, `w_tracking: 6.0`, `w_survival: 0.0`
- `w_smooth_ctrl: 0.05`, `w_smooth_throttle: 0.003`, `w_throttle_ctx: 0.8`
- `team_kill_bonus: 15.0`, `max_steps: 1000`
- `total_timesteps: 40_000_000`
- `communication.enable_comms: false` (--phase2 argümanı True'ya çeker)

### Bilinen Sorunlar
- `pool_start_step` (config: 500k) kodda kullanılmıyor — pool ep 200'den itibaren aktif (erken, sorunsuz)
- `phase3_spawn_dist: 2500.0` geçici — Faz-3'e girilince gerekirse ayarlanmalı
- Draw ~%80 (Faz-3): 2v2'de ikinci kill tamamlanamıyor — team_kill_bonus=15 ile iyileştirildi, yeterli mi test edilmedi
- **CurriculumManager checkpoint'te state saklamıyor:** Resume edince her zaman Faz-1'den başlar → GAT Faz-3 devamı için `--start-phase 4` zorunlu
- **OOB self-play evasion pattern:** Pool checkpoint'ler sertleştikçe Blue evasion manevrasına zorlanıyor → OOB artar (heuristic=0.07, self-play=0.34-0.42). `_CRIT_MAP_MARGIN=5000m` fix kısmen yardımcı oldu ama çözüm değil.
- **Pool hardening oscillation:** Yeni checkpoint'ler wr=0.00 başlıyor, maç görünce sertleşiyor → model win rate'i düşüyor → eski checkpoint baskın → tekrar yüksekliyor. 0.40 eşiğini aşmak zor.
- **ep42000 deterministic mode collapse:** Training win=0.30 ama deterministic inference'da 40/40 ep 0 kill. Stokastik→deterministic gap var — en iyi checkpoint ep37000.

### Throttle Reward Etkisi (2026-05-05)
**Deney:** Split smooth (w_smooth_ctrl=0.05, w_smooth_throttle=0.003) + w_throttle_ctx=0.8 + w_pursuit=2.5 (pursuit_norm_dist=10000m).
**Sonuç:** Draw %87→37-45% — çok pozitif. OOB biraz arttı (evasion manevraları) ama genel etki lehte.
**Lesson:** Throttle reward ayrı cezalandırılmalı — kontrol yüzeyleri kadar jitter beklenmez; throttle değişimi doğaldır.

### OOB Self-Play Evasion Pattern (2026-05-06)
**Gözlem:** Heuristic-only OOB~0.07, self-play aktifken OOB=0.34-0.42. 5000m margin ile kısmen düzeldi.
**Kök neden:** Pool checkpoint'lerin sertleşmesiyle Blue evasion'a zorlanıyor → sınıra yaklaşıyor. Throttle reward değil, evasion manevraları.
**Lesson:** OOB'nin gerçek çözümü: sınıra yaklaşınca reward gradient (boundary penalty shaping) veya daha güçlü evasion policy.

### Pool Hardening Oscillation (2026-05-06)
**Gözlem:** ep37000-42000 run'da win rate ~0.20-0.34 bandında oscillate etti, hiç 0.40'ı aşamadı.
**Mekanizma:** Yeni pool snapshot'lar wr=0.00 başlıyor → Blue kazanıyor → snapshot sertleşiyor → Blue kaybediyor → eski (daha kolay) checkpoint devreye → tekrar yükseliyor.
**Lesson:** Pool update interval'ı düşürmek (200 → 100?) veya new checkpoint başlangıç wr'ını daha yüksek tutmak oscillation'ı azaltabilir.

### GAT Deterministic Collapse (2026-05-06)
**Gözlem:** ep42000 — training win=0.30 ama deterministic inference 40/40 ep'de 0 kill.
**Kök neden:** Stokastik→greedy gap: Entropy yüksek (0.15) ile pol. stochastic eğitim görüyor, greedy aksiyon tamamen farklı davranıyor.
**Lesson:** Eğer checkpoint deterministic eval için kullanılacaksa, training sırasında deterministic eval metriğini de takip et.

### Eval Heuristic Overwrite Bug (Çözüldü, 2026-05-11)
**Sorun:** `compare_checkpoints.py` içinde `{**actions, **opp_actions}` sözlük birleştirme tüm aksiyonları (blue dahil) heuristic ile üzerine yazıyordu → tüm checkpoint'ler W≈0.04 görünüyordu.
**Çözüm:** `opp_actions` filtrelenerek yalnızca `red_ids` anahtarları alındı: `{k: v for k, v in all_heuristic.items() if k not in set(env.blue_ids)}`.
**Lesson:** Eval kodunda multi-agent aksiyon birleştirme yapılırken takım ID filtresi zorunlu.

### Role System / Supervisor Label Collapse (Öğrenildi, 2026-05-12)
**Deney:** GAT+OM+RoleSelector modeli (ep44000-ep52000). Rol dağılımı ep52000'de: sniper=0.00, pursuit=0.82-0.89.
**Kök neden 1 (supervisor):** Supervisor label `r_closing_raw > 0` → PURSUIT ataması. Heuristic rakip her adımda kapanıyor → r_closing_raw neredeyse her adımda pozitif → PURSUIT label %100 baskın.
**Kök neden 2 (epsilon):** ε tamamen decay edince (~%3) exploration biter → RoleSelector greedy seçim → pursuit reinforcement loop.
**Düzeltme denemesi:** `current_role == X` koşulu kaldırıldı, tactical context bazlı label (hp<0.5→DEF, r_sup>0.10→SUP, sniper_pos>0.10 AND closing≤0→SNP, closing>0 AND sniper_pos<0.05→PRS).
**Sonuç:** ep200 rol dağılımı düzeldi (sniper=0.15, pursuit=0.56) ama full run yapılmadan QMIX'e geçildi.
**Lesson:** Rol bazlı auxiliary loss ile ana policy aynı optimizasyonu paylaşırsa self-reinforcing loop kaçınılmaz. Supervisor label, role-independent tactical context'e dayalı olmalı.

### QMIX Mimarisi (2026-05-12)
**Karar:** MAPPO (on-policy, 5D continuous) → QMIX (off-policy, 162 discrete).
**Aksiyon ayrıştırma:** 3×3×3×3×2=162 (aileron/elevator/rudder:{-1,0,1}, throttle:{0,0.5,1}, fire:{0,1}). `ActionMapper` lookup tablosu.
**Ağ:** `AgentQNetwork` MLP (obs 50D→128→128→162), GRU yok — stateless.
**Mixer:** `QMixNet` hypernetwork — abs() ile pozitif ağırlıklar → monotonicity garantisi (IGM koşulu).
**Buffer:** Individual transitions, capacity=50k (episode değil adım bazlı).
**Keşif:** ε-greedy 1.0→0.05, 100k adım lineer decay.
**Target sync:** Her 200 episode hard copy.
**Rakip:** Heuristic (sabit), OpponentPool yok — QMIX için pool mimarisi henüz eklenmedi.
**Parametre sayısı:** 69,895 (MAPPO'ya göre çok daha küçük).

**QMIX overfit paterni (öğrenildi):** Rolling training W ile held-out eval W arasında ciddi gap oluşabilir (rolling W=0.92 → eval W=0.53). Q-overestimation + target network lag. `save_interval=500` ile peak checkpoint'i kaybetme.

### Code Language

Docstrings and inline comments are written in Turkish (Türkçe).
