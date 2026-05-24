"""
eval.py
=======
Eğitilmiş MAPPO politikasını heuristic baseline'a karşı değerlendirir.

Kullanım:
    python eval.py --checkpoint checkpoints/mappo_final.pt
    python eval.py --checkpoint checkpoints/mappo_ep1000.pt --episodes 50
    python eval.py --checkpoint checkpoints/mappo_final.pt --deterministic

Çıktılar:
    Konsol : episode bazlı sonuçlar + özet tablo
    JSON   : logs/eval_results.json

Metrikler:
    win_rate       — Blue (MAPPO) kazanma oranı
    loss_rate      — Blue kaybetme oranı
    draw_rate      — beraberlik oranı
    kill_per_ep    — episode başı ortalama kill
    survival_rate  — hayatta kalan Blue ajan oranı
    mean_reward    — episode başı ortalama reward
    mean_ep_len    — ortalama episode uzunluğu (adım)

Faz 1 geçiş kriteri (curriculum.yaml):
    win_rate   >= 0.40
    kill_per_ep >= 0.80
    oob_rate   <= 0.05

Bağımlılıklar:
    - torch
    - dogfight_env.py
    - heuristic_agent.py
    - train_mappo.py  (MAPPOActor, MAPPOTrainer)
"""

import sys
import json
import argparse
import yaml
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from envs.dogfight_env import DogfightEnv, BLUE, RED
from agents.heuristic_agent import MultiHeuristicPolicy
from utils.normalization import Normalizer
from envs.aircraft_model import STATE_X, STATE_Y, STATE_H, STATE_ALIVE
from envs.geometry_utils import distance_3d, bearing_angle, wrap_to_pi
from models.gat_comm import GATComm
from models.om_net import (
    CentralizedOpponentModel, CentralizedRoleAssigner,
    EnemyHistoryBuffer, build_team_state, INTENT_DEFENSIVE,
)

_BASE_OBS_DIM = 50   # tüm checkpoint türleri için base obs boyutu

# ---------------------------------------------------------------------------
# Faz 1 Geçiş Kriterleri
# ---------------------------------------------------------------------------
PHASE1_CRITERIA = {
    "win_rate":    0.40,
    "kill_per_ep": 0.80,
    "oob_rate":    0.05,   # maksimum
}


# ===========================================================================
# Evaluator
# ===========================================================================

class Evaluator:
    """
    Eğitilmiş MAPPO politikasını değerlendirir.

    Kullanım:
        ev = Evaluator(config, checkpoint_path, device="cuda")
        results = ev.run(n_episodes=20, deterministic=True)
        ev.print_summary(results)
    """

    def __init__(self, config: dict, checkpoint_path: str,
                 device: str = "auto"):
        assert _TORCH_AVAILABLE, "PyTorch bulunamadı. pip install torch"

        self.config = config
        self.device = self._resolve_device(device)

        # Ortam
        self.env       = DogfightEnv(config)
        self.env.set_curriculum_phase(4)   # 2v2 normal spawn (eval her zaman Faz-3)
        action_dim     = self.env.action_dim
        hidden         = int(config["training"].get("hidden_dim", 256))

        # Checkpoint'i önce oku — architecture tespiti
        ckpt      = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        actor_sd  = ckpt.get("actor", {})

        # Architecture tespiti:
        #   fc1_intent.weight → yeni 78D GAT+OM
        #   fc1_old.weight (fc1_intent yok) → eski 68D GAT
        #   otherwise → base 50D MAPPO
        self._use_om  = ('fc1_intent.weight' in actor_sd)
        self._gat_old = (not self._use_om) and ('fc1_old.weight' in actor_sd)
        self._use_gat = self._use_om or self._gat_old

        if self._use_gat:
            from training.train_mappo import GATMAPPOActor
            self.obs_dim = 78
            self.actor   = GATMAPPOActor(
                old_obs_dim=50, ext_dim=18, intent_dim=6, role_dim=4,
                action_dim=action_dim, hidden=hidden,
            ).to(self.device)
            # GAT iletişim modülü
            comm_cfg = config.get("communication", {})
            self._gat_node_dim  = int(comm_cfg.get("node_dim", 17))
            self._gat_wez_range = float(config.get("weapons", {}).get("wez_range_max", 8000.0))
            self.gat_comm = GATComm().to(self.device)
            if 'gat_comm' in ckpt:
                self.gat_comm.load_state_dict(ckpt['gat_comm'])
            self.gat_comm.eval()
            # OM bileşenleri (yeni checkpoint'lerde)
            if self._use_om:
                self.cent_om   = CentralizedOpponentModel().to(self.device)
                self.cent_role = CentralizedRoleAssigner().to(self.device)
                if 'cent_om' in ckpt:
                    self.cent_om.load_state_dict(ckpt['cent_om'])
                if 'cent_role' in ckpt:
                    self.cent_role.load_state_dict(ckpt['cent_role'])
                self.cent_om.eval(); self.cent_role.eval()
                self.enemy_hist = EnemyHistoryBuffer()
            else:
                self.cent_om    = None
                self.enemy_hist = None
        else:
            from training.train_mappo import MAPPOActor
            self.obs_dim  = _BASE_OBS_DIM
            self.gat_comm = None
            self.cent_om  = None
            self.enemy_hist = None
            self.actor = MAPPOActor(_BASE_OBS_DIM, action_dim,
                                    hidden=hidden).to(self.device)

        self._load_actor_weights(actor_sd)
        self.actor.eval()

        # Heuristic rakip (Red takımı)
        self.train_ids = self.env.blue_ids
        self.opp_ids   = self.env.red_ids
        team_map = {aid: ("blue" if "blue" in aid else "red")
                    for aid in self.env.agent_ids}
        self.opp_policy = MultiHeuristicPolicy(
            config, self.env.agent_ids, team_map
        )

        if self._use_om:
            mode = "78D GAT+OM"
        elif self._gat_old:
            mode = "68D GAT"
        else:
            mode = "50D base"
        print(f"[Eval] Device     : {self.device}")
        print(f"[Eval] Checkpoint : {checkpoint_path}")
        print(f"[Eval] Mode       : {mode}  obs_dim={self.obs_dim}")

    # -----------------------------------------------------------------------
    # Ana Değerlendirme
    # -----------------------------------------------------------------------

    def run(self, n_episodes: int = 20,
            deterministic: bool = True,
            seed: int = 0,
            verbose: bool = True) -> dict:
        """
        n_episodes kadar episode çalıştır, metrikleri topla.

        Returns
        -------
        results : dict — tüm metrikler + episode bazlı detay
        """
        self.env.seed(seed)

        # Sayaçlar
        wins = 0; losses = 0; draws = 0
        kills_list      = []
        rewards_list    = []
        lengths_list    = []
        survival_list   = []
        oob_list        = []

        ep_details = []

        for ep in range(n_episodes):
            obs_dict = self.env.reset()
            self.opp_policy.reset()
            if self.enemy_hist is not None:
                self.enemy_hist.reset()

            ep_reward  = {aid: 0.0 for aid in self.train_ids}
            ep_kills   = 0
            ep_oob     = 0
            done       = {"__all__": False}

            while not done["__all__"]:
                # GAT / GAT+OM obs uzantısı
                if self._use_gat:
                    obs_dict = self._extend_obs(obs_dict)
                # MAPPO aksiyonları
                actions = self._get_actions(obs_dict, deterministic)

                # Heuristic aksiyonları
                state_dict  = self.env.get_all_states()
                opp_actions = self.opp_policy.act(state_dict)

                red_actions = {k: v for k, v in opp_actions.items()
                               if k in set(self.opp_ids)}
                action_dict = {**actions, **red_actions}
                obs_dict, rew_dict, done, info_dict = self.env.step(action_dict)

                # İstatistik
                for aid in self.train_ids:
                    ep_reward[aid] += rew_dict[aid]
                    if info_dict[aid].get("r_kill", 0.0) > 0.5:
                        ep_kills += 1

            # Episode sonucu
            winner = done.get("winner", "draw")
            if winner == BLUE:
                wins   += 1
            elif winner == RED:
                losses += 1
            else:
                draws  += 1

            # Hayatta kalma
            n_survived = len(self.env.blue_alive)
            survival_list.append(n_survived / len(self.train_ids))

            # Sınır dışı (info'dan)
            for aid in self.train_ids:
                ep_summary = info_dict[aid].get("episode", {})
                # OOB: reward_model penalty'den tahmin
                # (reward_model r_penalty < -1 ise OOB sayılır)
                if ep_summary.get("mean/r_penalty", 0.0) < -0.5:
                    ep_oob += 1
            oob_rate_ep = ep_oob / len(self.train_ids)
            oob_list.append(oob_rate_ep)

            mean_rew = float(np.mean([ep_reward[aid]
                                       for aid in self.train_ids]))
            kills_list.append(ep_kills)
            rewards_list.append(mean_rew)
            lengths_list.append(self.env.get_step_count())

            detail = {
                "episode":  ep + 1,
                "winner":   winner,
                "kills":    ep_kills,
                "reward":   round(mean_rew, 3),
                "steps":    self.env.get_step_count(),
                "survived": n_survived,
            }
            ep_details.append(detail)

            if verbose:
                result_icon = {"blue": "🏆", "red": "💀", "draw": "🤝"}
                print(
                    f"  Ep {ep+1:>3}/{n_episodes} "
                    f"{result_icon.get(winner, '?')} {winner:<4} | "
                    f"kills={ep_kills} | "
                    f"rew={mean_rew:>7.2f} | "
                    f"steps={self.env.get_step_count():>4}"
                )

        # ── Özet Metrikler ────────────────────────────────────────────
        results = {
            "n_episodes":    n_episodes,
            "deterministic": deterministic,

            # Ana metrikler
            "win_rate":      wins   / n_episodes,
            "loss_rate":     losses / n_episodes,
            "draw_rate":     draws  / n_episodes,

            # Episode bazlı ortalamalar
            "kill_per_ep":   float(np.mean(kills_list)),
            "mean_reward":   float(np.mean(rewards_list)),
            "mean_ep_len":   float(np.mean(lengths_list)),
            "survival_rate": float(np.mean(survival_list)),
            "oob_rate":      float(np.mean(oob_list)),

            # Ham dağılımlar
            "wins":   wins,
            "losses": losses,
            "draws":  draws,
            "kill_std":    float(np.std(kills_list)),
            "reward_std":  float(np.std(rewards_list)),

            # Episode detayları
            "episodes": ep_details,
        }

        return results

    # -----------------------------------------------------------------------
    # Faz Geçiş Kriteri Kontrolü
    # -----------------------------------------------------------------------

    @staticmethod
    def check_phase_criteria(results: dict,
                              criteria: dict = None) -> tuple:
        """
        Curriculum geçiş kriterlerini kontrol eder.

        Returns
        -------
        passed  : bool
        report  : dict — her kriter için detay
        """
        if criteria is None:
            criteria = PHASE1_CRITERIA

        report = {}
        all_passed = True

        checks = [
            ("win_rate",    criteria["win_rate"],    ">="),
            ("kill_per_ep", criteria["kill_per_ep"], ">="),
            ("oob_rate",    criteria["oob_rate"],    "<="),
        ]

        for key, threshold, op in checks:
            val = results.get(key, 0.0)
            if op == ">=":
                ok = val >= threshold
            else:
                ok = val <= threshold
            report[key] = {
                "value":     round(val, 4),
                "threshold": threshold,
                "op":        op,
                "passed":    ok,
            }
            if not ok:
                all_passed = False

        return all_passed, report

    # -----------------------------------------------------------------------
    # Çıktı
    # -----------------------------------------------------------------------

    @staticmethod
    def print_summary(results: dict, show_criteria: bool = True):
        """Sonuçları konsola yazdır."""
        print("\n" + "=" * 55)
        print("📊 DEĞERLENDİRME SONUÇLARI")
        print("=" * 55)
        print(f"  Episode sayısı  : {results['n_episodes']}")
        print(f"  Deterministik   : {results['deterministic']}")
        print()
        print(f"  🏆 Win rate     : {results['win_rate']:.1%}  "
              f"({results['wins']} / {results['n_episodes']})")
        print(f"  💀 Loss rate    : {results['loss_rate']:.1%}  "
              f"({results['losses']} / {results['n_episodes']})")
        print(f"  🤝 Draw rate    : {results['draw_rate']:.1%}  "
              f"({results['draws']} / {results['n_episodes']})")
        print()
        print(f"  Kill / ep       : {results['kill_per_ep']:.2f} "
              f"± {results['kill_std']:.2f}")
        print(f"  Mean reward     : {results['mean_reward']:.2f} "
              f"± {results['reward_std']:.2f}")
        print(f"  Mean ep length  : {results['mean_ep_len']:.0f} adım")
        print(f"  Survival rate   : {results['survival_rate']:.1%}")
        print(f"  OOB rate        : {results['oob_rate']:.1%}")

        if show_criteria:
            passed, report = Evaluator.check_phase_criteria(results)
            print()
            print("─" * 55)
            print("🎯 FAZ 1 GEÇİŞ KRİTERLERİ")
            print("─" * 55)
            for key, detail in report.items():
                icon = "✅" if detail["passed"] else "❌"
                print(f"  {icon} {key:<14} : "
                      f"{detail['value']:.4f} "
                      f"{detail['op']} {detail['threshold']}")
            print()
            if passed:
                print("  🚀 TÜM KRİTERLER SAĞLANDI — Faz 2'ye geçilebilir!")
            else:
                print("  ⏳ Kriterler henüz sağlanmadı — eğitime devam.")
        print("=" * 55)

    @staticmethod
    def save_results(results: dict, path: str):
        """Sonuçları JSON'a kaydet."""
        # Episode detaylarını JSON-serializable yap
        out = {k: v for k, v in results.items() if k != "episodes"}
        out["episodes"] = results.get("episodes", [])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[Eval] Sonuçlar kaydedildi: {path}")

    # -----------------------------------------------------------------------
    # Yardımcı
    # -----------------------------------------------------------------------

    def _get_actions(self, obs_dict: dict,
                     deterministic: bool) -> dict:
        """Eğitilmiş actor ile aksiyon üret."""
        from training.train_mappo import MAPPOActor
        actions = {}
        with torch.no_grad():
            for aid in self.train_ids:
                o     = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                obs_t = torch.FloatTensor(o[:self.obs_dim]).unsqueeze(0).to(self.device)
                raw, _ = self.actor.act(obs_t, deterministic=deterministic)
                squashed     = MAPPOActor.squash(raw.squeeze(0))
                actions[aid] = squashed.cpu().numpy()
        return actions

    def _build_gat_edge_feats(self) -> np.ndarray:
        """Mavi takım arasında kenar özellik matrisi (N×N×3)."""
        N      = len(self.train_ids)
        edge   = np.zeros((N, N, 3), dtype=np.float32)
        states = self.env.get_all_states()
        for i, aid_i in enumerate(self.train_ids):
            for j, aid_j in enumerate(self.train_ids):
                if i == j:
                    continue
                s_i = states.get(aid_i)
                s_j = states.get(aid_j)
                if s_i is None or s_j is None:
                    continue
                pos_i = s_i[[STATE_X, STATE_Y, STATE_H]]
                pos_j = s_j[[STATE_X, STATE_Y, STATE_H]]
                dist  = distance_3d(pos_i, pos_j)
                bear  = bearing_angle(pos_i, pos_j)
                ts = 0.0
                for eid in self.opp_ids:
                    es = states.get(eid)
                    if es is not None and es[STATE_ALIVE] > 0.5:
                        d  = distance_3d(pos_j, es[[STATE_X, STATE_Y, STATE_H]])
                        ts = max(ts, float(np.clip(
                            1.0 - d / (self._gat_wez_range + 1e-9), 0.0, 1.0
                        )))
                edge[i, j] = [
                    float(np.clip(dist / (self.env.map_size + 1e-9), 0.0, 1.0)),
                    float(np.clip(wrap_to_pi(bear) / np.pi, -1.0, 1.0)),
                    ts,
                ]
        return edge

    def _extend_obs(self, obs_dict: dict) -> dict:
        """50D base obs'u 78D GAT(+OM) obs'a genişlet."""
        states = self.env.get_all_states()

        # GAT mesajları
        ego_list   = []
        alive_list = []
        for aid in self.train_ids:
            obs = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, dtype=np.float32))
            ego_list.append(obs[:self._gat_node_dim])
            s = states.get(aid)
            alive_list.append(float(s[STATE_ALIVE]) if s is not None else 0.0)
        edge_feats = self._build_gat_edge_feats()
        messages   = self.gat_comm.compute_messages(ego_list, edge_feats, alive_list, self.device)

        # Intent ve rol vektörleri
        if self._use_om:
            base_arr = np.stack([
                obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, dtype=np.float32))[:_BASE_OBS_DIM]
                for aid in self.train_ids
            ], axis=0)
            hist_960 = self.enemy_hist.update(base_arr)
            with torch.no_grad():
                hist_t    = torch.from_numpy(hist_960).unsqueeze(0).to(self.device)
                intent_np = self.cent_om.intent_flat(hist_t).squeeze(0).cpu().numpy()
            ts_np = build_team_state([states.get(bid) for bid in self.train_ids])
            with torch.no_grad():
                x_role = torch.from_numpy(
                    np.concatenate([intent_np, ts_np]).astype(np.float32)
                ).unsqueeze(0).to(self.device)
                role_0, role_1 = self.cent_role.assign(x_role)
            roles = [role_0.squeeze(0).cpu().numpy(), role_1.squeeze(0).cpu().numpy()]
        else:
            intent_np = np.zeros(6,  dtype=np.float32)
            roles     = [np.zeros(4, dtype=np.float32)] * len(self.train_ids)

        extended = {}
        for i, aid in enumerate(self.train_ids):
            base  = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, dtype=np.float32))[:_BASE_OBS_DIM]
            ext18 = np.concatenate([np.zeros(2, dtype=np.float32), messages[i]])
            extended[aid] = np.concatenate([base, ext18, intent_np, roles[i]])   # 78D
        for aid in self.opp_ids:
            extended[aid] = obs_dict.get(aid, np.zeros(_BASE_OBS_DIM, dtype=np.float32))
        return extended

    def _load_actor_weights(self, actor_sd: dict):
        """Actor state dict'ini mevcut self.actor'a yükle."""
        self.actor.load_state_dict(actor_sd, strict=False)

    @staticmethod
    def _resolve_device(device: str) -> "torch.device":
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


# ===========================================================================
# Hızlı Karşılaştırma: Heuristic vs Heuristic (baseline)
# ===========================================================================

def eval_heuristic_baseline(config: dict, n_episodes: int = 20,
                              seed: int = 0, verbose: bool = True) -> dict:
    """
    Heuristic vs Heuristic değerlendirmesi.
    MAPPO checkpointi olmadan baseline performansı ölçer.
    Beklenti: yaklaşık %50 win, %50 loss (simetrik).
    """
    env = DogfightEnv(config)
    env.seed(seed)

    team_map = {aid: ("blue" if "blue" in aid else "red")
                for aid in env.agent_ids}
    blue_policy = MultiHeuristicPolicy(config, env.agent_ids, team_map)
    red_policy  = MultiHeuristicPolicy(config, env.agent_ids, team_map)

    wins = 0; losses = 0; draws = 0
    kills_list   = []
    rewards_list = []
    lengths_list = []

    for ep in range(n_episodes):
        env.reset()
        blue_policy.reset()
        done = {"__all__": False}
        ep_kills = 0

        while not done["__all__"]:
            state_dict   = env.get_all_states()
            action_dict  = blue_policy.act(state_dict)
            _, rew, done, info = env.step(action_dict)
            for aid in env.blue_ids:
                if info[aid].get("r_kill", 0.0) > 0.5:
                    ep_kills += 1

        winner = done.get("winner", "draw")
        if winner == BLUE:   wins   += 1
        elif winner == RED:  losses += 1
        else:                draws  += 1

        kills_list.append(ep_kills)
        lengths_list.append(env.get_step_count())

        if verbose:
            icon = {"blue": "🏆", "red": "💀", "draw": "🤝"}
            print(f"  Ep {ep+1:>3}/{n_episodes} "
                  f"{icon.get(winner,'?')} {winner:<4} | "
                  f"kills={ep_kills} | steps={env.get_step_count()}")

    results = {
        "mode":        "heuristic_vs_heuristic",
        "n_episodes":  n_episodes,
        "win_rate":    wins   / n_episodes,
        "loss_rate":   losses / n_episodes,
        "draw_rate":   draws  / n_episodes,
        "kill_per_ep": float(np.mean(kills_list)),
        "mean_ep_len": float(np.mean(lengths_list)),
        "wins": wins, "losses": losses, "draws": draws,
    }
    return results


# ===========================================================================
# Entry Point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MAPPO Eval")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint yolu (.pt). Yoksa heuristic baseline çalışır.")
    p.add_argument("--config",    default="configs/config.yaml")
    p.add_argument("--episodes",  type=int, default=20)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--device",    default="auto")
    p.add_argument("--stochastic", action="store_true",
                   help="Deterministik yerine stokastik politika kullan")
    p.add_argument("--baseline",  action="store_true",
                   help="Sadece heuristic baseline değerlendirmesi yap")
    p.add_argument("--output",    default="logs/eval_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Heuristic baseline ────────────────────────────────────────
    if args.baseline or args.checkpoint is None:
        print("\n[Eval] Heuristic vs Heuristic baseline değerlendirmesi")
        print(f"       {args.episodes} episode, seed={args.seed}\n")
        results = eval_heuristic_baseline(
            config, n_episodes=args.episodes,
            seed=args.seed, verbose=True
        )
        print(f"\n  Win rate  : {results['win_rate']:.1%}")
        print(f"  Loss rate : {results['loss_rate']:.1%}")
        print(f"  Draw rate : {results['draw_rate']:.1%}")
        print(f"  Kill/ep   : {results['kill_per_ep']:.2f}")
        Evaluator.save_results(results, args.output)
        return

    # ── MAPPO değerlendirmesi ─────────────────────────────────────
    if not _TORCH_AVAILABLE:
        print("HATA: PyTorch bulunamadı. pip install torch")
        sys.exit(1)

    print(f"\n[Eval] MAPPO vs Heuristic — {args.episodes} episode")

    ev = Evaluator(config, args.checkpoint, device=args.device)
    results = ev.run(
        n_episodes    = args.episodes,
        deterministic = not args.stochastic,
        seed          = args.seed,
        verbose       = True,
    )

    Evaluator.print_summary(results)
    Evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
