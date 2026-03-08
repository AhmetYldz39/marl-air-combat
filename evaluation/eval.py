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
        eval_cfg = config.get("eval", {})
        self.env  = DogfightEnv(config)

        # MAPPO Actor yükle
        from training.train_mappo import MAPPOActor
        norm          = Normalizer(config)
        n_tm          = self.env.n_per_team - 1
        n_en          = self.env.n_per_team
        self.obs_dim  = norm.obs_dim(n_tm, n_en)
        action_dim    = self.env.action_dim
        hidden        = int(config["training"].get("hidden_dim", 256))

        self.actor = MAPPOActor(self.obs_dim, action_dim,
                                 hidden=hidden).to(self.device)
        self._load_actor(checkpoint_path)
        self.actor.eval()

        # Heuristic rakip (Red takımı)
        self.train_ids = self.env.blue_ids
        self.opp_ids   = self.env.red_ids
        team_map = {aid: ("blue" if "blue" in aid else "red")
                    for aid in self.env.agent_ids}
        self.opp_policy = MultiHeuristicPolicy(
            config, self.env.agent_ids, team_map
        )

        print(f"[Eval] Device     : {self.device}")
        print(f"[Eval] Checkpoint : {checkpoint_path}")
        print(f"[Eval] obs_dim    : {self.obs_dim}")

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

            ep_reward  = {aid: 0.0 for aid in self.train_ids}
            ep_kills   = 0
            ep_oob     = 0
            done       = {"__all__": False}

            while not done["__all__"]:
                # MAPPO aksiyonları
                actions = self._get_actions(obs_dict, deterministic)

                # Heuristic aksiyonları
                state_dict  = self.env.get_all_states()
                opp_actions = self.opp_policy.act(state_dict)

                action_dict = {**actions, **opp_actions}
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
        actions = {}
        with torch.no_grad():
            for aid in self.train_ids:
                obs_t = torch.FloatTensor(
                    obs_dict[aid]
                ).unsqueeze(0).to(self.device)
                raw, _ = self.actor.act(obs_t, deterministic=deterministic)
                from training.train_mappo import MAPPOActor
                squashed     = MAPPOActor.squash(raw.squeeze(0))
                actions[aid] = squashed.cpu().numpy()
        return actions

    def _load_actor(self, path: str):
        """Checkpoint'ten sadece actor ağırlıklarını yükle."""
        ckpt = torch.load(path, map_location=self.device)
        if "actor" in ckpt:
            self.actor.load_state_dict(ckpt["actor"])
            ep   = ckpt.get("episode",     "?")
            step = ckpt.get("global_step", "?")
            print(f"[Eval] Actor yüklendi — ep={ep}, step={step}")
        else:
            # Doğrudan state_dict olarak kaydedilmiş
            self.actor.load_state_dict(ckpt)
            print("[Eval] Actor yüklendi (raw state_dict)")

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
