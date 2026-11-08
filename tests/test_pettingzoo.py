"""
test_pettingzoo.py
==================
T3.1 — DogfightParallelEnv PettingZoo API conformance testleri.

Çalıştırma:
    pytest tests/test_pettingzoo.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from pettingzoo.test import parallel_api_test
    _HAS_PETTINGZOO = True
except ImportError:
    _HAS_PETTINGZOO = False

pytestmark = pytest.mark.skipif(
    not _HAS_PETTINGZOO,
    reason="pettingzoo not installed — pip install pettingzoo gymnasium",
)


@pytest.fixture(scope="module")
def config():
    with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def pz_env(config):
    from aircombat_bench.envs.pettingzoo_env import DogfightParallelEnv
    env = DogfightParallelEnv(config, curriculum_phase=4)
    yield env
    env.close()


# ===========================================================================
# 1. PettingZoo conformance
# ===========================================================================

class TestPettingZooConformance:
    """pettingzoo.test.parallel_api_test geçmeli."""

    def test_parallel_api_test(self, config):
        """Resmi PettingZoo API test suite'i geçmeli."""
        from aircombat_bench.envs.pettingzoo_env import DogfightParallelEnv
        env = DogfightParallelEnv(config, curriculum_phase=4)
        parallel_api_test(env, num_cycles=20)
        env.close()

    def test_parallel_api_test_scenario_1v1(self, config):
        """1v1 senaryosu da API testini geçmeli."""
        from aircombat_bench.envs.pettingzoo_env import DogfightParallelEnv
        env = DogfightParallelEnv(config, curriculum_phase=1)
        parallel_api_test(env, num_cycles=20)
        env.close()


# ===========================================================================
# 2. Temel özellikler
# ===========================================================================

class TestDogfightParallelEnvBasics:
    """Temel API davranışı testleri."""

    def test_possible_agents(self, pz_env):
        """possible_agents yalnızca blue ajan içermeli."""
        pa = pz_env.possible_agents
        assert len(pa) == 2
        assert all("blue" in a for a in pa)
        assert "blue_0" in pa
        assert "blue_1" in pa

    def test_observation_space(self, pz_env):
        """observation_space(agent) Box(50,) döndürmeli."""
        import gymnasium as gym
        for agent in pz_env.possible_agents:
            sp = pz_env.observation_space(agent)
            assert isinstance(sp, gym.spaces.Box)
            assert sp.shape == (50,)
            assert sp.dtype == np.float32

    def test_action_space(self, pz_env):
        """action_space(agent) Box(5,) döndürmeli."""
        import gymnasium as gym
        for agent in pz_env.possible_agents:
            sp = pz_env.action_space(agent)
            assert isinstance(sp, gym.spaces.Box)
            assert sp.shape == (5,)

    def test_action_space_bounds(self, pz_env):
        """Aksiyon alt/üst sınırları doğru olmalı."""
        sp = pz_env.action_space("blue_0")
        np.testing.assert_allclose(sp.low[:3],  [-1., -1., -1.])
        np.testing.assert_allclose(sp.low[3:],  [0., 0.])
        np.testing.assert_allclose(sp.high[:],   [1., 1., 1., 1., 1.])

    def test_reset_returns_obs_and_info(self, pz_env):
        """reset() (obs_dict, info_dict) döndürmeli."""
        result = pz_env.reset(seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_obs_shape(self, pz_env):
        """reset() 50D float32 obs döndürmeli."""
        obs, _ = pz_env.reset(seed=42)
        for agent in pz_env.possible_agents:
            assert agent in obs
            assert obs[agent].shape == (50,)
            assert obs[agent].dtype == np.float32

    def test_reset_obs_in_range(self, pz_env):
        """reset() obs [-1, 1] aralığında olmalı."""
        obs, _ = pz_env.reset(seed=42)
        for agent in pz_env.possible_agents:
            assert np.all(obs[agent] >= -1.0 - 1e-5), f"{agent} obs min underflow"
            assert np.all(obs[agent] <=  1.0 + 1e-5), f"{agent} obs max overflow"

    def test_reset_seeds_env(self, pz_env):
        """reset(seed) deterministik başlangıç üretmeli."""
        obs_a, _ = pz_env.reset(seed=42)
        obs_b, _ = pz_env.reset(seed=42)
        for agent in pz_env.possible_agents:
            np.testing.assert_allclose(
                obs_a[agent], obs_b[agent],
                err_msg=f"{agent} obs differs between two reset(seed=42) calls"
            )

    def test_agents_populated_after_reset(self, pz_env):
        """reset() sonrası agents = possible_agents olmalı."""
        pz_env.reset(seed=0)
        assert set(pz_env.agents) == set(pz_env.possible_agents)

    def test_num_agents_after_reset(self, pz_env):
        """num_agents == 2 olmalı."""
        pz_env.reset(seed=0)
        assert pz_env.num_agents == 2
        assert pz_env.max_num_agents == 2

    def test_step_returns_5_tuple(self, pz_env):
        """step() (obs, rew, term, trunc, info) 5-demeti döndürmeli."""
        pz_env.reset(seed=0)
        actions = {a: pz_env.action_space(a).sample() for a in pz_env.agents}
        result = pz_env.step(actions)
        assert len(result) == 5
        obs, rew, term, trunc, info = result
        for d in (obs, rew, term, trunc, info):
            assert isinstance(d, dict)

    def test_step_obs_shape(self, pz_env):
        """step() obs şekli reset() ile aynı olmalı."""
        pz_env.reset(seed=0)
        actions = {a: pz_env.action_space(a).sample() for a in pz_env.agents}
        obs, _, _, _, _ = pz_env.step(actions)
        for agent in pz_env.possible_agents:
            assert obs[agent].shape == (50,)

    def test_step_rew_are_floats(self, pz_env):
        """step() reward'ları float olmalı."""
        pz_env.reset(seed=0)
        actions = {a: pz_env.action_space(a).sample() for a in pz_env.agents}
        _, rew, _, _, _ = pz_env.step(actions)
        for agent in pz_env.possible_agents:
            assert isinstance(rew[agent], float)

    def test_step_term_trunc_are_bool(self, pz_env):
        """step() terminations ve truncations bool olmalı."""
        pz_env.reset(seed=0)
        actions = {a: pz_env.action_space(a).sample() for a in pz_env.agents}
        _, _, term, trunc, _ = pz_env.step(actions)
        for agent in pz_env.possible_agents:
            assert isinstance(term[agent], bool)
            assert isinstance(trunc[agent], bool)

    def test_state_shape(self, pz_env):
        """state() 2×50 = 100D birleştirilmiş gözlem döndürmeli."""
        pz_env.reset(seed=0)
        s = pz_env.state()
        assert s.shape == (100,)

    def test_from_yaml_classmethod(self):
        """from_yaml() env oluşturmalı ve reset() çalışmalı."""
        from aircombat_bench.envs.pettingzoo_env import DogfightParallelEnv
        env = DogfightParallelEnv.from_yaml(ROOT / "configs/config.yaml")
        obs, _ = env.reset(seed=0)
        assert len(obs) == 2
        env.close()

    def test_make_env_factory(self):
        """make_env('2v2_normal') çalışmalı."""
        from aircombat_bench.envs.pettingzoo_env import make_env
        env = make_env("2v2_normal")
        obs, _ = env.reset(seed=0)
        assert len(obs) == 2
        env.close()

    def test_make_env_invalid_scenario(self):
        """Geçersiz senaryo ValueError üretmeli."""
        from aircombat_bench.envs.pettingzoo_env import make_env
        with pytest.raises(ValueError, match="Bilinmeyen senaryo"):
            make_env("invalid_scenario")

    def test_load_scenario_all_four(self):
        """load_scenario() tüm 4 senaryo için geçerli dict döndürmeli."""
        from aircombat_bench.envs.pettingzoo_env import load_scenario
        expected = {
            "1v1_close":   {"phase": 1, "blue": 1, "red": 1},
            "1v1_dynamic": {"phase": 2, "blue": 1, "red": 1},
            "1v1_normal":  {"phase": 3, "blue": 1, "red": 1},
            "2v2_normal":  {"phase": 4, "blue": 2, "red": 2},
        }
        for name, exp in expected.items():
            sc = load_scenario(name)
            assert sc["curriculum_phase"] == exp["phase"], f"{name}: bad phase"
            assert sc["teams"]["blue"]["n_agents"] == exp["blue"]
            assert sc["teams"]["red"]["n_agents"] == exp["red"]
            assert sc["obs_dim"] == 50
            assert sc["action_dim"] == 5
            assert "schema_version" in sc
            assert "spawn" in sc

    def test_load_scenario_2v2_has_baselines(self):
        """2v2_normal senaryo YAML'ı 6 baseline kaydı içermeli."""
        from aircombat_bench.envs.pettingzoo_env import load_scenario
        sc = load_scenario("2v2_normal")
        assert sc["benchmark"]["is_headline"] is True
        baselines = sc["benchmark"]["baselines"]
        assert len(baselines) == 6
        algorithms = {b["algorithm"] for b in baselines}
        assert "facmac_omr" in algorithms
        assert "mappo" in algorithms

    def test_load_scenario_invalid_raises(self):
        """Geçersiz senaryo ValueError üretmeli."""
        from aircombat_bench.envs.pettingzoo_env import load_scenario
        with pytest.raises(ValueError, match="Bilinmeyen senaryo"):
            load_scenario("nonexistent")

    def test_native_env_accessible(self, pz_env):
        """pz_env.native DogfightEnv döndürmeli."""
        from aircombat_bench.envs.dogfight_env import DogfightEnv
        assert isinstance(pz_env.native, DogfightEnv)

    def test_close_no_error(self, pz_env):
        """close() hata vermemeli."""
        pz_env.reset(seed=0)
        pz_env.close()   # İki kez de çağrılabilmeli
        pz_env.close()

    def test_render_no_error(self, pz_env):
        """render() hata vermemeli (headless)."""
        pz_env.reset(seed=0)
        pz_env.render()   # hiçbir şey yapmaz ama hata vermemeli


# ===========================================================================
# 3. Episode bütünlüğü
# ===========================================================================

class TestEpisodeIntegrity:
    """Tam episode rollout testleri."""

    def test_full_episode_no_crash(self, pz_env, config):
        """max_steps=50 ile tam episode crash ve NaN olmadan tamamlanmalı."""
        # Hızlı test için kısa episode
        from aircombat_bench.envs.pettingzoo_env import DogfightParallelEnv
        cfg = {**config}
        cfg = dict(config)
        cfg["env"] = dict(config["env"])
        cfg["env"]["max_steps"] = 50

        env = DogfightParallelEnv(cfg, curriculum_phase=4)
        obs, _ = env.reset(seed=7)
        rng    = np.random.default_rng(7)
        step_n = 0

        while env.agents:
            actions = {
                a: rng.uniform(env.action_space(a).low,
                               env.action_space(a).high).astype(np.float32)
                for a in env.agents
            }
            obs, rew, term, trunc, info = env.step(actions)
            step_n += 1

            # NaN kontrolü
            for a in env.possible_agents:
                assert not np.any(np.isnan(obs[a])), f"NaN in {a} obs at step {step_n}"

        assert step_n > 0, "En az bir adım atılmalı"
        assert step_n <= 50, f"max_steps=50 aşıldı: {step_n}"
        env.close()

    def test_deterministic_episode(self, pz_env):
        """Aynı seed → aynı episode trajektorisi."""
        from aircombat_bench.envs.pettingzoo_env import DogfightParallelEnv
        import yaml
        with open(ROOT / "configs/config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["env"]["max_steps"] = 30

        rng = np.random.default_rng(0)
        def run_episode(env):
            obs, _ = env.reset(seed=42)
            rewards_sum = {a: 0.0 for a in env.possible_agents}
            while env.agents:
                actions = {a: rng.uniform(env.action_space(a).low,
                                          env.action_space(a).high).astype(np.float32)
                           for a in env.agents}
                _, rew, _, _, _ = env.step(actions)
                for a in env.possible_agents:
                    rewards_sum[a] += rew.get(a, 0.0)
            return rewards_sum

        rng = np.random.default_rng(0)
        env1 = DogfightParallelEnv(cfg)
        r1 = run_episode(env1); env1.close()

        rng = np.random.default_rng(0)
        env2 = DogfightParallelEnv(cfg)
        r2 = run_episode(env2); env2.close()

        for a in ['blue_0', 'blue_1']:
            assert abs(r1[a] - r2[a]) < 1e-6, (
                f"{a} reward differs: {r1[a]:.4f} vs {r2[a]:.4f}"
            )
