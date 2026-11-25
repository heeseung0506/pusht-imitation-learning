"""BC와 Diffusion Policy가 같이 쓰는 부분: 데이터, 정규화, 시뮬레이션 평가.

두 정책을 공정하게 비교하려고 데이터 분할, 입력 형식, 평가 방법을 한곳에 모았다.

입력과 출력 형식(두 정책 공통)
- 관측 1개 = T 블록 키포인트 16 + 로봇 끝점 2 = 18차원
- 입력 = 최근 관측 OBS_HORIZON(2)개를 이어 붙인 36차원
- 출력 = 앞으로의 행동 PRED_HORIZON(16)개 (각 2차원)
- 실행 = 예측한 16개 중 앞의 ACT_HORIZON(8)개만 실행하고 다시 예측
"""
import json
from collections import deque
from pathlib import Path

import gymnasium as gym
import gym_pusht  # noqa: F401  (환경 등록)
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "pusht_keypoints"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

OBS_HORIZON = 2
PRED_HORIZON = 16
ACT_HORIZON = 8
OBS_DIM = 18
ACT_DIM = 2
SUCCESS_COVERAGE = 0.95  # gym-pusht의 성공 기준(목표 영역 겹침 95% 이상)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- 데이터
def load_episodes():
    """에피소드별 (관측 [T, 18], 행동 [T, 2]) 목록을 돌려준다."""
    files = sorted(DATA_DIR.glob("data/**/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])
    episodes = []
    for _, g in df.groupby("episode_index"):
        env_state = np.stack(g["observation.environment_state"].to_numpy()).astype(np.float32)
        agent = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(g["action"].to_numpy()).astype(np.float32)
        episodes.append((np.concatenate([env_state, agent], axis=1), action))
    return episodes


def split_episodes(episodes, val_ratio=0.1, seed=0):
    """에피소드 단위로 학습용과 검증용을 나눈다(같은 에피소드가 양쪽에 섞이지 않게)."""
    idx = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_ratio))
    val = [episodes[i] for i in idx[:n_val]]
    train = [episodes[i] for i in idx[n_val:]]
    return train, val


class Normalizer:
    """값을 [-1, 1] 범위로 바꾼다. 학습 데이터의 최솟값, 최댓값만 쓴다."""

    def __init__(self, data):
        self.lo = data.min(axis=0)
        self.hi = data.max(axis=0)
        self.span = np.maximum(self.hi - self.lo, 1e-6)

    def norm(self, x):
        return (x - self.lo) / self.span * 2 - 1

    def unnorm(self, x):
        return (x + 1) / 2 * self.span + self.lo

    def to_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_dict(cls, d):
        n = cls.__new__(cls)
        n.lo = np.array(d["lo"], dtype=np.float32)
        n.hi = np.array(d["hi"], dtype=np.float32)
        n.span = np.maximum(n.hi - n.lo, 1e-6)
        return n


def make_windows(episodes, obs_norm, act_norm):
    """에피소드를 (관측 2개, 행동 16개) 학습 샘플로 자른다.

    에피소드 처음과 끝은 첫 값과 마지막 값을 반복해 채운다(원 논문과 같은 방식).
    """
    xs, ys = [], []
    for obs, act in episodes:
        obs_n, act_n = obs_norm.norm(obs), act_norm.norm(act)
        T = len(obs)
        for t in range(T):
            o_idx = np.clip(np.arange(t - OBS_HORIZON + 1, t + 1), 0, T - 1)
            a_idx = np.clip(np.arange(t, t + PRED_HORIZON), 0, T - 1)
            xs.append(obs_n[o_idx].reshape(-1))
            ys.append(act_n[a_idx])
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32))


# ---------------------------------------------------------------- 평가
def make_env(render=False):
    return gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos",
                    render_mode="rgb_array" if render else None, max_episode_steps=300)


def flat_obs(obs):
    return np.concatenate([obs["environment_state"], obs["agent_pos"]]).astype(np.float32)


def evaluate(policy_fn, n_episodes=50, seed0=1000, record_seed=None):
    """학습에 쓰지 않은 초기 배치(seed 1000부터)로 시뮬레이션을 돌려 성능을 잰다.

    policy_fn(관측 2개 [2, 18]) -> 행동 16개 [16, 2] (원래 좌표)
    지표: 에피소드별 최대 겹침 비율(coverage)의 평균, 성공률(겹침 95% 이상)
    """
    env = make_env(render=record_seed is not None)
    coverages, successes, frames = [], [], []
    for k in range(n_episodes):
        seed = seed0 + k
        obs, _ = env.reset(seed=seed)
        hist = deque([flat_obs(obs)] * OBS_HORIZON, maxlen=OBS_HORIZON)
        best, success, done = 0.0, False, False
        while not done:
            actions = policy_fn(np.stack(hist))
            for a in actions[:ACT_HORIZON]:
                obs, _, terminated, truncated, info = env.step(a)
                hist.append(flat_obs(obs))
                best = max(best, float(info["coverage"]))  # 보상(겹침/0.95)이 아니라 실제 겹침 비율
                success = success or bool(info["is_success"])
                if seed == record_seed:
                    frames.append(env.render())
                done = terminated or truncated
                if done:
                    break
        coverages.append(best)
        successes.append(success)
    env.close()
    result = {
        "episodes": n_episodes,
        "mean_max_coverage": float(np.mean(coverages)),
        "success_rate": float(np.mean(successes)),
        "coverages": [round(c, 4) for c in coverages],
    }
    return result, frames


def evaluate_batch(policy_fn_batch, n_episodes=50, seed0=1000):
    """evaluate와 같은 게임 50판(같은 출발 위치)을 동시에 돌린다.

    Diffusion Policy는 행동 한 번 만드는 데 시간이 오래 걸려서, 50판의 관측을 한꺼번에 묶어
    한 번에 계산한다. 채점 방식은 evaluate와 같다.
    policy_fn_batch(관측 [N, 2, 18]) -> 행동 [N, 16, 2] (원래 좌표)
    """
    envs = [make_env() for _ in range(n_episodes)]
    hists, best, success, active = [], [0.0] * n_episodes, [False] * n_episodes, [True] * n_episodes
    for k, env in enumerate(envs):
        obs, _ = env.reset(seed=seed0 + k)
        hists.append(deque([flat_obs(obs)] * OBS_HORIZON, maxlen=OBS_HORIZON))
    while any(active):
        ids = [k for k in range(n_episodes) if active[k]]
        actions = policy_fn_batch(np.stack([np.stack(hists[k]) for k in ids]))
        for k, acts in zip(ids, actions):
            for a in acts[:ACT_HORIZON]:
                obs, _, terminated, truncated, info = envs[k].step(a)
                hists[k].append(flat_obs(obs))
                best[k] = max(best[k], float(info["coverage"]))
                success[k] = success[k] or bool(info["is_success"])
                if terminated or truncated:
                    active[k] = False
                    break
    for env in envs:
        env.close()
    return {
        "episodes": n_episodes,
        "mean_max_coverage": float(np.mean(best)),
        "success_rate": float(np.mean(success)),
        "coverages": [round(c, 4) for c in best],
    }


def save_json(obj, name):
    (OUT_DIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
