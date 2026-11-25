"""gym-pusht 시뮬레이션이 데이터와 같은 관측 형식으로 도는지 확인한다.

무작위 행동으로 한 에피소드를 돌려 관측 차원, 보상, 종료 조건을 출력한다.
"""
import gymnasium as gym
import gym_pusht  # noqa: F401  (환경 등록)
import numpy as np


def main():
    env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos", max_episode_steps=300)
    obs, info = env.reset(seed=0)
    print("관측 키:", {k: np.asarray(v).shape for k, v in obs.items()})
    print("행동 공간:", env.action_space)
    total, best = 0.0, 0.0
    for t in range(300):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        best = max(best, reward)
        if terminated or truncated:
            break
    print(f"{t + 1} 스텝, 최대 보상(목표 영역 겹침 비율) {best:.3f}, 성공 {info.get('is_success')}")
    env.close()


if __name__ == "__main__":
    main()
