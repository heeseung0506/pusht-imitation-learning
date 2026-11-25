"""Behavior Cloning(BC): 시연의 '관측에서 행동으로'를 신경망이 그대로 따라 배운다.

- 모델: MLP(은닉층 3개, 512 유닛)
- 입력: 최근 관측 2개(36차원), 출력: 앞으로의 행동 16개(32차원)
- 손실: 시연 행동과의 평균제곱오차(MSE)
- 한계(예상): 같은 상황에서 사람이 왼쪽, 오른쪽 어느 쪽으로도 밀 수 있으면
  MSE는 두 답의 평균(가운데)을 배워 어중간하게 움직일 수 있다(멀티모달 문제).
"""
import time

import numpy as np
import torch
import torch.nn as nn

from common import (ACT_DIM, DEVICE, OBS_DIM, OBS_HORIZON, OUT_DIR, PRED_HORIZON, Normalizer,
                    evaluate, load_episodes, make_windows, save_json, set_seed, split_episodes)

EPOCHS = 300
BATCH = 256
LR = 1e-3
SEED = 4


class MLPPolicy(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM * OBS_HORIZON, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACT_DIM * PRED_HORIZON),
        )

    def forward(self, x):
        return self.net(x).view(-1, PRED_HORIZON, ACT_DIM)


def main(epochs=EPOCHS, seed=SEED):
    """학습 횟수(epochs)와 시드(seed)를 받아 학습하고 평가 결과를 돌려준다."""
    run_name = f"bc_e{epochs}" if seed == 0 else f"bc_e{epochs}_s{seed}"  # 결과 파일 이름
    set_seed(seed)
    episodes = load_episodes()
    train_eps, val_eps = split_episodes(episodes)
    obs_norm = Normalizer(np.concatenate([o for o, _ in train_eps]))
    act_norm = Normalizer(np.concatenate([a for _, a in train_eps]))
    x_tr, y_tr = make_windows(train_eps, obs_norm, act_norm)
    x_va, y_va = make_windows(val_eps, obs_norm, act_norm)
    print(f"학습 샘플 {len(x_tr)}개(에피소드 {len(train_eps)}), 검증 샘플 {len(x_va)}개(에피소드 {len(val_eps)}), 장치 {DEVICE}")

    model = MLPPolicy().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x_tr, y_tr, x_va, y_va = (t.to(DEVICE) for t in (x_tr, y_tr, x_va, y_va))

    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_tr), device=DEVICE)
        total = 0.0
        for i in range(0, len(x_tr), BATCH):
            b = perm[i:i + BATCH]
            loss = nn.functional.mse_loss(model(x_tr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(x_va), y_va).item()
        history.append({"epoch": epoch, "train_mse": total / len(x_tr), "val_mse": val})
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train {total / len(x_tr):.4f}  val {val:.4f}  ({time.time() - t0:.0f}s)")

    torch.save({"model": model.state_dict(), "obs_norm": obs_norm.to_dict(), "act_norm": act_norm.to_dict()},
               OUT_DIR / f"{run_name}_policy.pt")

    @torch.no_grad()
    def policy_fn(obs_hist):
        x = torch.tensor(obs_norm.norm(obs_hist).reshape(1, -1), dtype=torch.float32, device=DEVICE)
        return act_norm.unnorm(model(x)[0].cpu().numpy())

    result, _ = evaluate(policy_fn, n_episodes=50)
    result["train_seconds"] = round(time.time() - t0, 1)
    result["history"] = history
    save_json(result, f"{run_name}_result.json")
    print(f"[{run_name}] 평균 최대 겹침 {result['mean_max_coverage']:.3f}, 성공률 {result['success_rate']:.0%} (50 에피소드)")
    return result


if __name__ == "__main__":
    main()
