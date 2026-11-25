"""Diffusion Policy: 흐트러진 동작에서 시작해 조금씩 다듬어 '그럴듯한 동작 하나'를 만든다.

쉬운 설명
- 학습: 사람 동작(정답 16걸음)에 일부러 잡음을 섞고, 신경망에게 "섞인 잡음이 무엇인지" 맞히게 한다.
  잡음을 정확히 맞히면, 거꾸로 잡음을 빼서 원래 동작을 되살릴 수 있다.
- 실행: 완전히 무작위인 16걸음에서 시작해 "잡음 빼기"를 여러 번(INFER_STEPS) 반복하면
  점점 자연스러운 동작 하나가 나온다.
- BC와 다른 점: BC는 여러 정답의 '평균' 하나를 외우지만, Diffusion은 매번 여러 정답 중
  '하나'를 골라 끝까지 만든다(왼쪽 아니면 오른쪽, 어정쩡한 가운데가 아니다).

BC와 똑같이 맞춘 것: 데이터 분할, 입력(관측 2개), 출력(행동 16개, 8개 실행), 평가 게임 50판
신경망: 1차원 U-Net(원 논문 Chi et al., 2023의 구조를 작게 줄인 것)
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from common import (ACT_DIM, DEVICE, OBS_DIM, OBS_HORIZON, OUT_DIR, PRED_HORIZON, Normalizer,
                    evaluate_batch, load_episodes, make_windows, save_json, set_seed, split_episodes)

EPOCHS = 500
BATCH = 1024       # 이 노트북에서는 묶음을 크게 해야 빨라서 1024로 키움(BC는 256)
LR = 2e-4
SEED = 0
TRAIN_STEPS = 100    # 학습할 때 잡음을 섞는 단계 수
INFER_STEPS = 100    # 실행할 때 잡음을 빼는 횟수(적을수록 빠르지만 거칠어질 수 있음)
EMA_DECAY = 0.999    # 학습 중 가중치를 천천히 따라가는 '평균 모델'(결과를 안정시킨다)


# ---------------------------------------------------------------- 신경망(1차원 U-Net)
class TimeEmbedding(nn.Module):
    """잡음 단계 번호(0~99)를 신경망이 이해하는 숫자 묶음으로 바꾼다."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        x = t.float()[:, None] * freq[None]
        return torch.cat([x.sin(), x.cos()], dim=-1)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.GroupNorm(8, c_out), nn.Mish())

    def forward(self, x):
        return self.net(x)


class CondResBlock(nn.Module):
    """관측(조건)에 따라 중간 결과의 크기와 위치를 조절하는 블록(FiLM)."""

    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.b1, self.b2 = ConvBlock(c_in, c_out), ConvBlock(c_out, c_out)
        self.cond = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, c_out * 2))
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, cond):
        h = self.b1(x)
        scale, bias = self.cond(cond).unsqueeze(-1).chunk(2, dim=1)
        h = self.b2(h * scale + bias)
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, dims=(64, 128, 256), t_dim=64):
        super().__init__()
        cond_dim = t_dim + OBS_DIM * OBS_HORIZON
        self.t_enc = nn.Sequential(TimeEmbedding(t_dim), nn.Linear(t_dim, t_dim * 4), nn.Mish(),
                                   nn.Linear(t_dim * 4, t_dim))
        in_out = list(zip((ACT_DIM,) + dims[:-1], dims))
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(in_out):
            last = i == len(in_out) - 1
            self.down.append(nn.ModuleList([CondResBlock(a, b, cond_dim), CondResBlock(b, b, cond_dim),
                                            nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        self.mid = nn.ModuleList([CondResBlock(dims[-1], dims[-1], cond_dim) for _ in range(2)])
        self.up = nn.ModuleList()
        for a, b in reversed(in_out[1:]):
            self.up.append(nn.ModuleList([CondResBlock(b * 2, a, cond_dim), CondResBlock(a, a, cond_dim),
                                          nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        self.final = nn.Sequential(ConvBlock(dims[0], dims[0]), nn.Conv1d(dims[0], ACT_DIM, 1))

    def forward(self, noisy_actions, t, obs):
        x = noisy_actions.transpose(1, 2)                  # [B, 2, 16]
        cond = torch.cat([self.t_enc(t), obs], dim=-1)     # 잡음 단계 + 관측
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for r in self.mid:
            x = r(x, cond)
        for r1, r2, up in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up(r2(r1(x, cond), cond))
        return self.final(x).transpose(1, 2)               # [B, 16, 2]: 섞인 잡음 예측


# ---------------------------------------------------------------- 잡음 일정
def make_schedule(n):
    """잡음을 얼마나 섞을지 단계별로 정한다(코사인 일정)."""
    s = 0.008
    x = torch.linspace(0, n, n + 1)
    ac = torch.cos(((x / n) + s) / (1 + s) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - ac[1:] / ac[:-1], 0, 0.999)
    return torch.cumprod(1 - betas, dim=0)  # alpha_bar: 단계 t에서 원래 신호가 남은 비율


@torch.no_grad()
def sample(net, obs, alpha_bar, infer_steps):
    """무작위 16걸음에서 시작해 잡음 빼기를 infer_steps번 반복한다(DDIM 방식)."""
    x = torch.randn(obs.shape[0], PRED_HORIZON, ACT_DIM, device=obs.device)
    ts = torch.linspace(len(alpha_bar) - 1, 0, infer_steps).round().long().tolist()
    for i, t in enumerate(ts):
        ab = alpha_bar[t]
        ab_prev = alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=obs.device)
        eps = net(x, torch.full((obs.shape[0],), t, device=obs.device), obs)
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)   # 지금 추정한 '깨끗한 동작'
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps         # 한 단계 덜 흐린 상태로
    return x


def main(epochs=EPOCHS, seed=SEED, infer_steps=INFER_STEPS, eval_only_from=None):
    run_name = f"dp_e{epochs}" + (f"_s{seed}" if seed else "") + (f"_k{infer_steps}" if infer_steps != 100 else "")
    set_seed(seed)
    episodes = load_episodes()
    train_eps, val_eps = split_episodes(episodes)
    obs_norm = Normalizer(np.concatenate([o for o, _ in train_eps]))
    act_norm = Normalizer(np.concatenate([a for _, a in train_eps]))
    alpha_bar = make_schedule(TRAIN_STEPS).to(DEVICE)

    net = UNet1D().to(DEVICE)
    ema = copy.deepcopy(net).eval()
    t0 = time.time()
    history = []

    if eval_only_from:  # 이미 학습한 모델로 평가만(예: 잡음 빼기 횟수만 바꾸는 실험)
        ema.load_state_dict(torch.load(OUT_DIR / eval_only_from, map_location=DEVICE)["ema"])
    else:
        x_tr, y_tr = make_windows(train_eps, obs_norm, act_norm)
        x_tr, y_tr = x_tr.to(DEVICE), y_tr.to(DEVICE)
        print(f"학습 샘플 {len(x_tr)}개, 파라미터 {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M개, 장치 {DEVICE}")
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-6)
        total_steps = epochs * math.ceil(len(x_tr) / BATCH)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))
        for epoch in range(1, epochs + 1):
            net.train()
            perm = torch.randperm(len(x_tr), device=DEVICE)
            total = 0.0
            for i in range(0, len(x_tr), BATCH):
                b = perm[i:i + BATCH]
                clean, obs = y_tr[b], x_tr[b]
                # ---- 학습의 핵심 4줄 ----
                t = torch.randint(0, TRAIN_STEPS, (len(b),), device=DEVICE)       # 1) 잡음 단계를 무작위로 고르고
                noise = torch.randn_like(clean)                                    # 2) 무작위 잡음을 만들어
                ab = alpha_bar[t][:, None, None]
                noisy = ab.sqrt() * clean + (1 - ab).sqrt() * noise                # 3) 사람 동작에 섞은 뒤
                loss = nn.functional.mse_loss(net(noisy, t, obs), noise)           # 4) 섞은 잡음을 맞히게 한다
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), net.parameters()):
                        pe.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)
                total += loss.item() * len(b)
            history.append({"epoch": epoch, "train_noise_mse": total / len(x_tr)})
            if epoch % 25 == 0 or epoch == 1:
                print(f"epoch {epoch:3d}  잡음 맞히기 오차 {total / len(x_tr):.4f}  ({time.time() - t0:.0f}s)")
        torch.save({"ema": ema.state_dict(), "obs_norm": obs_norm.to_dict(), "act_norm": act_norm.to_dict()},
                   OUT_DIR / f"{run_name}_policy.pt")

    train_seconds = round(time.time() - t0, 1)

    def policy_fn_batch(obs_hist):  # [N, 2, 18] -> [N, 16, 2], 50판을 묶어서 한 번에 계산
        obs = torch.tensor(obs_norm.norm(obs_hist).reshape(len(obs_hist), -1), dtype=torch.float32, device=DEVICE)
        return act_norm.unnorm(sample(ema, obs, alpha_bar, infer_steps).cpu().numpy())

    t1 = time.time()
    result = evaluate_batch(policy_fn_batch, n_episodes=50)
    result.update({"train_seconds": train_seconds, "eval_seconds": round(time.time() - t1, 1),
                   "infer_steps": infer_steps, "history": history})
    save_json(result, f"{run_name}_result.json")
    print(f"[{run_name}] 평균 최대 겹침 {result['mean_max_coverage']:.3f}, 성공률 {result['success_rate']:.0%} "
          f"(50 에피소드, 평가 {result['eval_seconds']:.0f}초)")
    return result


if __name__ == "__main__":
    main()
