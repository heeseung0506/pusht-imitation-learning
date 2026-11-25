"""결과 정리용 그림 3장을 만든다(results/ 폴더).

1. fig1_compare.png: BC(시드 4개)와 Diffusion Policy의 성공률, 판별 겹침 분포
2. fig2_bc_overfit.png: BC 학습 곡선(학습 오차와 검증 오차, 50회독과 300회독 비교)
3. fig3_rollout.png: 같은 출발 위치에서 두 정책이 블록을 미는 장면(처음, 중간, 마지막)
"""
import importlib
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import DEVICE, OUT_DIR, Normalizer, evaluate

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
BLUE, GRAY, DARK = "#1f6fd1", "#9aa3ad", "#222222"

BC_RUNS = ["bc_e300", "bc_e300_s1", "bc_e300_s2", "bc_e300_s4"]
DP_RUN = "dp_e500"


def load(name):
    return json.loads((OUT_DIR / f"{name}_result.json").read_text(encoding="utf-8"))


def fig_compare():
    bc = [load(n) for n in BC_RUNS]
    dp = load(DP_RUN)
    bc_succ = [r["success_rate"] * 100 for r in bc]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})

    a1.bar([0], [np.mean(bc_succ)], color=GRAY, width=0.55)
    a1.scatter(np.zeros(len(bc_succ)) + np.linspace(-0.12, 0.12, len(bc_succ)), bc_succ, color=DARK, s=18, zorder=3,
               label="BC 시드별")
    a1.bar([1], [dp["success_rate"] * 100], color=BLUE, width=0.55)
    a1.set_xticks([0, 1], ["BC\n(시드 4개 평균)", "Diffusion Policy\n(시드 1개)"])
    a1.set_ylabel("성공률 (%)  [겹침 95% 이상]")
    a1.set_ylim(0, 70)
    for x, v, y in [(0, np.mean(bc_succ), max(bc_succ) + 3), (1, dp["success_rate"] * 100, dp["success_rate"] * 100 + 2)]:
        a1.text(x, y, f"{v:.1f}%" if x == 0 else f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")
    a1.legend(frameon=False, loc="upper left")
    a1.set_title("게임 50판 성공률", fontsize=12)

    bc_cov = np.concatenate([r["coverages"] for r in bc])
    dp_cov = np.array(dp["coverages"])
    bins = np.linspace(0, 1, 21)
    a2.hist(bc_cov, bins=bins, color=GRAY, alpha=0.9, weights=np.ones_like(bc_cov) / len(bc_cov) * 100,
            label="BC (200판)")
    a2.hist(dp_cov, bins=bins, color=BLUE, alpha=0.75, weights=np.ones_like(dp_cov) / len(dp_cov) * 100,
            label="Diffusion Policy (50판)")
    a2.axvline(0.95, color=DARK, ls="--", lw=1)
    a2.text(0.94, a2.get_ylim()[1] * 0.92, "성공 기준 95%", ha="right", fontsize=9)
    a2.set_xlabel("판별 최대 겹침 비율")
    a2.set_ylabel("판 비율 (%)")
    a2.legend(frameon=False, loc="upper left")
    a2.set_title("판별 결과 분포: BC는 '크게 실패'와 '아깝게 실패'가 많음", fontsize=12)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_compare.png", dpi=160)


def fig_bc_overfit():
    h = load("bc_e300")["history"]
    ep = [x["epoch"] for x in h]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ep, [x["train_mse"] for x in h], color=GRAY, label="학습 오차(연습 문제)")
    ax.plot(ep, [x["val_mse"] for x in h], color=BLUE, label="검증 오차(모의고사)")
    best = min(h, key=lambda x: x["val_mse"])
    ax.scatter([best["epoch"]], [best["val_mse"]], color=BLUE, zorder=3)
    ax.annotate(f"검증 오차 최저 구간(약 {best['epoch']}~50회독)\n그러나 실제 성공률은 300회독이 더 높은 경향",
                (best["epoch"], best["val_mse"]), xytext=(110, 0.032), fontsize=9,
                arrowprops={"arrowstyle": "-", "color": DARK, "lw": 0.8})
    ax.set_xlabel("회독(epoch)")
    ax.set_ylabel("평균제곱오차 (작을수록 좋음)")
    ax.legend(frameon=False, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("BC 학습 곡선: 검증 오차가 낮다고 해서 과제 성공률이 높지는 않다", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_bc_overfit.png", dpi=160)


def bc_policy():
    bc = importlib.import_module("03_train_bc")
    ck = torch.load(OUT_DIR / "bc_e300_policy.pt", map_location=DEVICE)
    m = bc.MLPPolicy().to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    on, an = Normalizer.from_dict(ck["obs_norm"]), Normalizer.from_dict(ck["act_norm"])

    @torch.no_grad()
    def f(h):
        x = torch.tensor(on.norm(h).reshape(1, -1), dtype=torch.float32, device=DEVICE)
        return an.unnorm(m(x)[0].cpu().numpy())
    return f


def dp_policy():
    dp = importlib.import_module("05_train_diffusion")
    ck = torch.load(OUT_DIR / f"{DP_RUN}_policy.pt", map_location=DEVICE)
    net = dp.UNet1D().to(DEVICE)
    net.load_state_dict(ck["ema"])
    net.eval()
    on, an = Normalizer.from_dict(ck["obs_norm"]), Normalizer.from_dict(ck["act_norm"])
    ab = dp.make_schedule(dp.TRAIN_STEPS).to(DEVICE)

    def f(h):
        x = torch.tensor(on.norm(h).reshape(1, -1), dtype=torch.float32, device=DEVICE)
        return an.unnorm(dp.sample(net, x, ab, dp.INFER_STEPS)[0].cpu().numpy())
    return f


def fig_rollout():
    bc_cov = np.array(load("bc_e300")["coverages"])
    dp_cov = np.array(load(DP_RUN)["coverages"])
    # Diffusion은 성공하고 BC는 실패한 판 중 BC 겹침이 가장 낮은 판을 고른다
    cand = [k for k in range(len(dp_cov)) if dp_cov[k] > 0.95 and bc_cov[k] < 0.95]
    k = min(cand, key=lambda i: bc_cov[i])
    seed = 1000 + k
    rows = []
    for name, fn in [("BC", bc_policy()), ("Diffusion Policy", dp_policy())]:
        res, frames = evaluate(fn, n_episodes=1, seed0=seed, record_seed=seed)
        idx = np.linspace(0, len(frames) - 1, 4).round().astype(int)
        rows.append((name, res["mean_max_coverage"], [frames[i] for i in idx], idx))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for r, (name, cov, frs, idx) in enumerate(rows):
        for c, (fr, i) in enumerate(zip(frs, idx)):
            ax = axes[r, c]
            ax.imshow(fr)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{i}번째 행동" if c else "처음", fontsize=10)
        axes[r, 0].set_ylabel(f"{name}\n최대 겹침 {cov:.0%}", fontsize=11)
    fig.suptitle(f"같은 출발 위치(시드 {seed})에서 두 정책 비교: 초록 영역이 목표, 파란 점이 로봇 손끝", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_rollout.png", dpi=130)
    print(f"장면 비교 판: 시드 {seed} (BC {bc_cov[k]:.0%}, Diffusion {dp_cov[k]:.0%})")


if __name__ == "__main__":
    fig_compare()
    fig_bc_overfit()
    fig_rollout()
    print("그림 3장 저장 완료:", OUT_DIR)
