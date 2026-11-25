"""반복 실험: 50회독과 300회독의 차이가 우연인지 확인한다.

- 바꾸는 것: 학습 시드(신경망의 처음 상태와 문제 푸는 순서)만 0, 1, 2로 바꾼다.
- 그대로 두는 것: 데이터 분할, 평가용 게임 50판(같은 출발 위치)
- 시드 0 결과는 이미 있으므로(bc_e50, bc_e300) 다시 돌리지 않고 불러온다.
- 소요 시간: GPU 기준 약 10분
"""
import json
import importlib

import numpy as np

from common import OUT_DIR, save_json

bc = importlib.import_module("03_train_bc")  # 파일 이름이 숫자로 시작해서 이렇게 불러온다

EPOCH_SETTINGS = [50, 300]
SEEDS = [0, 1, 2]


def load_or_run(epochs, seed):
    name = f"bc_e{epochs}" if seed == 0 else f"bc_e{epochs}_s{seed}"
    path = OUT_DIR / f"{name}_result.json"
    if path.exists():
        print(f"[{name}] 이미 있는 결과를 불러옴")
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"\n===== {epochs}회독, 시드 {seed} 학습 시작 =====")
    return bc.main(epochs=epochs, seed=seed)


def main():
    summary = {}
    for epochs in EPOCH_SETTINGS:
        rows = [load_or_run(epochs, s) for s in SEEDS]
        cov = [r["mean_max_coverage"] for r in rows]
        suc = [r["success_rate"] for r in rows]
        summary[f"e{epochs}"] = {"seeds": SEEDS, "mean_max_coverage": cov, "success_rate": suc}

    print("\n================ 반복 실험 결과 ================")
    print(f"{'설정':<10}{'시드별 평균 겹침':<28}{'시드별 성공률':<24}{'평균 겹침(평균±편차)':<22}{'성공률(평균±편차)'}")
    for key, s in summary.items():
        cov, suc = np.array(s["mean_max_coverage"]), np.array(s["success_rate"])
        print(f"{key:<10}{str([round(c, 3) for c in cov]):<28}{str([f'{x:.0%}' for x in suc]):<24}"
              f"{cov.mean():.3f} ± {cov.std():.3f}{'':<8}{suc.mean():.0%} ± {suc.std():.0%}")
    save_json(summary, "bc_repeat_summary.json")


if __name__ == "__main__":
    main()
