"""PushT 사람 시연 데이터(lerobot/pusht_keypoints)를 내려받고 구조를 확인한다.

- 영상 없이 상태(state) 기반 데이터만 받는다: data/*.parquet, meta/*
- observation.state: 로봇 끝점(agent) 위치 x, y (2차원)
- observation.environment_state: T 블록 키포인트 8개의 x, y (16차원)
- action: 다음 목표 위치 x, y (2차원)
"""
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "pusht_keypoints"


def main():
    path = snapshot_download(
        repo_id="lerobot/pusht_keypoints",
        repo_type="dataset",
        local_dir=DATA_DIR,
        allow_patterns=["data/*", "meta/*"],
    )
    files = sorted(Path(path).glob("data/**/*.parquet"))
    print(f"parquet 파일 {len(files)}개")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print("열 목록:", list(df.columns))
    print("전체 프레임 수:", len(df))
    print("에피소드 수:", df["episode_index"].nunique())
    lengths = df.groupby("episode_index").size()
    print(f"에피소드 길이: 평균 {lengths.mean():.0f}, 최소 {lengths.min()}, 최대 {lengths.max()}")
    row = df.iloc[0]
    for col in ["observation.state", "observation.environment_state", "action"]:
        if col in df.columns:
            print(f"{col}: 차원 {len(row[col])}, 예시 {[round(float(v), 1) for v in row[col]][:6]}")


if __name__ == "__main__":
    main()
