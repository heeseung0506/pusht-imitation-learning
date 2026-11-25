"""결과 리포트(A4 HTML)를 만든다. 그림은 파일 안에 넣어(base64) 한 파일로 공유할 수 있게 한다."""
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def img_s(name, cap):
    return img(name, cap).replace("<figure>", '<figure class="small">', 1)


def img(name, cap):
    b64 = base64.b64encode((RES / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'


HTML = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>PushT 모방 학습 미니 프로젝트 리포트</title>
<link href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" rel="stylesheet">
<style>
@page {{ size: A4; margin: 14mm 15mm; }}
:root {{ --ink:#151515; --text:#3d3d3d; --sub:#6b6b6b; --line:#e2e5e9; --tint:#f5f7fa; --point:#1f6fd1; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Pretendard', sans-serif; color: var(--text); font-size: 9.6pt; line-height: 1.6; word-break: keep-all;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
header {{ border-top: 1.2mm solid var(--ink); padding-top: 4mm; margin-bottom: 4mm; }}
.eyebrow {{ font-size: 8.5pt; font-weight: 700; color: var(--point); letter-spacing: .03em; }}
h1 {{ font-size: 17pt; color: var(--ink); line-height: 1.3; margin-top: 1mm; letter-spacing: -.01em; }}
.meta {{ font-size: 8.6pt; color: var(--sub); margin-top: 1.4mm; }}
h2 {{ font-size: 11.5pt; color: var(--ink); margin: 5mm 0 2mm; padding-bottom: 1mm; border-bottom: .3mm solid var(--ink); }}
h2 span {{ color: var(--point); font-size: 8.5pt; margin-right: 2mm; letter-spacing: .03em; }}
p {{ margin-bottom: 1.6mm; }}
ul {{ list-style: none; margin-bottom: 1.6mm; }}
li {{ padding-left: 3.4mm; position: relative; margin-bottom: .6mm; }}
li::before {{ content: ""; position: absolute; left: .6mm; top: .62em; width: 1.2mm; height: 1.2mm; border-radius: 50%; background: var(--point); }}
b {{ color: var(--ink); }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 2.5mm; margin: 1mm 0 2mm; }}
.kpi {{ background: var(--tint); border-radius: 2mm; padding: 2.4mm 3mm; }}
.kpi strong {{ display: block; font-size: 15pt; color: var(--point); line-height: 1.2; }}
.kpi span {{ font-size: 8pt; color: var(--sub); }}
table {{ width: 100%; border-collapse: collapse; margin: 1mm 0 2mm; font-size: 9pt; }}
th, td {{ text-align: left; padding: 1.2mm 2mm; border-bottom: .25mm solid var(--line); vertical-align: top; }}
th {{ background: var(--tint); color: var(--ink); font-weight: 700; border-top: .5mm solid var(--point); }}
td.num, th.num {{ text-align: right; }}
figure {{ margin: 2mm 0 3mm; break-inside: avoid; }}
figure.small img {{ width: 78%; margin: 0 auto; }}
figure img {{ width: 100%; display: block; border: .25mm solid var(--line); border-radius: 1.5mm; }}
figcaption {{ font-size: 8.4pt; color: var(--sub); margin-top: 1mm; }}
.note {{ background: var(--tint); border-left: 1mm solid var(--point); padding: 2mm 3mm; border-radius: 0 1.5mm 1.5mm 0; margin: 1mm 0 2mm; }}
.pb {{ break-before: page; }}
.keep {{ break-inside: avoid; }}
</style></head><body>

<header>
  <p class="eyebrow">MINI PROJECT REPORT | 2026.11</p>
  <h1>사람 시연으로 로봇 정책 학습하기: BC와 Diffusion Policy 비교</h1>
  <p class="meta">한희승 | 개인 미니 프로젝트 | PyTorch, gym-pusht 시뮬레이션, RTX 3050 노트북 GPU</p>
</header>

<div class="kpis">
  <div class="kpi"><strong>58%</strong><span>Diffusion Policy 성공률</span></div>
  <div class="kpi"><strong>18.5%</strong><span>BC 성공률(시드 4개 평균)</span></div>
  <div class="kpi"><strong>4판</strong><span>크게 실패한 판(BC 14~19판)</span></div>
  <div class="kpi"><strong>206개</strong><span>사람 시연 데이터</span></div>
</div>

<h2><span>01</span>요약</h2>
<ul>
  <li><b>목적:</b> 사람이 시범을 보이면 로봇이 따라 배우는 모방 학습을 직접 구현하고, 방법에 따라 실제 과제 성공이 어떻게 달라지는지 확인했습니다.</li>
  <li><b>결과:</b> 같은 데이터와 같은 평가 조건에서 Diffusion Policy가 성공률 58%로 Behavior Cloning(BC) 18.5%보다 높았습니다.</li>
  <li><b>배운 점:</b> 검증 오차가 낮은 모델이 과제를 더 잘하는 것은 아니었습니다. 로봇 정책은 실제 과제 성공률로, 시드를 바꿔 반복해 평가해야 했습니다.</li>
</ul>

<h2><span>02</span>과제와 방법</h2>
<p><b>과제(PushT):</b> 로봇 손끝으로 T자 블록을 밀어 바닥의 목표 자리에 맞추는 2D 시뮬레이션입니다. 블록 겹침이 95% 이상이면 성공입니다. 데이터는 사람이 직접 조작한 시연 206개(25,650 장면)를 썼습니다.</p>
<table>
  <tr><th></th><th>Behavior Cloning (BC)</th><th>Diffusion Policy</th></tr>
  <tr><td><b>쉬운 설명</b></td><td>시연의 "상황과 행동" 짝을 그대로 따라 배운다</td><td>무작위 동작에서 시작해 잡음을 조금씩 지워 자연스러운 동작 하나를 만든다</td></tr>
  <tr><td><b>신경망</b></td><td>MLP, 약 56만 파라미터</td><td>1차원 U-Net, 약 420만 파라미터</td></tr>
  <tr><td><b>예상 약점</b></td><td>정답이 여러 개면 평균을 배워 어정쩡하게 움직인다</td><td>동작 하나에 계산을 100번 반복해 느리다</td></tr>
</table>
<p><b>공통 조건:</b> 시연 단위로 학습 186개와 검증 20개로 나눴습니다. 최근 관측 2개를 입력으로 앞으로의 행동 16개를 예측하고 8개만 실행한 뒤 다시 예측합니다. 평가는 학습에 쓰지 않은 출발 위치 50곳에서 했습니다.</p>

<h2><span>03</span>결과</h2>
{img("fig1_compare.png", "그림 1. 왼쪽: 게임 50판 성공률(점은 BC 시드별 결과). 오른쪽: 판별 최대 겹침 분포. BC는 크게 실패한 판과 아깝게 실패한 판이 많고, Diffusion Policy는 성공 기준 근처에 몰려 있습니다.")}
<table class="keep">
  <tr><th>지표</th><th class="num">BC 300회독 (시드 4개 평균)</th><th class="num">Diffusion Policy (시드 1개)</th></tr>
  <tr><td>성공률 (겹침 95% 이상)</td><td class="num">18.5%</td><td class="num"><b>58%</b></td></tr>
  <tr><td>평균 최대 겹침</td><td class="num">64.4%</td><td class="num"><b>88.3%</b></td></tr>
  <tr><td>크게 실패한 판 (겹침 50% 미만, 50판 중)</td><td class="num">14~19판</td><td class="num"><b>4판</b></td></tr>
  <tr><td>학습 시간 (RTX 3050 노트북)</td><td class="num">약 2분</td><td class="num">약 48분</td></tr>
</table>

{img("fig3_rollout.png", "그림 2. 같은 출발 위치에서 두 정책 비교. BC는 블록을 목표와 다른 곳으로 밀어 놓고 멈췄고(겹침 0%), Diffusion Policy는 블록을 돌려 목표에 맞췄습니다(겹침 96%).")}

<h2><span>04</span>직접 한 실험</h2>
<p><b>실험 1. 검증 오차가 가장 낮을 때 멈추면 더 잘할까?</b> 과적합이 덜한 50회독이 더 좋을 거라 예상했지만, 시드 0에서 성공률은 50회독 6%, 300회독 18%로 반대였습니다. 검증 오차는 사람 행동과 평균적으로 얼마나 비슷한지를 볼 뿐, 마지막에 블록을 정확히 맞추는 미세 동작은 잘 반영하지 못한다고 해석했습니다.</p>
{img_s("fig2_bc_overfit.png", "그림 3. BC 학습 곡선. 검증 오차는 약 30~50회독에서 가장 낮았지만, 실제 성공률은 300회독이 더 높은 경향을 보였습니다.")}
<p class="keep"><b>실험 2. 그 차이가 우연은 아닐까?</b> 학습 시드를 바꿔 반복했습니다.</p>
<table class="keep">
  <tr><th>설정</th><th>시드별 성공률</th><th class="num">평균 성공률</th><th class="num">평균 최대 겹침</th></tr>
  <tr><td>BC 300회독</td><td>18%, 22%, 26%, 8%</td><td class="num">18.5%</td><td class="num">64.4%</td></tr>
  <tr><td>BC 50회독</td><td>6%, 16%, 12%</td><td class="num">11.3%</td><td class="num">59.9%</td></tr>
</table>
<p>처음 시드 3개에서는 300회독이 세 번 모두 나았습니다. 그런데 시드 4를 더하자 성공률이 8%로 가장 낮아졌습니다. 이 모델은 평균 겹침이 68.7%로 가장 높았지만, 25판이 85~95%에서 아깝게 실패했습니다. 그래서 결론을 <b>"300회독이 나은 경향은 있지만 확정할 수 없다"</b>로 고쳤고, 목표 근처까지 미는 능력과 마지막에 맞추는 능력을 따로 봐야 한다고 정리했습니다.</p>

<h2><span>05</span>한계와 다음 단계</h2>
<ul>
  <li><b>Diffusion Policy는 시드 1개 결과입니다.</b> BC가 시드에 따라 8~26%로 흔들렸으므로 반복 실험이 필요합니다.</li>
  <li><b>조건이 완전히 같지 않습니다.</b> Diffusion Policy가 모델이 더 크고 더 오래 학습했습니다(500회독 대 300회독). 차이가 전부 방법 덕분이라고 단정할 수 없습니다.</li>
  <li><b>평가가 50판이고 2D 시뮬레이션입니다.</b> 실제 로봇의 접촉, 지연, 센서 잡음은 다루지 않았습니다.</li>
  <li><b>다음 단계:</b> Diffusion Policy 시드 반복, 잡음 지우기 횟수를 줄였을 때 속도와 성능 비교, 마찰과 질량을 바꿨을 때의 성능 변화를 확인하려고 합니다.</li>
</ul>
<div class="note">코드 초안 작성에 생성형 AI(Claude)를 활용했습니다. 학습 횟수와 시드를 바꾼 실험 실행, 예상과 결과 비교, 결론 수정은 직접 했습니다. 참고: Chi et al., Diffusion Policy, RSS 2023 | 데이터: lerobot/pusht_keypoints</div>
</body></html>
"""

(ROOT / "PushT_미니프로젝트_리포트.html").write_text(HTML, encoding="utf-8")
print("saved")
