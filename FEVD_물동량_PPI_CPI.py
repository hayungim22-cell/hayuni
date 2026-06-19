# ============================================================
# 예측오차 분산분해 (FEVD)
# 여수광양항 물동량 · PPI · CPI VAR 모형
# "각 변수의 변동을 어떤 충격이 몇 % 설명하는가"
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.vector_ar.var_model import VAR

# 한글 폰트
plt.rcParams['axes.unicode_minus'] = False
for font in ['AppleGothic', 'NanumGothic', 'Malgun Gothic', 'DejaVu Sans']:
    try:
        plt.rcParams['font.family'] = font
        break
    except:
        continue

# ============================================================
# 0. 데이터 로드
# ============================================================

cargo = pd.read_csv('여수광양항_물동량_wide_VAR.csv')
cargo['date'] = pd.to_datetime(cargo['year_month'])
cargo = cargo.set_index('date').sort_index()
cargo_s = cargo.loc['2018-01':'2021-07', '총물동량_합계_RT']

# ── PPI / CPI 교체 필요 ────────────────────────────────────
n = len(cargo_s)
np.random.seed(42)
dummy_ppi = 100 + np.cumsum(np.random.randn(n) * 0.3)
dummy_cpi = 100 + np.cumsum(np.random.randn(n) * 0.2)

df = pd.DataFrame({
    'cargo': cargo_s.values,
    'ppi':   dummy_ppi,      # ← ECOS 실제값으로 교체
    'cpi':   dummy_cpi,      # ← ECOS 실제값으로 교체
}, index=cargo_s.index)

# 로그 변환
df_log = np.log(df)

labels = {'cargo': '총물동량', 'ppi': 'PPI(전남)', 'cpi': 'CPI(전국)'}
colors = {'cargo': '#2196F3', 'ppi': '#E91E63', 'cpi': '#4CAF50'}

print("분석 기간:", df_log.index[0].strftime('%Y-%m'),
      "~", df_log.index[-1].strftime('%Y-%m'))
print("관측치:", len(df_log), "개월\n")

# ============================================================
# 1. VAR 추정 (FEVD 전 필수)
# ============================================================

# 변수 순서 = Cholesky 식별 순서 (외생 → 내생)
#   cargo(물동량) → ppi → cpi
# FEVD는 이 순서에 민감하므로 이론적 인과순서로 배치

model     = VAR(df_log)
lag_order = model.select_order(maxlags=6)
best_lag  = max(lag_order.bic, 1)
print(f"선택 시차 (BIC): {best_lag}")

var_result = model.fit(best_lag, trend='c')
names = var_result.names

# ============================================================
# 2. FEVD 계산
# ============================================================

PERIODS = 18   # 분해 기간 (개월)

fevd = var_result.fevd(periods=PERIODS)

# decomp shape: (n_variables, periods, n_shocks)
# decomp[i, h, j] = 변수 i의 h기 예측오차 분산 중 충격 j의 기여도
decomp = fevd.decomp   # 0~1 비율

# ============================================================
# 3. 그림 ① 스택 영역 그래프 (변수별)
# ============================================================

x = np.arange(1, PERIODS + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for v_idx, target in enumerate(names):
    ax = axes[v_idx]
    bottom = np.zeros(PERIODS)

    for s_idx, shock in enumerate(names):
        contrib = decomp[v_idx, :, s_idx] * 100  # %
        ax.fill_between(x, bottom, bottom + contrib,
                        color=colors[shock], alpha=0.8,
                        label=f'{labels[shock]} 충격')
        bottom += contrib

    ax.set_title(f'{labels[target]} 분산분해', fontsize=11, fontweight='bold')
    ax.set_xlabel('예측시계 (개월)', fontsize=9)
    ax.set_ylabel('기여율 (%)', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(1, PERIODS)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

fig.suptitle('예측오차 분산분해 (FEVD) — 변수별 충격 기여도',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('FEVD_스택영역.png', dpi=180, bbox_inches='tight')
plt.show()
print("저장: FEVD_스택영역.png")

# ============================================================
# 4. 그림 ② CPI 집중 분석 (보고서 핵심)
# ============================================================

cpi_idx = names.index('cpi')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ── 왼쪽: CPI 분산분해 스택 ───────────────────────────────
bottom = np.zeros(PERIODS)
for s_idx, shock in enumerate(names):
    contrib = decomp[cpi_idx, :, s_idx] * 100
    ax1.fill_between(x, bottom, bottom + contrib,
                     color=colors[shock], alpha=0.8,
                     label=f'{labels[shock]} 충격')
    bottom += contrib

ax1.set_title('전국 CPI 변동의 충격별 기여도', fontsize=11, fontweight='bold')
ax1.set_xlabel('예측시계 (개월)', fontsize=9)
ax1.set_ylabel('기여율 (%)', fontsize=9)
ax1.set_ylim(0, 100)
ax1.set_xlim(1, PERIODS)
ax1.legend(fontsize=9, loc='center right')
ax1.grid(True, alpha=0.2)

# ── 오른쪽: 물동량·PPI 기여도만 선그래프 ──────────────────
for s_idx, shock in enumerate(names):
    if shock == 'cpi':
        continue
    contrib = decomp[cpi_idx, :, s_idx] * 100
    ax2.plot(x, contrib, color=colors[shock], linewidth=2.5,
             marker='o', markersize=4, label=f'{labels[shock]} → CPI')

ax2.set_title('물동량·PPI 충격의 CPI 설명력', fontsize=11, fontweight='bold')
ax2.set_xlabel('예측시계 (개월)', fontsize=9)
ax2.set_ylabel('기여율 (%)', fontsize=9)
ax2.set_xlim(1, PERIODS)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)

fig.suptitle('전국 CPI 분산분해 — 항만 물동량의 물가 설명력',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('FEVD_CPI집중.png', dpi=180, bbox_inches='tight')
plt.show()
print("저장: FEVD_CPI집중.png")

# ============================================================
# 5. 수치 테이블 출력
# ============================================================

print("\n" + "=" * 60)
print("【FEVD 수치 테이블】 — 단위: %")
print("=" * 60)

# 주요 시점만 (1, 3, 6, 12, 18개월)
key_periods = [p for p in [1, 3, 6, 12, 18] if p <= PERIODS]

for v_idx, target in enumerate(names):
    print(f"\n■ {labels[target]} 예측오차 분산분해")
    header = f"{'시계':>5}"
    for shock in names:
        header += f"{labels[shock]:>12}"
    print(header)
    print("-" * (5 + 12 * len(names)))
    for h in key_periods:
        row = f"{h:>4}월"
        for s_idx in range(len(names)):
            row += f"{decomp[v_idx, h-1, s_idx]*100:>11.2f}%"
        print(row)

# ============================================================
# 6. 엑셀 저장
# ============================================================

with pd.ExcelWriter('FEVD_결과.xlsx', engine='openpyxl') as writer:
    for v_idx, target in enumerate(names):
        sheet = pd.DataFrame(
            decomp[v_idx, :, :] * 100,
            columns=[f'{labels[s]}_충격' for s in names],
            index=range(1, PERIODS + 1),
        )
        sheet.index.name = '예측시계(개월)'
        sheet.to_excel(writer, sheet_name=f'{labels[target]}_분산분해')

print("\n엑셀 저장: FEVD_결과.xlsx")

# ============================================================
# 7. 핵심 결론 자동 요약
# ============================================================

print("\n" + "=" * 60)
print("【핵심 결론 요약】")
print("=" * 60)

# 18개월(또는 최종) 시점 CPI 분산분해
final = PERIODS - 1
cargo_to_cpi = decomp[cpi_idx, final, names.index('cargo')] * 100
ppi_to_cpi   = decomp[cpi_idx, final, names.index('ppi')] * 100
cpi_self     = decomp[cpi_idx, final, names.index('cpi')] * 100

print(f"\n{PERIODS}개월 후 전국 CPI 변동 설명력:")
print(f"  • 물동량 충격 기여:  {cargo_to_cpi:.2f}%")
print(f"  • PPI 충격 기여:     {ppi_to_cpi:.2f}%")
print(f"  • CPI 자체 충격:     {cpi_self:.2f}%")
print(f"\n→ 항만 물동량이 전국 물가 변동의 약 {cargo_to_cpi:.1f}%를 설명")

print("\n✅ FEVD 분석 완료!")
print("\n생성된 파일:")
print("  FEVD_스택영역.png  — 변수별 분산분해")
print("  FEVD_CPI집중.png   — CPI 집중 분석")
print("  FEVD_결과.xlsx     — 수치 테이블")
