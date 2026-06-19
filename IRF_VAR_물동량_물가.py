# ============================================================
# 충격반응함수 (IRF) 분석 — 여수광양항 물동량 → PPI → CPI
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# 0. 데이터 로드 (물동량 + PPI + CPI)
# ============================================================

# ── 물동량 ──────────────────────────────────────────────────
cargo = pd.read_csv('여수광양항_물동량_wide_VAR.csv')
cargo['date'] = pd.to_datetime(cargo['year_month'])
cargo = cargo.set_index('date').sort_index()
cargo_s = cargo.loc['2018-01':'2021-07', '총물동량_합계_RT']

# ── PPI / CPI ── 아래를 실제 ECOS 값으로 교체 ──────────────
# ECOS 다운로드: https://ecos.bok.or.kr
#   PPI: 통계 2.2 생산자물가지수 → 지역별(전남)
#   CPI: 통계 2.5 소비자물가지수 → 전국
#
# 예시 (더미 — 실제값으로 교체 필수):
n = len(cargo_s)
np.random.seed(42)
dummy_ppi = 100 + np.cumsum(np.random.randn(n) * 0.3)
dummy_cpi = 100 + np.cumsum(np.random.randn(n) * 0.2)

df = pd.DataFrame({
    'cargo': cargo_s.values,       # 총물동량 (R/T)
    'ppi':   dummy_ppi,            # ← ECOS PPI 값으로 교체
    'cpi':   dummy_cpi,            # ← ECOS CPI 값으로 교체
}, index=cargo_s.index)

# 로그 변환 (선택: 분산 안정화)
df_log = np.log(df)

print("분석 데이터 미리보기:")
print(df_log.head())
print(f"\n기간: {df_log.index[0].strftime('%Y-%m')} ~ {df_log.index[-1].strftime('%Y-%m')}")
print(f"관측치: {len(df_log)}개월")

# ============================================================
# 1. VAR 추정 (IRF 전 필수)
# ============================================================

# 최적 시차 선택
model     = VAR(df_log)
lag_order = model.select_order(maxlags=6)
best_lag  = max(lag_order.bic, 1)
print(f"\n선택 시차 (BIC): {best_lag}")

# 모형 적합
var_result = model.fit(best_lag, trend='c')

# ============================================================
# 2. IRF 계산
# ============================================================

PERIODS   = 18       # 충격반응 추적 기간 (개월)
ALPHA     = 0.05     # 신뢰구간 수준 (95%)
ORTH      = True     # True = 직교화 충격 (Cholesky)
                     # False = 비직교화 충격

irf = var_result.irf(periods=PERIODS)

# 변수명 → 인덱스 매핑
names  = var_result.names                     # ['cargo', 'ppi', 'cpi']
idx    = {n: i for i, n in enumerate(names)}
x      = np.arange(PERIODS + 1)              # 0 ~ PERIODS 개월

# 충격반응 행렬 및 표준오차
irfs = irf.orth_irfs if ORTH else irf.irfs   # shape: (T+1, n, n)
sems = irf.orth_stderr() if ORTH else irf.stderr()  # shape: (T+1, n, n)

def get_ci(row_var, col_var, z=1.96):
    """response of row_var to impulse in col_var"""
    ri, ci = idx[row_var], idx[col_var]
    mean  = irfs[:, ri, ci]
    se    = sems[:, ri, ci]
    return mean, mean - z * se, mean + z * se

# ============================================================
# 3. 그림 ① 전체 IRF 매트릭스 (3×3)
# ============================================================

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle('충격반응함수 (Orthogonalized IRF) — 전체 매트릭스',
             fontsize=14, fontweight='bold', y=1.01)

labels_kr = {'cargo': '총물동량', 'ppi': 'PPI(전남)', 'cpi': 'CPI(전국)'}
colors    = {'cargo': '#2196F3', 'ppi': '#E91E63', 'cpi': '#4CAF50'}

for ri, response in enumerate(names):
    for ci_v, impulse in enumerate(names):
        ax = axes[ri, ci_v]
        mean, lo, hi = get_ci(response, impulse)

        ax.plot(x, mean, color=colors[impulse], linewidth=2)
        ax.fill_between(x, lo, hi, alpha=0.15, color=colors[impulse])
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        if ri == 0:
            ax.set_title(f'{labels_kr[impulse]} 충격', fontsize=10, fontweight='bold')
        if ci_v == 0:
            ax.set_ylabel(f'{labels_kr[response]} 반응', fontsize=9)
        ax.set_xlabel('개월', fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

        # 유의 구간 음영
        sig = (lo > 0) | (hi < 0)
        for t in range(len(sig) - 1):
            if sig[t]:
                ax.axvspan(t, t + 1, alpha=0.08, color=colors[impulse])

plt.tight_layout()
plt.savefig('IRF_전체매트릭스.png', dpi=180, bbox_inches='tight')
plt.show()
print("저장: IRF_전체매트릭스.png")

# ============================================================
# 4. 그림 ② 핵심 파급경로 (물동량 → PPI → CPI)
# ============================================================

fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

paths = [
    ('ppi', 'cargo', '물동량 충격 → PPI 반응', '#E91E63'),
    ('cpi', 'ppi',   'PPI 충격 → CPI 반응',   '#4CAF50'),
    ('cpi', 'cargo', '물동량 충격 → CPI 반응', '#9C27B0'),
]

for k, (resp, imp, title, color) in enumerate(paths):
    ax   = fig.add_subplot(gs[k])
    mean, lo, hi = get_ci(resp, imp)

    ax.plot(x, mean, color=color, linewidth=2.5, label='IRF')
    ax.fill_between(x, lo, hi, alpha=0.18, color=color, label='95% CI')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # 유의 구간 강조
    sig = (lo > 0) | (hi < 0)
    for t in range(len(sig) - 1):
        if sig[t]:
            ax.axvspan(t, t + 1, alpha=0.12, color=color)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('개월 후', fontsize=9)
    ax.set_ylabel('반응', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

    # 최대 반응 시점 표시
    peak_t = np.argmax(np.abs(mean))
    ax.annotate(f'최대반응\n{peak_t}개월',
                xy=(peak_t, mean[peak_t]),
                xytext=(peak_t + 2, mean[peak_t] * 0.8),
                arrowprops=dict(arrowstyle='->', color=color),
                fontsize=8, color=color)

fig.suptitle('여수광양항 물동량 → PPI → CPI 파급경로 IRF',
             fontsize=13, fontweight='bold', y=1.03)
plt.savefig('IRF_핵심파급경로.png', dpi=180, bbox_inches='tight')
plt.show()
print("저장: IRF_핵심파급경로.png")

# ============================================================
# 5. 그림 ③ 누적 IRF (Cumulative IRF)
# ============================================================

cum_irfs = irf.orth_cum_effects if ORTH else irf.cum_effects  # (T+1, n, n)
# 누적 표준오차 (bootstrap 또는 analytic)
try:
    cum_se = irf.cum_effect_stderr(orth=ORTH)
except:
    cum_se = np.cumsum(sems, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('누적 충격반응함수 (Cumulative IRF)', fontsize=13,
             fontweight='bold', y=1.02)

for k, (resp, imp, title, color) in enumerate(paths):
    ax   = axes[k]
    ri, ci_v = idx[resp], idx[imp]
    mean  = cum_irfs[:, ri, ci_v]
    se    = cum_se[:, ri, ci_v]
    lo, hi = mean - 1.96 * se, mean + 1.96 * se

    ax.plot(x, mean, color=color, linewidth=2.5)
    ax.fill_between(x, lo, hi, alpha=0.18, color=color)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('개월 후', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('IRF_누적.png', dpi=180, bbox_inches='tight')
plt.show()
print("저장: IRF_누적.png")

# ============================================================
# 6. 수치 테이블 출력 (보고서용)
# ============================================================

print("\n" + "=" * 60)
print("【IRF 수치 테이블 — 핵심 파급경로】")
print("=" * 60)

for resp, imp, title, _ in paths:
    mean, lo, hi = get_ci(resp, imp)
    print(f"\n{title}")
    print(f"{'기간':>4} {'IRF':>10} {'하한(95%)':>12} {'상한(95%)':>12} {'유의':>6}")
    print("-" * 48)
    for t in range(PERIODS + 1):
        sig_mark = '★' if (lo[t] > 0 or hi[t] < 0) else ''
        print(f"{t:>4} {mean[t]:>10.5f} {lo[t]:>12.5f} {hi[t]:>12.5f} {sig_mark:>6}")

# ============================================================
# 7. 엑셀 저장 (보고서 첨부용)
# ============================================================

with pd.ExcelWriter('IRF_결과.xlsx', engine='openpyxl') as writer:
    for resp, imp, title, _ in paths:
        mean, lo, hi = get_ci(resp, imp)
        sheet = pd.DataFrame({
            '기간(개월)': x,
            'IRF': mean,
            '하한(95%)': lo,
            '상한(95%)': hi,
            '유의여부': ['★' if (lo[t]>0 or hi[t]<0) else '' for t in x],
        })
        sheet.to_excel(writer, sheet_name=title[:20], index=False)

    # 전체 매트릭스
    for resp in names:
        for imp in names:
            mean, lo, hi = get_ci(resp, imp)
            sheet = pd.DataFrame({
                '기간': x, 'IRF': mean, '하한': lo, '상한': hi,
            })
            sheet.to_excel(writer, sheet_name=f'{imp[:4]}→{resp[:4]}', index=False)

print("\n엑셀 저장: IRF_결과.xlsx")
print("\n✅ IRF 분석 완료!")
print("\n생성된 파일:")
print("  IRF_전체매트릭스.png  — 3×3 전체 반응")
print("  IRF_핵심파급경로.png  — 물동량→PPI→CPI")
print("  IRF_누적.png          — 누적 충격반응")
print("  IRF_결과.xlsx         — 수치 테이블")
