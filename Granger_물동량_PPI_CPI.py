# ============================================================
# 그랜저 인과관계 검정 (Granger Causality Test)
# 여수광양항 물동량 → PPI → CPI 파급경로
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
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

# 로그 변환 → 1차 차분 (정상성 확보)
df_log  = np.log(df)
df_diff = df_log.diff().dropna()

labels = {'cargo': '총물동량', 'ppi': 'PPI(전남)', 'cpi': 'CPI(전국)'}

print("분석 기간:", df_diff.index[0].strftime('%Y-%m'),
      "~", df_diff.index[-1].strftime('%Y-%m'))
print("관측치:", len(df_diff), "개월\n")

# ============================================================
# 1. 단위근 사전 확인 (정상성 검증)
# ============================================================

print("=" * 55)
print("【사전 단위근 검정 — 차분 후 정상성 확인】")
print("=" * 55)
for col in df_diff.columns:
    p = adfuller(df_diff[col], autolag='AIC')[1]
    print(f"  {labels[col]:12s}: ADF p={p:.4f}  "
          f"→ {'✓ 정상' if p < 0.05 else '✗ 비정상'}")

# ============================================================
# 2. 그랜저 인과관계 검정 — 전체 쌍
# ============================================================

MAX_LAG = 6   # 최대 검정 시차

# 검정할 모든 방향 (양방향 포함)
pairs = [
    ('ppi',   'cargo', '물동량 → PPI'),
    ('cpi',   'ppi',   'PPI → CPI'),
    ('cpi',   'cargo', '물동량 → CPI'),
    ('cargo', 'ppi',   'PPI → 물동량  [역방향]'),
    ('ppi',   'cpi',   'CPI → PPI     [역방향]'),
    ('cargo', 'cpi',   'CPI → 물동량  [역방향]'),
]

print("\n" + "=" * 55)
print("【그랜저 인과관계 검정 결과】")
print(f"  (H0: X가 Y를 그랜저 인과하지 않는다, 최대 시차={MAX_LAG})")
print("=" * 55)

summary_rows = []

for caused, cause, direction in pairs:
    data_gc = df_diff[[caused, cause]].dropna()
    test    = grangercausalitytests(data_gc, maxlag=MAX_LAG, verbose=False)

    print(f"\n  {direction}")
    print(f"  {'시차':>4} {'F통계량':>10} {'p값':>10} {'결론':>12}")
    print("  " + "-" * 40)

    best_p, best_lag, best_f = 1.0, 1, 0.0
    for lag in range(1, MAX_LAG + 1):
        f_stat = test[lag][0]['ssr_ftest'][0]
        p_val  = test[lag][0]['ssr_ftest'][1]
        sig    = '★ p<0.05' if p_val < 0.05 else ('△ p<0.10' if p_val < 0.10 else '')
        print(f"  {lag:>4} {f_stat:>10.4f} {p_val:>10.4f} {sig:>12}")
        if p_val < best_p:
            best_p, best_lag, best_f = p_val, lag, f_stat

    conclusion = ('유의 ★' if best_p < 0.05
                  else ('약유의 △' if best_p < 0.10 else '비유의'))
    print(f"  → 최소 p값: {best_p:.4f} (시차 {best_lag})  [{conclusion}]")

    summary_rows.append({
        '방향': direction,
        '원인변수': labels[cause],
        '결과변수': labels[caused],
        '최소p값': round(best_p, 4),
        '최적시차': best_lag,
        'F통계량': round(best_f, 4),
        '유의여부': conclusion,
    })

summary = pd.DataFrame(summary_rows)

# ============================================================
# 3. 최적 시차별 상세 p값 히트맵
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── 왼쪽: p값 히트맵 ─────────────────────────────────────
p_matrix = pd.DataFrame(index=[f'{r["방향"]}' for r in summary_rows],
                         columns=range(1, MAX_LAG + 1))

for i, (caused, cause, direction) in enumerate(pairs):
    data_gc = df_diff[[caused, cause]].dropna()
    test    = grangercausalitytests(data_gc, maxlag=MAX_LAG, verbose=False)
    for lag in range(1, MAX_LAG + 1):
        p_matrix.loc[direction, lag] = test[lag][0]['ssr_ftest'][1]

p_matrix = p_matrix.astype(float)

im = axes[0].imshow(p_matrix.values, aspect='auto', cmap='RdYlGn_r',
                    vmin=0, vmax=0.15)
axes[0].set_xticks(range(MAX_LAG))
axes[0].set_xticklabels([f'시차{i+1}' for i in range(MAX_LAG)], fontsize=9)
axes[0].set_yticks(range(len(pairs)))
axes[0].set_yticklabels([r[2] for r in pairs], fontsize=9)
axes[0].set_title('그랜저 검정 p값 히트맵\n(녹색=유의, 적색=비유의)', fontsize=11)
plt.colorbar(im, ax=axes[0], shrink=0.8, label='p값')

# p값 수치 표시
for i in range(len(pairs)):
    for j in range(MAX_LAG):
        val = p_matrix.values[i, j]
        txt = f'{val:.3f}'
        color = 'white' if val < 0.05 else 'black'
        axes[0].text(j, i, txt, ha='center', va='center',
                     fontsize=7.5, color=color, fontweight='bold' if val < 0.05 else 'normal')

# ── 오른쪽: 인과관계 요약 다이어그램 ────────────────────────
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].axis('off')
axes[1].set_title('그랜저 인과관계 다이어그램', fontsize=11)

# 노드
node_pos = {
    'cargo': (5, 8),
    'ppi':   (2, 4),
    'cpi':   (8, 4),
}
node_colors = {'cargo': '#2196F3', 'ppi': '#E91E63', 'cpi': '#4CAF50'}

for var, (x, y) in node_pos.items():
    circle = plt.Circle((x, y), 1.1, color=node_colors[var], alpha=0.85, zorder=3)
    axes[1].add_patch(circle)
    axes[1].text(x, y, labels[var], ha='center', va='center',
                 fontsize=10, fontweight='bold', color='white', zorder=4)

# 유의한 화살표 그리기
sig_pairs_draw = [(caused, cause, direction)
                  for caused, cause, direction in pairs
                  if summary.loc[summary['방향']==direction, '최소p값'].values[0] < 0.05]

arrow_props = dict(arrowstyle='->', color='black', lw=2)
for caused, cause, direction in pairs:
    x1, y1 = node_pos[cause]
    x2, y2 = node_pos[caused]
    min_p   = summary.loc[summary['방향']==direction, '최소p값'].values[0]
    is_sig  = min_p < 0.05
    is_weak = 0.05 <= min_p < 0.10

    if is_sig or is_weak:
        color  = '#D32F2F' if is_sig else '#FF9800'
        style  = '->' if is_sig else '->'
        lw     = 2.5 if is_sig else 1.5
        ls     = '-' if is_sig else '--'
        label  = f'p={min_p:.3f}{"★" if is_sig else "△"}'

        axes[1].annotate('', xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle=style, color=color,
                                         lw=lw, linestyle=ls,
                                         shrinkA=45, shrinkB=45))
        mx, my = (x1+x2)/2, (y1+y2)/2
        axes[1].text(mx, my + 0.3, label, ha='center', fontsize=8,
                     color=color, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor=color, alpha=0.8))

# 범례
leg_patches = [
    mpatches.Patch(color='#D32F2F', label='유의 (p<0.05) ★'),
    mpatches.Patch(color='#FF9800', label='약유의 (p<0.10) △'),
]
axes[1].legend(handles=leg_patches, loc='lower center', fontsize=9,
               framealpha=0.9)
axes[1].text(5, 1.2, '화살표 없음 = 비유의', ha='center', fontsize=8, color='gray')

plt.suptitle('그랜저 인과관계 분석 — 여수광양항 물동량 · PPI · CPI',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Granger_인과관계.png', dpi=180, bbox_inches='tight')
plt.show()
print("\n저장: Granger_인과관계.png")

# ============================================================
# 4. 결과 요약 출력
# ============================================================

print("\n" + "=" * 55)
print("【최종 요약】")
print("=" * 55)
print(summary[['방향','최소p값','최적시차','유의여부']].to_string(index=False))

# ============================================================
# 5. 엑셀 저장
# ============================================================

with pd.ExcelWriter('Granger_결과.xlsx', engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='요약', index=False)
    p_matrix.to_excel(writer, sheet_name='p값_시차별')

print("\n엑셀 저장: Granger_결과.xlsx")
print("\n✅ 그랜저 인과관계 분석 완료!")
print("\n생성된 파일:")
print("  Granger_인과관계.png  — p값 히트맵 + 다이어그램")
print("  Granger_결과.xlsx     — 수치 테이블")
