# ============================================================
# Local Projections (LP) 분석 — Jordà (2005)
# 여수광양항 물동량 충격 → PPI → CPI 동태적 반응
# VAR-IRF의 대안: 모형 오설정에 강건한 충격반응 추정
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

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

# 로그 변환 (변화율 해석 가능)
df_log = np.log(df)

labels = {'cargo': '총물동량', 'ppi': 'PPI(전남)', 'cpi': 'CPI(전국)'}

print("분석 기간:", df_log.index[0].strftime('%Y-%m'),
      "~", df_log.index[-1].strftime('%Y-%m'))
print("관측치:", len(df_log), "개월\n")

# ============================================================
# 1. Local Projections 핵심 함수
# ============================================================

def local_projection(data, shock_var, response_var,
                     H=18, n_lags=2, control_vars=None,
                     cumulative=False):
    """
    Jordà (2005) Local Projections 충격반응 추정.

    각 예측시계 h에 대해 다음 회귀식을 개별 추정:
        y_{t+h} = α_h + β_h · shock_t
                  + Σ γ · (control lags) + ε_{t+h}

    β_h 들의 수열이 곧 충격반응함수(IRF).

    Parameters
    ----------
    data         : DataFrame (시계열)
    shock_var    : 충격 변수 (예: 'cargo')
    response_var : 반응 변수 (예: 'ppi')
    H            : 예측 시계 (개월)
    n_lags       : 통제 시차 수
    control_vars : 추가 통제변수 리스트 (None이면 전체 변수 사용)
    cumulative   : True면 누적반응 (수준), False면 시점별 반응

    Returns
    -------
    betas, se, lo90, hi90, lo95, hi95  (길이 H+1 배열)
    """
    if control_vars is None:
        control_vars = list(data.columns)

    betas, ses = [], []

    for h in range(H + 1):
        # 종속변수: h기 후 반응변수
        if cumulative:
            # 누적: y_{t+h} - y_{t-1}
            y = data[response_var].shift(-h) - data[response_var].shift(1)
        else:
            # 시점별: y_{t+h}
            y = data[response_var].shift(-h)

        # 설명변수 구성
        X_parts = [data[shock_var].rename('shock')]   # 핵심: 당기 충격

        # 통제변수 시차 (lagged controls)
        for var in control_vars:
            for lag in range(1, n_lags + 1):
                X_parts.append(data[var].shift(lag).rename(f'{var}_L{lag}'))

        X = pd.concat(X_parts, axis=1)
        reg = pd.concat([y.rename('y'), X], axis=1).dropna()

        if len(reg) < len(X.columns) + 2:
            betas.append(np.nan); ses.append(np.nan)
            continue

        yv = reg['y']
        Xv = sm.add_constant(reg.drop(columns='y'))

        # Newey-West (HAC) 표준오차 — LP에서 표준
        model = sm.OLS(yv, Xv).fit(
            cov_type='HAC', cov_kwds={'maxlags': h + 1})

        betas.append(model.params['shock'])
        ses.append(model.bse['shock'])

    betas = np.array(betas)
    ses   = np.array(ses)

    return {
        'beta': betas,
        'se':   ses,
        'lo90': betas - 1.645 * ses,
        'hi90': betas + 1.645 * ses,
        'lo95': betas - 1.96 * ses,
        'hi95': betas + 1.96 * ses,
    }


# ============================================================
# 2. 핵심 파급경로 LP 추정
# ============================================================

H = 18  # 충격반응 추적 개월

paths = [
    ('cargo', 'ppi', '물동량 충격 → PPI 반응',  '#E91E63'),
    ('ppi',   'cpi', 'PPI 충격 → CPI 반응',    '#4CAF50'),
    ('cargo', 'cpi', '물동량 충격 → CPI 반응',  '#9C27B0'),
]

results = {}
for shock, resp, title, color in paths:
    results[(shock, resp)] = local_projection(
        df_log, shock_var=shock, response_var=resp,
        H=H, n_lags=2, cumulative=False)

# ============================================================
# 3. 그림 ① 시점별 LP 충격반응
# ============================================================

x = np.arange(H + 1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for k, (shock, resp, title, color) in enumerate(paths):
    ax  = axes[k]
    r   = results[(shock, resp)]

    ax.plot(x, r['beta'], color=color, linewidth=2.5, marker='o', markersize=4)
    # 90% / 95% 신뢰밴드 (이중)
    ax.fill_between(x, r['lo95'], r['hi95'], alpha=0.12, color=color)
    ax.fill_between(x, r['lo90'], r['hi90'], alpha=0.22, color=color)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # 유의 구간 음영 (95% 기준 0 미포함)
    sig = (r['lo95'] > 0) | (r['hi95'] < 0)
    for t in range(len(sig) - 1):
        if sig[t]:
            ax.axvspan(t, t + 1, alpha=0.08, color=color)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('개월 후 (h)', fontsize=9)
    ax.set_ylabel('반응 (로그)', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

fig.suptitle('Local Projections 충격반응 — 여수광양항 물동량 → 물가\n'
             '(진한 밴드 90%, 옅은 밴드 95%)',
             fontsize=13, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('LP_시점별반응.png', dpi=180, bbox_inches='tight')
plt.show()
print("저장: LP_시점별반응.png")

# ============================================================
# 4. 그림 ② 누적 LP 충격반응
# ============================================================

results_cum = {}
for shock, resp, title, color in paths:
    results_cum[(shock, resp)] = local_projection(
        df_log, shock_var=shock, response_var=resp,
        H=H, n_lags=2, cumulative=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for k, (shock, resp, title, color) in enumerate(paths):
    ax  = axes[k]
    r   = results_cum[(shock, resp)]

    ax.plot(x, r['beta'], color=color, linewidth=2.5, marker='s', markersize=4)
    ax.fill_between(x, r['lo95'], r['hi95'], alpha=0.12, color=color)
    ax.fill_between(x, r['lo90'], r['hi90'], alpha=0.22, color=color)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    ax.set_title(f'[누적] {title}', fontsize=11, fontweight='bold')
    ax.set_xlabel('개월 후 (h)', fontsize=9)
    ax.set_ylabel('누적 반응 (로그)', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

fig.suptitle('Local Projections 누적 충격반응', fontsize=13,
             fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('LP_누적반응.png', dpi=180, bbox_inches='tight')
plt.show()
print("저장: LP_누적반응.png")

# ============================================================
# 5. 그림 ③ LP vs VAR-IRF 비교 (선택)
# ============================================================

try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    var_res = VAR(df_log).fit(2, trend='c')
    var_irf = var_res.irf(periods=H)
    names   = var_res.names
    vidx    = {nm: i for i, nm in enumerate(names)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k, (shock, resp, title, color) in enumerate(paths):
        ax = axes[k]
        # LP
        r = results[(shock, resp)]
        ax.plot(x, r['beta'], color=color, linewidth=2.5,
                marker='o', markersize=3, label='LP')
        ax.fill_between(x, r['lo95'], r['hi95'], alpha=0.12, color=color)
        # VAR-IRF
        var_resp = var_irf.orth_irfs[:, vidx[resp], vidx[shock]]
        ax.plot(x, var_resp, color='gray', linewidth=2,
                linestyle='--', marker='^', markersize=3, label='VAR-IRF')

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('개월 후', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('LP vs VAR-IRF 충격반응 비교', fontsize=13,
                 fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('LP_vs_VAR비교.png', dpi=180, bbox_inches='tight')
    plt.show()
    print("저장: LP_vs_VAR비교.png")
except Exception as e:
    print(f"VAR 비교 그림 생략: {e}")

# ============================================================
# 6. 수치 테이블 출력
# ============================================================

print("\n" + "=" * 60)
print("【LP 충격반응 수치 테이블】")
print("=" * 60)

for shock, resp, title, _ in paths:
    r = results[(shock, resp)]
    print(f"\n{title}")
    print(f"{'h':>3} {'β(반응)':>10} {'표준오차':>10} "
          f"{'하한95':>10} {'상한95':>10} {'유의':>5}")
    print("-" * 52)
    for h in range(H + 1):
        sig = '★' if (r['lo95'][h] > 0 or r['hi95'][h] < 0) else ''
        print(f"{h:>3} {r['beta'][h]:>10.5f} {r['se'][h]:>10.5f} "
              f"{r['lo95'][h]:>10.5f} {r['hi95'][h]:>10.5f} {sig:>5}")

# ============================================================
# 7. 엑셀 저장
# ============================================================

with pd.ExcelWriter('LP_결과.xlsx', engine='openpyxl') as writer:
    for shock, resp, title, _ in paths:
        r = results[(shock, resp)]
        sheet = pd.DataFrame({
            'h(개월)': x,
            'β': r['beta'], '표준오차': r['se'],
            '하한90': r['lo90'], '상한90': r['hi90'],
            '하한95': r['lo95'], '상한95': r['hi95'],
            '유의': ['★' if (r['lo95'][h]>0 or r['hi95'][h]<0) else '' for h in x],
        })
        sheet.to_excel(writer, sheet_name=f'{title[:20]}', index=False)
        # 누적
        rc = results_cum[(shock, resp)]
        sheet_c = pd.DataFrame({
            'h(개월)': x, '누적β': rc['beta'], '표준오차': rc['se'],
            '하한95': rc['lo95'], '상한95': rc['hi95'],
        })
        sheet_c.to_excel(writer, sheet_name=f'누적_{title[:16]}', index=False)

print("\n엑셀 저장: LP_결과.xlsx")
print("\n✅ Local Projections 분석 완료!")
print("\n생성된 파일:")
print("  LP_시점별반응.png  — 시점별 LP-IRF")
print("  LP_누적반응.png    — 누적 LP-IRF")
print("  LP_vs_VAR비교.png  — LP와 VAR 비교")
print("  LP_결과.xlsx       — 수치 테이블")
