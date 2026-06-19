# ============================================================
# VAR 모형 분석: 여수광양항 물동량 → 전남동부 PPI → 전국 CPI
# 작성: 한국은행 보고서용
# 필요 패키지: pandas, numpy, statsmodels, matplotlib, scipy
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# 한글 폰트 설정 (Mac: AppleGothic / Linux: NanumGothic)
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['font.family']  # test
except:
    try:
        plt.rcParams['font.family'] = 'NanumGothic'
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# 1. 데이터 로드 및 전처리
# ============================================================

def load_data(
    cargo_path='여수광양항_물동량_wide_VAR.csv',
    ppi_path='전남동부_PPI.csv',       # ← ECOS에서 다운받은 PPI 파일 경로
    cpi_path='전국_CPI.csv',            # ← ECOS에서 다운받은 CPI 파일 경로
):
    """
    데이터 로드. PPI/CPI 파일은 ECOS(한국은행 경제통계시스템)에서 다운로드.
    
    ECOS 다운로드 경로:
    - PPI: 통계검색 → 2.2 생산자물가지수 → 품목별(지역별) → 전라남도(또는 동부권)
    - CPI: 통계검색 → 2.5 소비자물가지수 → 전국
    
    CSV 형식 가정: year_month(YYYY-MM), value 두 컬럼
    """
    # 물동량 로드
    cargo = pd.read_csv(cargo_path)
    cargo['date'] = pd.to_datetime(cargo['year_month'])
    cargo = cargo.set_index('date').sort_index()

    # PPI 로드 (형식에 맞게 수정 필요)
    ppi = pd.read_csv(ppi_path)
    ppi['date'] = pd.to_datetime(ppi['year_month'])
    ppi = ppi.set_index('date').sort_index()

    # CPI 로드
    cpi = pd.read_csv(cpi_path)
    cpi['date'] = pd.to_datetime(cpi['year_month'])
    cpi = cpi.set_index('date').sort_index()

    return cargo, ppi, cpi


def build_panel(cargo, ppi, cpi,
                start='2018-01', end='2021-07',
                cargo_col='총물동량_합계_RT'):
    """
    분석용 패널 데이터 구성.
    기본 기간: 2018-01 ~ 2021-07 (RT+TEU+PPI+CPI 모두 완전한 구간)
    """
    df = pd.DataFrame({
        'cargo': cargo.loc[start:end, cargo_col],
        'ppi':   ppi.loc[start:end, 'value'],
        'cpi':   cpi.loc[start:end, 'value'],
    }).dropna()

    print(f"분석 기간: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')}")
    print(f"관측치: {len(df)}개월\n")
    print(df.describe().round(2))
    return df


# ============================================================
# 2. 기초 통계 및 시각화
# ============================================================

def plot_series(df):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = {
        'cargo': '총물동량 (R/T)',
        'ppi':   '전남동부 PPI',
        'cpi':   '전국 CPI',
    }
    for ax, (col, label) in zip(axes, labels.items()):
        ax.plot(df.index, df[col], color='steelblue', linewidth=1.5)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_title('여수광양항 물동량 · PPI · CPI 시계열', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('01_시계열_원자료.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("그림 저장: 01_시계열_원자료.png")


# ============================================================
# 3. 단위근 검정 (ADF + KPSS)
# ============================================================

def unit_root_test(df):
    """
    ADF: H0 = 단위근 있음 (비정상)  → p<0.05 이면 정상
    KPSS: H0 = 정상            → p<0.05 이면 비정상
    """
    print("=" * 60)
    print("【단위근 검정】")
    print("=" * 60)

    results = {}
    for col in df.columns:
        # 수준값 (level)
        adf_lv  = adfuller(df[col], autolag='AIC')
        kpss_lv = kpss(df[col], regression='c', nlags='auto')

        # 1차 차분
        diff = df[col].diff().dropna()
        adf_d1  = adfuller(diff, autolag='AIC')
        kpss_d1 = kpss(diff, regression='c', nlags='auto')

        print(f"\n── {col} ──────────────────────")
        print(f"  [수준값]  ADF p={adf_lv[1]:.4f}  KPSS p={kpss_lv[1]:.4f}  "
              f"→ {'정상' if adf_lv[1]<0.05 else '비정상'}")
        print(f"  [1차차분] ADF p={adf_d1[1]:.4f}  KPSS p={kpss_d1[1]:.4f}  "
              f"→ {'정상' if adf_d1[1]<0.05 else '비정상'}")

        results[col] = {
            'adf_level': adf_lv[1], 'kpss_level': kpss_lv[1],
            'adf_diff':  adf_d1[1], 'kpss_diff':  kpss_d1[1],
            'integrated': adf_lv[1] > 0.05,  # True면 비정상 → 차분 필요
        }

    # 통합 차수 판단
    n_integrated = sum(v['integrated'] for v in results.values())
    print(f"\n→ I(1) 변수 개수: {n_integrated}/{len(df.columns)}")
    print("→ 공적분 검정 필요 여부:", "예" if n_integrated >= 2 else "아니오 (수준값 VAR 사용 가능)")
    return results, n_integrated


# ============================================================
# 4. 공적분 검정 (Johansen)
# ============================================================

def johansen_test(df, det_order=0, k_ar_diff=1):
    """
    모든 변수가 I(1)일 때 장기균형 존재 여부 확인.
    det_order: -1(상수없음), 0(상수), 1(추세)
    """
    print("\n" + "=" * 60)
    print("【요한슨 공적분 검정】")
    print("=" * 60)

    result = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)
    n = df.shape[1]

    print(f"\n{'r':>5} {'Trace 통계량':>15} {'95% 임계값':>12} {'결론':>10}")
    print("-" * 50)
    for i in range(n):
        trace = result.lr1[i]
        crit  = result.cvt[i, 1]   # 95%
        sign  = "공적분 있음 ✓" if trace > crit else "기각 못함"
        print(f"{i:>5} {trace:>15.4f} {crit:>12.4f} {sign:>10}")

    n_coint = sum(result.lr1[i] > result.cvt[i, 1] for i in range(n))
    print(f"\n→ 공적분 관계 수: {n_coint}")
    if n_coint >= 1:
        print("→ VECM 또는 수준값 VAR 사용 권고")
    else:
        print("→ 1차 차분 VAR 사용 권고")
    return n_coint


# ============================================================
# 5. 최적 시차 선택
# ============================================================

def select_lag(df, max_lag=6):
    print("\n" + "=" * 60)
    print("【최적 시차 선택 (AIC / BIC / HQIC)】")
    print("=" * 60)

    model = VAR(df)
    lag_result = model.select_order(maxlags=max_lag)
    print(lag_result.summary())

    aic_lag = lag_result.aic
    bic_lag = lag_result.bic
    hqic_lag = lag_result.hqic
    print(f"\nAIC 최적 시차: {aic_lag}  BIC: {bic_lag}  HQIC: {hqic_lag}")
    # 보통 BIC(정보량 패널티 강함) 우선, 단 최소 1 보장
    best = max(bic_lag, 1)
    print(f"→ 선택 시차: {best} (BIC 기준)")
    return best


# ============================================================
# 6. VAR 모형 추정
# ============================================================

def fit_var(df, lag, use_diff=False):
    """
    use_diff=True  : 1차 차분 VAR (I(1) 비공적분)
    use_diff=False : 수준값 VAR  (정상 또는 공적분 있음)
    """
    print("\n" + "=" * 60)
    print(f"【VAR({lag}) 추정 — {'차분' if use_diff else '수준값'}】")
    print("=" * 60)

    data = df.diff().dropna() if use_diff else df
    model = VAR(data)
    result = model.fit(lag, trend='c')
    print(result.summary())
    return result, data


# ============================================================
# 7. 잔차 진단
# ============================================================

def residual_diagnostics(result):
    print("\n" + "=" * 60)
    print("【잔차 진단】")
    print("=" * 60)

    resid = result.resid

    # (1) Durbin-Watson
    print("\n[Durbin-Watson 통계량]")
    for i, col in enumerate(resid.columns):
        dw = durbin_watson(resid.iloc[:, i])
        print(f"  {col}: {dw:.4f}  (2에 가까울수록 자기상관 없음)")

    # (2) Ljung-Box
    print("\n[Ljung-Box Q 검정 (lag=10, H0=자기상관 없음)]")
    for col in resid.columns:
        lb = acorr_ljungbox(resid[col], lags=[10], return_df=True)
        p = lb['lb_pvalue'].values[0]
        print(f"  {col}: p={p:.4f}  → {'✓ 자기상관 없음' if p>0.05 else '✗ 자기상관 있음'}")

    # (3) 정규성 (Jarque-Bera)
    print("\n[잔차 정규성 검정 (Jarque-Bera)]")
    for col in resid.columns:
        jb_stat, jb_p = stats.jarque_bera(resid[col])
        print(f"  {col}: stat={jb_stat:.4f}  p={jb_p:.4f}  "
              f"→ {'✓ 정규분포' if jb_p>0.05 else '△ 비정규'}")

    # (4) 잔차 시각화
    fig, axes = plt.subplots(len(resid.columns), 2, figsize=(12, 3*len(resid.columns)))
    for i, col in enumerate(resid.columns):
        axes[i, 0].plot(resid.index, resid[col], color='steelblue', linewidth=1)
        axes[i, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        axes[i, 0].set_title(f'{col} 잔차', fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)
        stats.probplot(resid[col], plot=axes[i, 1])
        axes[i, 1].set_title(f'{col} Q-Q plot', fontsize=10)
    plt.tight_layout()
    plt.savefig('02_잔차진단.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n그림 저장: 02_잔차진단.png")


# ============================================================
# 8. 그랜저 인과관계 검정
# ============================================================

def granger_test(df, max_lag=4):
    """
    물동량 → PPI, PPI → CPI, 물동량 → CPI 파급 여부 검정.
    H0: x는 y를 그랜저 인과하지 않음  → p<0.05면 인과관계 있음
    """
    print("\n" + "=" * 60)
    print("【그랜저 인과관계 검정】")
    print("=" * 60)

    pairs = [
        ('ppi',   'cargo', '물동량 → PPI'),
        ('cpi',   'ppi',   'PPI → CPI'),
        ('cpi',   'cargo', '물동량 → CPI'),
        ('cargo', 'ppi',   'PPI → 물동량 (역방향 확인)'),
        ('cargo', 'cpi',   'CPI → 물동량 (역방향 확인)'),
    ]

    for caused, cause, label in pairs:
        data_gc = df[[caused, cause]].dropna()
        test = grangercausalitytests(data_gc, maxlag=max_lag, verbose=False)
        # F검정 p값 (각 시차)
        ps = [test[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
        min_p = min(ps)
        best_lag = ps.index(min_p) + 1
        sig = '★ 유의 (p<0.05)' if min_p < 0.05 else '  비유의'
        print(f"\n  {label}")
        print(f"    최소 p값={min_p:.4f} (시차 {best_lag})  {sig}")


# ============================================================
# 9. 충격반응함수 (IRF)
# ============================================================

def plot_irf(result, periods=12, orth=True):
    """
    orth=True: 직교화 충격반응 (Cholesky)
    변수 순서: cargo → ppi → cpi (이론적 인과 순서)
    """
    print("\n" + "=" * 60)
    print("【충격반응함수 (IRF)】")
    print("=" * 60)

    irf = result.irf(periods=periods)

    # 핵심 경로만 별도 시각화
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    var_names = result.names
    impulse_labels = {n: n for n in var_names}

    irf.plot(orth=orth, figsize=(14, 8))
    plt.suptitle('충격반응함수 (Orthogonalized IRF)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('03_IRF_전체.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("그림 저장: 03_IRF_전체.png")

    # 핵심 경로: 물동량 충격 → PPI/CPI 반응
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    irf_vals = irf.irfs  # shape: (periods+1, n, n)
    col_idx = {name: i for i, name in enumerate(var_names)}

    if 'cargo' in col_idx and 'ppi' in col_idx:
        ci = col_idx['cargo']
        pi = col_idx['ppi']
        axes[0].plot(irf_vals[:, pi, ci], color='steelblue', linewidth=2, marker='o', markersize=4)
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[0].fill_between(range(periods+1),
                              irf.cum_effect_stderr(orth=orth)[:, pi, ci] * -1.96,
                              irf.cum_effect_stderr(orth=orth)[:, pi, ci] *  1.96,
                              alpha=0.15, color='steelblue')
        axes[0].set_title('물동량 충격 → PPI 반응', fontsize=11)
        axes[0].set_xlabel('개월')
        axes[0].grid(True, alpha=0.3)

    if 'ppi' in col_idx and 'cpi' in col_idx:
        pi = col_idx['ppi']
        ci_idx = col_idx['cpi']
        axes[1].plot(irf_vals[:, ci_idx, pi], color='tomato', linewidth=2, marker='o', markersize=4)
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_title('PPI 충격 → CPI 반응', fontsize=11)
        axes[1].set_xlabel('개월')
        axes[1].grid(True, alpha=0.3)

    plt.suptitle('핵심 파급경로 IRF', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_IRF_핵심경로.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("그림 저장: 04_IRF_핵심경로.png")


# ============================================================
# 10. 분산분해 (FEVD)
# ============================================================

def plot_fevd(result, periods=12):
    print("\n" + "=" * 60)
    print("【예측오차 분산분해 (FEVD)】")
    print("=" * 60)

    fevd = result.fevd(periods=periods)
    fevd.summary()

    fevd.plot(figsize=(12, 8))
    plt.suptitle('예측오차 분산분해 (FEVD)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('05_FEVD.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("그림 저장: 05_FEVD.png")

    # CPI 분산분해 수치 출력
    print("\n[CPI 분산분해] — 각 변수가 CPI 변동을 얼마나 설명하는가 (%)")
    df_fevd = pd.DataFrame(fevd.decomp[:, -1, :],   # -1 = CPI (마지막 변수)
                           columns=result.names)
    df_fevd.index.name = '기간(개월)'
    print((df_fevd * 100).round(2).to_string())


# ============================================================
# 11. 예측 (옵션)
# ============================================================

def forecast_var(result, data, steps=6):
    print("\n" + "=" * 60)
    print(f"【VAR 예측 — 향후 {steps}개월】")
    print("=" * 60)

    fc, lower, upper = result.forecast_interval(data.values[-result.k_ar:], steps=steps, alpha=0.05)
    idx = pd.date_range(data.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq='MS')

    fc_df = pd.DataFrame(fc, index=idx, columns=result.names)
    lo_df = pd.DataFrame(lower, index=idx, columns=result.names)
    hi_df = pd.DataFrame(upper, index=idx, columns=result.names)

    print("\n예측값 (95% 신뢰구간):")
    for col in result.names:
        print(f"\n  [{col}]")
        for dt in idx:
            print(f"    {dt.strftime('%Y-%m')}: {fc_df.loc[dt, col]:,.2f} "
                  f"[{lo_df.loc[dt, col]:,.2f} ~ {hi_df.loc[dt, col]:,.2f}]")

    # 시각화
    fig, axes = plt.subplots(len(result.names), 1, figsize=(12, 3*len(result.names)), sharex=False)
    if len(result.names) == 1: axes = [axes]
    for ax, col in zip(axes, result.names):
        ax.plot(data.index[-24:], data[col].iloc[-24:], color='steelblue', label='실제값')
        ax.plot(idx, fc_df[col], color='tomato', linestyle='--', marker='o', label='예측값')
        ax.fill_between(idx, lo_df[col], hi_df[col], alpha=0.15, color='tomato')
        ax.set_title(f'{col} 예측', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'VAR 모형 예측 (향후 {steps}개월)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('06_예측.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n그림 저장: 06_예측.png")

    return fc_df


# ============================================================
# 12. 결과 저장
# ============================================================

def save_results(result, irf_data=None, fevd_data=None, output_path='VAR_결과.xlsx'):
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # 계수
        coef_df = pd.DataFrame(result.params, index=result.names)
        coef_df.to_excel(writer, sheet_name='계수')

        # p값
        pval_df = pd.DataFrame(result.pvalues, index=result.names)
        pval_df.to_excel(writer, sheet_name='p값')

        # 잔차
        result.resid.to_excel(writer, sheet_name='잔차')

        # 적합값
        pd.DataFrame(result.fittedvalues, columns=result.names).to_excel(
            writer, sheet_name='적합값')

    print(f"\n결과 저장 완료: {output_path}")


# ============================================================
# MAIN — 전체 분석 실행
# ============================================================

if __name__ == '__main__':

    # ── ① 데이터 로드 ───────────────────────────────────────
    # 방법 A: 실제 PPI/CPI 파일이 있을 때
    # cargo, ppi, cpi = load_data(
    #     cargo_path='여수광양항_물동량_wide_VAR.csv',
    #     ppi_path='전남동부_PPI.csv',
    #     cpi_path='전국_CPI.csv',
    # )
    # df = build_panel(cargo, ppi, cpi, start='2018-01', end='2021-07')

    # 방법 B: 수동으로 데이터프레임 구성 (PPI/CPI를 직접 붙여넣을 때)
    # 아래에서 ppi_values, cpi_values를 ECOS에서 받은 값으로 교체
    cargo = pd.read_csv('여수광양항_물동량_wide_VAR.csv')
    cargo['date'] = pd.to_datetime(cargo['year_month'])
    cargo = cargo.set_index('date').sort_index()

    # 예시: 2018-01 ~ 2021-07 기간 물동량
    cargo_series = cargo.loc['2018-01':'2021-07', '총물동량_합계_RT']

    # ── PPI/CPI는 아래 리스트를 ECOS 실제값으로 교체 ──────
    # ECOS 주소: https://ecos.bok.or.kr
    # 전남동부 PPI: 통계분류 → 2.2 생산자물가지수
    # 전국 CPI:    통계분류 → 2.5 소비자물가지수
    date_idx = pd.date_range('2018-01', periods=len(cargo_series), freq='MS')

    # ▼▼▼ 실제 데이터로 교체 필요 ▼▼▼
    ppi_values = np.ones(len(cargo_series)) * np.nan  # ← ECOS PPI 값
    cpi_values = np.ones(len(cargo_series)) * np.nan  # ← ECOS CPI 값
    # ▲▲▲ 실제 데이터로 교체 필요 ▲▲▲

    df = pd.DataFrame({
        'cargo': cargo_series.values,
        'ppi':   ppi_values,
        'cpi':   cpi_values,
    }, index=date_idx).dropna()

    if df['ppi'].isna().all():
        print("=" * 60)
        print("⚠️  PPI/CPI 데이터를 입력해야 분석이 실행됩니다.")
        print("   ECOS(https://ecos.bok.or.kr)에서 다운로드 후")
        print("   ppi_values, cpi_values 리스트를 교체하세요.")
        print("=" * 60)
        print("\n물동량 데이터 미리보기:")
        print(cargo_series.head(10))
    else:
        # ── ② 시각화 ─────────────────────────────────────────
        print("\n[1단계] 시계열 시각화")
        plot_series(df)

        # ── ③ 단위근 검정 ─────────────────────────────────────
        print("\n[2단계] 단위근 검정")
        ur_results, n_I1 = unit_root_test(df)

        # ── ④ 공적분 검정 ─────────────────────────────────────
        use_diff = False
        if n_I1 >= 2:
            print("\n[3단계] 공적분 검정")
            n_coint = johansen_test(df)
            use_diff = (n_coint == 0)   # 공적분 없으면 차분

        # ── ⑤ 최적 시차 선택 ─────────────────────────────────
        print("\n[4단계] 최적 시차 선택")
        data_for_lag = df.diff().dropna() if use_diff else df
        best_lag = select_lag(data_for_lag)

        # ── ⑥ VAR 추정 ───────────────────────────────────────
        print("\n[5단계] VAR 모형 추정")
        var_result, var_data = fit_var(df, best_lag, use_diff=use_diff)

        # ── ⑦ 잔차 진단 ──────────────────────────────────────
        print("\n[6단계] 잔차 진단")
        residual_diagnostics(var_result)

        # ── ⑧ 그랜저 인과 ────────────────────────────────────
        print("\n[7단계] 그랜저 인과관계 검정")
        granger_test(var_data)

        # ── ⑨ IRF ─────────────────────────────────────────────
        print("\n[8단계] 충격반응함수")
        plot_irf(var_result, periods=12)

        # ── ⑩ FEVD ────────────────────────────────────────────
        print("\n[9단계] 분산분해")
        plot_fevd(var_result, periods=12)

        # ── ⑪ 예측 ────────────────────────────────────────────
        print("\n[10단계] 예측")
        fc = forecast_var(var_result, var_data, steps=6)

        # ── ⑫ 결과 저장 ───────────────────────────────────────
        save_results(var_result, output_path='VAR_결과.xlsx')

        print("\n✅ 분석 완료!")
