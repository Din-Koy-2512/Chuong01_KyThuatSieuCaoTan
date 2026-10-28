"""WaveLab – Mô phỏng VSWR | Ứng dụng Streamlit một file duy nhất.
Chạy: python -m streamlit run VSWR_Streamlit_V2.py

Phần giải thuật dưới đây được giữ nguyên từ vswr_core.py (phiên bản V1).
Toàn bộ thay đổi trong bản này chỉ thuộc giao diện, đồ thị và trải nghiệm tương tác.
Các tham chiếu học thuật cho công thức được tóm tắt trong ghi chú ngay đầu khối lõi.
"""
from __future__ import annotations

import math
import base64
from io import BytesIO
from urllib.parse import urlsplit
from html import escape as escape_xml
from functools import lru_cache
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.signal import find_peaks
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ===========================================================================
# CHỖ NHẬP LINK WEB CHÍNH THỨC ĐỂ TẠO MÃ QR
# Khi đưa WaveLab lên Streamlit Community Cloud, dán URL công khai vào đây.
# Ví dụ: WEB_APP_URL = "https://wavelab-vswr.streamlit.app/"
# Để trống khi chưa triển khai. QR chưa xuất hiện cho đến khi có URL hợp lệ.
# Khi điền tại đây, mọi người truy cập ứng dụng đều thấy cùng một mã QR.
# ===========================================================================
WEB_APP_URL = "https://n8cd3kythuatsieucaotan.streamlit.app/"  # <-- DÁN LINK WEB SAU KHI ĐƯA LÊN ĐÁM MÂY, GIỮ DẤU NGOẶC KÉP

# ===========================================================================
# KHỐI LÕI KHOA HỌC – GIỮ NGUYÊN TỪ V1, KHÔNG CHỈNH SỬA
# Giả thiết: đường dây TEM, Z0 thực, alpha>=0; d=0 tại tải, d=L tại nguồn;
# điện áp sóng tới tại nguồn được chuẩn hoá bằng 1.
# Nguồn gốc: D. M. Pozar, Microwave Engineering, 4th ed., Ch. 2 §§2.1,
# 2.3, 2.7 (2.89–2.91); S. J. Orfanidis, Electromagnetic Waves and Antennas,
# Ch. 11 §§11.7 và 11.12; M. Steer, Microwave and RF Design II §§2.3, 2.5.
# Các biểu thức vị trí/đạo hàm được biến đổi từ sóng tới và sóng phản xạ.
# ===========================================================================
C0 = 299_792_458.0
Mode = Literal["general", "matched", "short", "open", "reactive"]


@dataclass(frozen=True)
class LineConfig:
    f_hz: float = 1.0e9
    vp_mps: float = 2.0e8
    z0_ohm: float = 50.0
    length_wavelengths: float = 2.0
    alpha_np_m: float = 0.0
    mode: Mode = "general"
    r_load_ohm: float = 50.0
    x_load_ohm: float = 50.0
    samples: int = 1601

    def __post_init__(self) -> None:
        fields = (self.f_hz, self.vp_mps, self.z0_ohm,
                  self.length_wavelengths, self.alpha_np_m,
                  self.r_load_ohm, self.x_load_ohm)
        if not all(np.isfinite(v) for v in fields):
            raise ValueError("Các tham số đầu vào phải là số hữu hạn.")
        if self.f_hz <= 0 or not 0 < self.vp_mps <= C0:
            raise ValueError("Cần f > 0 và 0 < vₚ ≤ c₀.")
        if self.z0_ohm <= 0 or self.length_wavelengths <= 0 or self.alpha_np_m < 0:
            raise ValueError("Cần Z₀ > 0, L > 0 và α ≥ 0.")
        if self.r_load_ohm < 0:
            raise ValueError("Chỉ hỗ trợ tải thụ động: R tải ≥ 0.")
        if self.mode not in ("general", "matched", "short", "open", "reactive"):
            raise ValueError("Chế độ tải không hợp lệ.")
        if self.mode == "reactive" and self.x_load_ohm == 0:
            raise ValueError("Tải thuần phản kháng cần X tải khác 0.")
        if not 101 <= self.samples <= 12001:
            raise ValueError("Số điểm lấy mẫu phải trong khoảng 101–12001.")
        if self.alpha_np_m * self.length_m > 20:
            raise ValueError("αL > 20; ngoài miền mô phỏng ổn định số của ứng dụng.")

    @property
    def wavelength_m(self) -> float:
        return self.vp_mps / self.f_hz

    @property
    def length_m(self) -> float:
        return self.length_wavelengths * self.wavelength_m

    @property
    def beta_rad_m(self) -> float:
        return 2 * math.pi / self.wavelength_m

    @property
    def gamma(self) -> complex:
        return complex(self.alpha_np_m, self.beta_rad_m)

    @property
    def load_impedance(self) -> complex | None:
        """None explicitly represents the OPEN circuit, never a huge fake value."""
        if self.mode == "open":
            return None
        if self.mode == "short":
            return 0j
        if self.mode == "matched":
            return complex(self.z0_ohm, 0)
        if self.mode == "reactive":
            return complex(0, self.x_load_ohm)
        return complex(self.r_load_ohm, self.x_load_ohm)

    @property
    def gamma_load(self) -> complex:
        z_l = self.load_impedance
        if z_l is None:
            return 1 + 0j
        return complex((z_l - self.z0_ohm) / (z_l + self.z0_ohm))


@dataclass
class LineSolution:
    cfg: LineConfig
    d_m: NDArray[np.float64]
    gamma_d: NDArray[np.complex128]
    v_plus: NDArray[np.complex128]
    v_minus: NDArray[np.complex128]
    v_total: NDArray[np.complex128]
    i_total_norm: NDArray[np.complex128]
    zin: NDArray[np.complex128]
    vswr_local: NDArray[np.float64]
    envelope_high: NDArray[np.float64]
    envelope_low: NDArray[np.float64]

    @property
    def d_lambda(self) -> NDArray[np.float64]:
        return self.d_m / self.cfg.wavelength_m

    @property
    def grid_step_m(self) -> float:
        return self.cfg.length_m / (self.cfg.samples - 1)


def safe_vswr(rho: NDArray | float) -> NDArray | float:
    rho_ar = np.asarray(rho, dtype=float)
    output = np.divide(1 + rho_ar, 1 - rho_ar,
                       out=np.full_like(rho_ar, np.inf), where=(1-rho_ar) > 1e-13)
    return float(output) if output.ndim == 0 else output


def return_loss_db(rho: float) -> float:
    return math.inf if rho <= 1e-14 else -20 * math.log10(rho)


def mismatch_loss_db(rho: float) -> float:
    return math.inf if rho >= 1-1e-13 else -10 * math.log10(1-rho*rho)


def solve(cfg: LineConfig) -> LineSolution:
    d = np.linspace(0.0, cfg.length_m, cfg.samples)
    # Phasor incident at source is exactly 1 (+0j); phasor at the load decays.
    v_plus = np.exp(-cfg.gamma * (cfg.length_m - d))
    v_minus = cfg.gamma_load * np.exp(-cfg.gamma * (cfg.length_m + d))
    v_tot = v_plus + v_minus
    j_tot = v_plus - v_minus  # J = Z0*I / V_plus(source)
    gamma_d = cfg.gamma_load * np.exp(-2 * cfg.gamma * d)
    den = 1 - gamma_d
    zin = np.full(d.shape, complex(np.inf, 0), dtype=complex)
    mask = np.abs(den) > 1e-12
    zin[mask] = cfg.z0_ohm * (1 + gamma_d[mask]) / den[mask]
    rho = np.abs(gamma_d)
    high = np.abs(v_plus) + np.abs(v_minus)
    low = np.maximum(0, np.abs(v_plus)-np.abs(v_minus))
    return LineSolution(cfg=cfg, d_m=d, gamma_d=gamma_d, v_plus=v_plus,
                        v_minus=v_minus, v_total=v_tot, i_total_norm=j_tot,
                        zin=zin, vswr_local=np.asarray(safe_vswr(rho)),
                        envelope_high=high, envelope_low=low)


def phase_points(cfg: LineConfig) -> dict[str, NDArray[np.float64]]:
    """In-phase and anti-phase positions (not necessarily |V| extrema if alpha>0)."""
    rho = abs(cfg.gamma_load)
    if rho < 1e-13:
        return {"max": np.array([], dtype=float), "min": np.array([], dtype=float)}
    theta = np.angle(cfg.gamma_load) % (2*np.pi)
    starts = {"max": theta * cfg.wavelength_m / (4*np.pi),
              "min": ((theta+np.pi) % (2*np.pi)) * cfg.wavelength_m / (4*np.pi)}
    result = {}
    for label, base in starts.items():
        n_max = max(0, math.ceil(2*cfg.length_wavelengths)+2)
        values = base + np.arange(n_max+1) * cfg.wavelength_m/2
        result[label] = values[values <= cfg.length_m + 1e-10 * cfg.wavelength_m]
    return result


def stationary_derivative(cfg: LineConfig, d_m: NDArray | float) -> NDArray | float:
    """Derivative of |V|² / exp(-2*alpha*L) with respect to d (source-normalized)."""
    d = np.asarray(d_m)
    a, b = cfg.alpha_np_m, cfg.beta_rad_m
    r = abs(cfg.gamma_load)
    theta = np.angle(cfg.gamma_load)
    return (2*a*np.exp(2*a*d) - 2*a*r*r*np.exp(-2*a*d)
            - 4*b*r*np.sin(2*b*d-theta))


def stationary_points(cfg: LineConfig) -> dict[str, NDArray[np.float64]]:
    """True INTERIOR local extrema of |V(d)| via exact-derivative root finding.

    Detect sign changes; classifies extrema on squared amplitude, so short/open
    circuit voltage zeros (cusps in |V| itself) remain well-defined minima.
    """
    empty = {"max": np.array([], dtype=float), "min": np.array([], dtype=float)}
    if abs(cfg.gamma_load) < 1e-13:
        return empty
    n = max(3000, math.ceil(700 * cfg.length_wavelengths))
    xs = np.linspace(0, cfg.length_m, n)
    ys = np.asarray(stationary_derivative(cfg, xs))
    output: dict[str, list[float]] = {"max": [], "min": []}
    for i in range(n-1):
        # Exclude endpoints; these are not interior peaks.
        if ys[i] == 0:
            root = float(xs[i])
        elif ys[i] * ys[i+1] < 0:
            root = brentq(lambda x: float(stationary_derivative(cfg, x)),
                          xs[i], xs[i+1], xtol=max(1e-14, cfg.length_m*1e-12))
        else:
            continue
        if root <= 1e-10*cfg.wavelength_m or root >= cfg.length_m-1e-10*cfg.wavelength_m:
            continue
        # A derivative changing from + to - is a maximum; - to + is a minimum.
        h = min(cfg.wavelength_m * 1e-5, root/2, (cfg.length_m-root)/2)
        left = float(stationary_derivative(cfg, root-h))
        right = float(stationary_derivative(cfg, root+h))
        typ = "max" if left > 0 and right < 0 else "min" if left < 0 and right > 0 else None
        if typ and (not output[typ] or abs(output[typ][-1] - root) > cfg.wavelength_m*1e-7):
            output[typ].append(root)
    return {k: np.array(v, dtype=float) for k, v in output.items()}


def critical_distance(cfg: LineConfig) -> float | None:
    """Necessary threshold boundary, NOT a sufficient existence condition.
    Returns d_crit; negative means no d>=0 can pass the necessary criterion.
    """
    a, b, r = cfg.alpha_np_m, cfg.beta_rad_m, abs(cfg.gamma_load)
    if a == 0 or r < 1e-13:
        return None
    return (np.log(r) + np.arcsinh(b/a)) / (2*a)


def numerical_grid_peaks(sol: LineSolution) -> dict[str, NDArray[np.float64]]:
    y = np.abs(sol.v_total)
    # No arbitrary peak prominence: retain weak interference extrema.
    hi, _ = find_peaks(y)
    lo, _ = find_peaks(-y)
    return {"max": sol.d_m[hi], "min": sol.d_m[lo]}


def cascade_abcd(sol: LineSolution) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Independent transmission-matrix stepping, in normalized (V,J=Z0 I) basis.

    For each small segment Δd: [V(d+Δ),J(d+Δ)] =
        [[cosh(γΔ), sinh(γΔ)],[sinh(γΔ),cosh(γΔ)]] · [V(d),J(d)].
    Starting condition is set only at d=0, not by analytical V(d) samples.
    """
    v = np.empty(sol.cfg.samples, dtype=np.complex128)
    j = np.empty(sol.cfg.samples, dtype=np.complex128)
    v[0] = sol.v_plus[0] + sol.v_minus[0]
    j[0] = sol.v_plus[0] - sol.v_minus[0]
    gd = sol.cfg.gamma * sol.grid_step_m
    c, s = np.cosh(gd), np.sinh(gd)
    for k in range(1, sol.cfg.samples):
        v[k] = c*v[k-1] + s*j[k-1]
        j[k] = s*v[k-1] + c*j[k-1]
    return v, j


def verification(sol: LineSolution) -> dict:
    """Sampled-grid vs analytical, and independent ABCD cascade vs analytical."""
    v_abcd, j_abcd = cascade_abcd(sol)
    norm_v = max(float(np.max(np.abs(sol.v_total))), 1e-14)
    norm_j = max(float(np.max(np.abs(sol.i_total_norm))), 1e-14)
    errors = {
        "voltage_abcd_rel": float(np.max(np.abs(v_abcd-sol.v_total))/norm_v),
        "current_abcd_rel": float(np.max(np.abs(j_abcd-sol.i_total_norm))/norm_j),
        "grid_step_m": sol.grid_step_m,
    }
    # Gamma reconstructed from independent voltage and current samples.
    denom = v_abcd+j_abcd
    if np.min(np.abs(denom)) > 1e-12:
        g_abcd = (v_abcd-j_abcd)/denom
        errors["gamma_abcd_abs"] = float(np.max(np.abs(g_abcd-sol.gamma_d)))
    else:
        errors["gamma_abcd_abs"] = float("nan")
    # Independent grid peak locations versus actual stationary roots.
    analytic = stationary_points(sol.cfg)
    numeric = numerical_grid_peaks(sol)
    diffs: dict[str, list[float]] = {"max": [], "min": []}
    for k in ("max", "min"):
        for true_pt in analytic[k]:
            if len(numeric[k]):
                d = float(np.min(np.abs(numeric[k] - true_pt)))
                # Do not match a different nearby extremum if grid missed this one.
                if d <= 2.1*sol.grid_step_m:
                    diffs[k].append(d)
    errors["extrema_errors_m"] = diffs
    errors["extrema_counts"] = {"analytic": {k: len(v) for k, v in analytic.items()},
                                "grid": {k: len(v) for k, v in numeric.items()},
                                "matched": {k: len(v) for k, v in diffs.items()}}
    return errors


def compact_complex(x: complex | None, digits: int = 4) -> str:
    if x is None:
        return "∞ (hở mạch)"
    if not (np.isfinite(x.real) and np.isfinite(x.imag)):
        return "∞"
    return f"{x.real:.{digits}f} {'+' if x.imag >= 0 else '−'} j{abs(x.imag):.{digits}f}"


# ===========================================================================
# GIAO DIỆN – V2 · NEON WAVE / CHUẨN HÓA NỘI DUNG VÀ TYPOGRAPHY
# ===========================================================================
st.set_page_config(page_title="WaveLab | Mô phỏng VSWR", page_icon="〰️",
                   layout="wide", initial_sidebar_state="expanded")

COL = {
    "cyan": "#57EFDC", "blue": "#F4FF62", "amber": "#FFC76B",
    "coral": "#FF82A9", "violet": "#C4ABFF", "green": "#63E9C2",
    "ink": "#F2F7F9", "muted": "#BBC8D4",
}

# Đồ thị vẫn sử dụng đúng các mảng nghiệm từ solve(); chỉ thay phần hiển thị.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "filename": "WaveLab_VSWR", "scale": 2},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root {color-scheme: dark;--bg:#080e19;--panel:#121e2c;--panel-hi:#192839;
  --edge:#314357;--text:#f3f7fd;--muted:#c2d0df;--accent:#83e2d6;--lime:#e7f96c;
  --bodyfont:'Be Vietnam Pro','Segoe UI',Arial,sans-serif;}
html, body, .stApp, .stApp button, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"] *, .stApp [data-testid="stWidgetLabel"] * {
  font-family:var(--bodyfont)!important;font-kerning:normal;}
html,body{font-size:16px}
.stApp,[data-testid="stAppViewContainer"]{color:var(--text)!important;
  background:radial-gradient(ellipse 50% 25% at 100% 2%,rgba(59,90,137,.24),transparent 80%),
  radial-gradient(ellipse 30% 24% at 0% 57%,rgba(30,118,112,.09),transparent 80%),
  linear-gradient(180deg,#080d18 0%,#0b121d 54%,#080e17 100%)!important;}
[data-testid="stHeader"]{background:rgba(8,13,24,.76)!important;backdrop-filter:blur(14px);
 border-bottom:1px solid rgba(132,161,188,.12)}
[data-testid="stDecoration"]{background:linear-gradient(90deg,var(--accent),var(--lime))}
[data-testid="stAppViewContainer"] .block-container{max-width:1380px!important;
  padding:1.2rem clamp(1.1rem,3.0vw,2.7rem) 3.0rem!important;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0c1523,#101b28)!important;
  border-right:1px solid #2c4052!important;}
[data-testid="stSidebar"]>div:first-child{padding-top:1rem}
.stApp [data-testid="stMarkdownContainer"] p,.stApp [data-testid="stMarkdownContainer"] li{
  color:#e0eaf4!important;font-size:.88rem!important;line-height:1.72!important;font-weight:400!important;
  letter-spacing:0!important;}
.stApp [data-testid="stMarkdownContainer"] strong{color:var(--text)!important;font-weight:700!important;}
.stApp [data-testid="stMarkdownContainer"] h4{
  color:#f3f7fd!important;font-size:1.05rem!important;line-height:1.45!important;
  letter-spacing:-.01em!important;margin:1.15rem 0 .75rem!important;font-weight:700!important;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{
 color:#bbcddd!important;font-size:.79rem!important;line-height:1.64!important;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{
  color:#e8f3fa!important;font-size:.83rem!important;font-weight:600!important;line-height:1.45!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#bbccdc!important;}
[data-testid="stSidebar"] hr{border-color:var(--edge)!important;}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-baseweb="select"]>div{
  background:#1c2b3b!important;color:#f8fafc!important;border:1px solid #4f687c!important;
  border-radius:10px!important;font-weight:600!important;}
[data-testid="stNumberInput"] input:focus,[data-testid="stTextInput"] input:focus{border-color:var(--accent)!important;}
[data-baseweb="popover"] [role="listbox"]{background:#1b2b3b!important;}
[data-baseweb="popover"] [role="option"]{color:#eef6fd!important;}
[data-baseweb="popover"] [role="option"]:hover{background:#355363!important;}
[data-testid="stSlider"] [role="slider"]{background:var(--accent)!important;border-color:var(--accent)!important;}
/* Heading and brand */
.side-brand{background:linear-gradient(125deg,rgba(33,61,74,.8),rgba(30,40,69,.85));
  border:1px solid #4c6772;border-radius:16px;padding:18px 18px;margin:0 0 18px;}
.side-brand strong{font-size:1.35rem!important;font-weight:800!important;color:var(--lime)!important;
  letter-spacing:-.03em!important;line-height:1.25!important;display:block;}
.side-brand small{font-size:.71rem!important;color:#c7dfeb!important;display:block;
  letter-spacing:.01em!important;margin-top:5px;line-height:1.6!important;}
.side-head{font-size:.86rem!important;color:var(--accent)!important;font-weight:700!important;
  margin:22px 0 12px!important;line-height:1.5!important;letter-spacing:0!important;}
/* Cinematic but compact intro: four distinct sections */
.hero{position:relative;overflow:hidden;isolation:isolate;border-radius:21px;
 background:radial-gradient(ellipse at 81% 30%,rgba(39,117,137,.33),transparent 52%),
 radial-gradient(ellipse at 10% 0%,rgba(86,73,156,.17),transparent 68%),
 linear-gradient(120deg,#182743,#111c31 58%,#0d1b2b);border:1px solid #456274;
 padding:32px clamp(22px,3.3vw,43px) 25px;margin:0 0 17px;
 box-shadow:0 20px 48px rgba(0,0,0,.16);}
.hero:before{content:'';position:absolute;right:-7%;top:-50%;height:220%;width:57%;z-index:-1;
 background:repeating-radial-gradient(ellipse at 100% 50%,transparent 0 22px,rgba(106,239,224,.14) 23px,transparent 25px 43px);
 transform:rotate(-14deg);pointer-events:none;}
.hero:after{content:'';position:absolute;right:10%;bottom:10%;width:25%;height:25%;
 border-radius:50%;background:rgba(178,231,118,.12);filter:blur(56px);z-index:-1;pointer-events:none;}
.hero-kicker{color:var(--accent)!important;letter-spacing:.12em!important;text-transform:uppercase!important;
 font-size:.69rem!important;line-height:1.65!important;font-weight:800!important;}
.hero-title{color:#f4f8fd!important;font-size:clamp(1.75rem,3.0vw,2.72rem)!important;
 font-weight:800!important;line-height:1.25!important;letter-spacing:-.034em!important;
 font-style:normal!important;margin:9px 0 9px!important;max-width:880px;}
.hero-title span{color:var(--lime)!important;}
.hero-sub{color:#dcebf3!important;max-width:810px;line-height:1.76!important;
 font-size:.91rem!important;}
.hero-features{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;
 border-top:1px solid rgba(191,221,228,.17);margin-top:26px;padding-top:16px;}
.hero-feature{padding:10px 12px 11px;border:1px solid rgba(148,195,205,.25);
 border-radius:11px;background:rgba(12,25,37,.39);transition:background .2s,border-color .2s,transform .2s;}
.hero-feature:hover{transform:translateY(-2px);background:rgba(25,49,60,.65);border-color:rgba(140,238,219,.6)}
.hero-feature-index{color:var(--lime)!important;font-weight:800!important;font-size:.73rem!important;display:block;margin-bottom:4px;}
.hero-feature-label{color:#e9f3f8!important;font-weight:600!important;font-size:.76rem!important;line-height:1.45;display:block;}
/* Distinct navigation buttons — actual Streamlit controls */
.nav-kicker{display:flex;align-items:center;gap:10px;margin:16px 0 12px;
 color:var(--accent)!important;font-size:.82rem!important;font-weight:700!important;line-height:1.6!important;}
.nav-kicker:before{content:'';width:7px;height:7px;background:var(--lime);border-radius:50%;flex:none;box-shadow:0 0 12px rgba(231,249,108,.26);}
.stButton button[kind="secondary"],.stButton button[kind="primary"]{
 width:100%!important;min-height:62px!important;padding:9px 10px!important;white-space:normal!important;
 border:1px solid #50687b!important;border-radius:13px!important;
 background:linear-gradient(180deg,#1b2a3a,#152131)!important;color:#e9f2f8!important;
 font-size:clamp(.75rem,.87vw,.87rem)!important;font-weight:700!important;line-height:1.45!important;
 letter-spacing:0!important;text-align:center!important;
 box-shadow:0 4px 11px rgba(0,0,0,.10);
 transition:transform .22s ease,background .22s ease,border-color .22s ease,box-shadow .22s ease!important;}
.stButton button[kind="secondary"] p,.stButton button[kind="secondary"] *{color:#eef5f9!important;}
.stButton button[kind="secondary"]:hover{background:linear-gradient(180deg,#223c4b,#182b3c)!important;
 border-color:var(--accent)!important;transform:translateY(-2px)!important;box-shadow:0 9px 22px rgba(0,0,0,.18);}
.stButton button[kind="primary"]{background:linear-gradient(120deg,#efff85,#dff366)!important;
 border-color:var(--lime)!important;color:#142328!important;
 box-shadow:0 7px 20px rgba(220,247,106,.16)!important;}
.stButton button[kind="primary"] p,.stButton button[kind="primary"] *{color:#142328!important;}
.stButton button[kind="primary"]:hover{transform:translateY(-2px)!important;box-shadow:0 12px 28px rgba(220,247,106,.20)!important;}
.stButton button:active{transform:translateY(1px) scale(.99)!important;}
.stButton button:focus-visible,[data-testid="stDownloadButton"] button:focus-visible,
[data-testid="stRadio"] label:focus-within{outline:2px solid var(--accent)!important;outline-offset:3px!important;}
/* Input load modes as outlined selectable rows */
[data-testid="stRadio"] [role="radiogroup"]{gap:4px;}
[data-testid="stRadio"] label{padding:10px 12px!important;min-height:43px!important;margin-bottom:6px!important;
 border-radius:10px!important;border:1px solid #4b6073!important;
 background:rgba(27,42,57,.82)!important;
 transition:background .18s,border-color .18s,box-shadow .18s!important;}
[data-testid="stRadio"] label:hover{border-color:var(--accent)!important;background:#22374a!important;}
[data-testid="stRadio"] label:has(input:checked){border-color:var(--lime)!important;background:#2b3840!important;
 box-shadow:inset 3px 0 0 var(--lime)!important;}
[data-testid="stRadio"] label p{color:#edf5fa!important;font-weight:600!important;font-size:.80rem!important;line-height:1.45!important;}
/* KPI readouts */
.info-rail{display:flex;flex-wrap:wrap;gap:9px;margin:13px 0 13px;}
.info-chip{border:1px solid #3a5064;border-radius:8px;background:rgba(23,39,55,.8);
 color:#dcebf5!important;padding:9px 12px;font-size:.78rem!important;line-height:1.48;
 transition:border-color .18s ease,background .18s ease;}
.info-chip:hover{border-color:var(--accent)!important;background:#1c3440;}
.info-chip b{color:var(--accent)!important;}
[data-testid="stMetric"]{background:linear-gradient(140deg,#182837,#131f2b 90%)!important;
 border:1px solid #3b5164!important;border-top:2px solid var(--accent)!important;
 border-radius:14px!important;padding:17px 16px!important;min-height:123px!important;
 transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease!important;}
[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:var(--accent)!important;
 box-shadow:0 13px 27px rgba(0,0,0,.17)!important;}
[data-testid="stMetricLabel"] p{color:#c9d8e3!important;font-size:.74rem!important;
 font-weight:600!important;line-height:1.50!important;}
[data-testid="stMetricValue"]{color:var(--lime)!important;font-weight:800!important;
 font-size:clamp(1.15rem,1.65vw,1.6rem)!important;font-variant-numeric:tabular-nums;
 letter-spacing:-.025em!important;}
/* Clear content hierarchy */
.section-head{display:flex;align-items:center;gap:11px;color:#f3f8fc!important;
 font-size:clamp(1.22rem,1.76vw,1.72rem)!important;font-weight:800!important;
 letter-spacing:-.024em!important;line-height:1.45!important;margin:23px 0 7px;}
.section-head:before{content:'';width:5px;height:24px;background:var(--accent);border-radius:7px;flex:none;}
.section-desc{font-size:.86rem!important;line-height:1.68!important;color:var(--muted)!important;
 max-width:950px;margin:0 0 16px!important;}
.note{background:#172a35!important;border:1px solid #396273!important;
 border-left:3px solid var(--accent)!important;border-radius:10px!important;
 padding:12px 15px!important;margin:12px 0!important;
 color:#e4f0f7!important;font-size:.83rem!important;line-height:1.7!important;}
.note b{color:#f3f8fc!important;}
/* Plotly owns its height. Override width only to prevent nested scrolling. */
[data-testid="stPlotlyChart"]{
 width:100%!important;max-width:100%!important;min-width:0!important;
 border:1px solid #3c5264!important;border-radius:15px!important;
 margin:6px 0 18px!important;padding:3px!important;
 overflow:hidden!important;box-sizing:border-box!important;
 box-shadow:0 10px 24px rgba(0,0,0,.08)!important;
 transition:border-color .20s ease,box-shadow .20s ease!important;}
[data-testid="stPlotlyChart"]:hover{border-color:#678b95!important;box-shadow:0 13px 28px rgba(0,0,0,.15)!important;}
[data-testid="stPlotlyChart"]>div{width:100%!important;max-width:100%!important;min-width:0!important;}
[data-testid="stHorizontalBlock"] > div{min-width:0!important;}
[data-testid="stDataFrame"]{border:1px solid #3d5265!important;border-radius:11px!important;
 overflow:hidden!important;margin:8px 0 18px!important;}
[data-testid="stExpander"]{border:1px solid #3e5264!important;background:#142436!important;border-radius:10px!important;}
[data-testid="stAlert"]{border-radius:11px!important;}
[data-testid="stDownloadButton"] button{background:linear-gradient(120deg,#efff85,#dff366)!important;
 color:#142328!important;border:1px solid var(--lime)!important;border-radius:11px!important;
 min-height:48px!important;padding:10px 17px!important;font-weight:800!important;}
[data-testid="stDownloadButton"] button *{color:#142328!important;}
.stApp [data-testid="stDownloadButton"] button [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stDownloadButton"] button p{color:#142328!important;font-weight:800!important;}
[data-testid="stDownloadButton"] button:hover{box-shadow:0 9px 23px rgba(220,247,106,.17)!important;transform:translateY(-1px);}
/* Four-scenario comparison: separate responsive figure cards, not a giant fixed chart. */
.scenario-title{font-size:.88rem!important;font-weight:700!important;color:#f2f7fc!important;
 line-height:1.5!important;letter-spacing:-.01em!important;margin:6px 0 7px!important;}
.scenario-note{color:#c3d3e1!important;font-size:.78rem!important;line-height:1.6!important;margin:0 0 8px!important;}
.grid-step{background:linear-gradient(100deg,#152637,#13212d);border:1px solid #3c5b6d;
 border-radius:12px;padding:15px 18px;margin:14px 0 18px;color:#eff6fc!important;
 font-variant-numeric:tabular-nums;line-height:1.75;font-size:.90rem;}
.grid-step strong{color:var(--accent)!important;}
.grid-step .grid-number{color:var(--lime)!important;font-weight:700!important;}
/* Gentle motion without endless cursor following/glows */
@media(prefers-reduced-motion:no-preference){
 .hero{animation:wl-appear .52s cubic-bezier(.2,.8,.2,1) both;}
 [data-testid="stMetric"]{animation:wl-appear .4s ease both;}
 @keyframes wl-appear{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
 @keyframes wl-drift{0%,100%{transform:rotate(-14deg) translateY(0)}50%{transform:rotate(-14deg) translateY(-10px)}}
 .hero:before{animation:wl-drift 13s ease-in-out infinite;}
}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;}}
@media(max-width:1050px){.hero-features{grid-template-columns:repeat(2,minmax(0,1fr));}
 [data-testid="stAppViewContainer"] .block-container{padding:1.15rem 1.2rem 3rem!important;}}
@media(max-width:620px){.hero{padding:24px 18px;}.hero-title{font-size:1.65rem!important;}
 .hero-features{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;}
 .stButton button[kind="secondary"],.stButton button[kind="primary"]{font-size:.75rem!important;min-height:65px!important;}
 .section-head{font-size:1.20rem!important;}}
</style>
""", unsafe_allow_html=True)


def style(fig: go.Figure, title: str | None = None, height: int = 390) -> go.Figure:
    """Only the visual presentation changes; analytical traces are untouched."""
    light = st.session_state.get("wl_chart_theme", "Nền tối") == "Nền sáng"
    paper = "#FFFFFF" if light else "#122031"
    plot = "#FFFFFF" if light else "#122031"
    foreground = "#1B2C3F" if light else "#F1F7FC"
    muted = "#45596D" if light else "#CBD8E6"
    grid = "#E6EDF1" if light else "#304255"
    frame = "#A8BAC9" if light else "#546B7D"
    fig.update_layout(
        template="plotly_white" if light else "plotly_dark", height=height + 55,
        autosize=True, width=None,
        paper_bgcolor=paper, plot_bgcolor=plot,
        font=dict(family="Be Vietnam Pro, Segoe UI, Arial, sans-serif", color=foreground, size=12),
        title=dict(text=title or "", x=.045, y=.96,
                   font=dict(size=16, color=foreground, family="Be Vietnam Pro, Segoe UI, Arial, sans-serif")),
        margin=dict(l=70, r=30, t=76 if title else 42, b=157),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#F7FAFC" if light else "#23364A",
                        bordercolor="#0D8D89" if light else "#61E8D6",
                        font=dict(color=foreground, size=12)),
        modebar=dict(bgcolor="rgba(230,237,243,.95)" if light else "rgba(24,37,52,.94)",
                     color=muted, activecolor="#0D8D89" if light else "#E7F96C"),
        legend=dict(orientation="h", y=-.44, yanchor="top", x=.015,
                    bgcolor="rgba(255,255,255,.90)" if light else "rgba(18,32,49,.86)",
                    bordercolor=grid, borderwidth=1, font=dict(size=11,color=foreground)),
        xaxis=dict(gridcolor=grid,zerolinecolor=grid,showline=True,linecolor=frame,
                   ticks="outside",tickcolor=frame,automargin=True,
                   title_font=dict(color=foreground,size=12),tickfont=dict(color=muted,size=11)),
        yaxis=dict(gridcolor=grid,zerolinecolor=grid,showline=True,linecolor=frame,
                   ticks="outside",tickcolor=frame,automargin=True,
                   title_font=dict(color=foreground,size=12),tickfont=dict(color=muted,size=11)),
    )
    return fig

@lru_cache(maxsize=12)
def fraction_svg(numerator: str, denominator: str, color: str) -> str:
    """Render a true stacked fraction as SVG; independent of Plotly MathJax.

    Embedded in the figure, so titles remain readable in Plotly and exported PNG.
    The fraction line is a real SVG stroke, not a slash or raw TeX string.
    """
    numerator = escape_xml(numerator)
    denominator = escape_xml(denominator)
    color = escape_xml(color)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="84" viewBox="0 0 80 84">'
        f'<g fill="{color}" font-family="Cambria Math,DejaVu Sans,Times New Roman,serif"'
        ' font-size="31" text-anchor="middle" font-style="italic">'
        f'<text x="40" y="30">{numerator}</text>'
        f'<text x="40" y="74">{denominator}</text></g>'
        f'<path d="M14 40 H66" stroke="{color}" stroke-width="2.4" />'
        '</svg>'
    )
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode('utf-8')).decode('ascii')


def add_fraction_axis(fig: go.Figure, *, axis: str = 'x',
                      numerator: str = 'd', denominator: str = 'λ',
                      x: float = .5, y: float = -.16) -> None:
    """True stacked mathematical fraction for axis label, not raw LaTeX.

    Legend occupies a separate strip below the x label.
    """
    light = st.session_state.get('wl_chart_theme', 'Nền tối') == 'Nền sáng'
    color = '#26384B' if light else '#E6EFF7'
    if axis == 'x':
        fig.update_xaxes(title_text='')
        fig.add_layout_image(dict(source=fraction_svg(numerator, denominator, color),
            xref='paper', yref='paper', x=x, y=y,
            sizex=.09, sizey=.20, sizing='contain',
            xanchor='center', yanchor='top', layer='above'))
    else:
        fig.update_yaxes(title_text='')
        fig.add_layout_image(dict(source=fraction_svg(numerator, denominator, color),
            xref='paper', yref='paper', x=-.18, y=.5,
            sizex=.13, sizey=.18, sizing='contain',
            xanchor='center', yanchor='middle', layer='above'))


def section(title: str, subtitle: str | None = None):
    """Tiêu đề thống nhất giữa các vùng giao diện."""
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-desc">{subtitle}</div>', unsafe_allow_html=True)


def ffloat(x: float, n: int = 4) -> str:
    if math.isinf(x):
        return "∞"
    if math.isnan(x):
        return "—"
    return f"{x:,.{n}f}"


def plot_envelope(sol, show_marks: bool = True, height: int = 440):
    x, y = sol.d_lambda, np.abs(sol.v_total)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,y=sol.envelope_high, mode="lines", name="Đường bao trên |V⁺| + |V⁻|",
                             line=dict(color=COL["amber"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x,y=sol.envelope_low, mode="lines", name="Đường bao dưới |V⁺| − |V⁻|",
                             line=dict(color=COL["green"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=x,y=y, mode="lines", name="Biên độ |V(d)|",
                             line=dict(color=COL["blue"], width=3.2)))
    if show_marks:
        phase = phase_points(sol.cfg)
        ext = stationary_points(sol.cfg)
        for k, sym, clr, nm in [("max","triangle-up",COL["amber"],"Điểm đồng pha"),
                                ("min","triangle-down",COL["green"],"Điểm ngược pha")]:
            coords = phase[k]
            if len(coords):
                y_marks = np.interp(coords, sol.d_m, y)
                fig.add_trace(go.Scatter(x=coords/sol.cfg.wavelength_m,y=y_marks,
                                         mode="markers",name=nm, marker=dict(symbol=sym,color=clr,size=9)))
        for k, sym, clr, nm in [("max","circle",COL["coral"],"Cực đại biên độ"),
                                ("min","x",COL["violet"],"Cực tiểu biên độ")]:
            coords = ext[k]
            if len(coords):
                yy = np.interp(coords,sol.d_m,y)
                fig.add_trace(go.Scatter(x=coords/sol.cfg.wavelength_m,y=yy,
                    mode="markers",name=nm,marker=dict(symbol=sym,color=clr,size=9,
                                                        line=dict(width=1.5,color="#11151e"))))
    style(fig,"Biên độ điện áp dọc đường dây",height)
    add_fraction_axis(fig)
    fig.update_yaxes(title="Biên độ điện áp chuẩn hóa",rangemode="tozero")
    return fig


def plot_current(sol, show_voltage_reference: bool = True):
    """|I| và hai đường bao dòng điện; chỉ đổi biểu diễn, không đổi nghiệm solve().

    Với Z₀ thuần trở, J(d) = Z₀ I(d)/|V⁺ tại nguồn| = V⁺(d) − V⁻(d).
    Bất đẳng thức tam giác cho bao trên |V⁺|+|V⁻| và bao dưới
    ||V⁺|−|V⁻||, cùng chuẩn hóa. Đây là đường bao cục bộ, không
    được hiểu là cặp cực trị toàn tuyến khi đường dây có suy hao.
    Cơ sở: Pozar, Microwave Engineering, 4th ed., Ch. 2, §§2.3, 2.7.
    """
    i_abs = np.abs(sol.i_total_norm)
    incident = np.abs(sol.v_plus)
    reflected = np.abs(sol.v_minus)
    i_upper = incident + reflected
    i_lower = np.abs(incident - reflected)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sol.d_lambda, y=i_upper, mode="lines", name="Bao trên dòng điện",
        line=dict(color=COL["amber"], width=2, dash="dash"),
        hovertemplate="Vị trí chuẩn hóa: %{x:.4f}<br>Bao trên: %{y:.5f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=sol.d_lambda, y=i_lower, mode="lines", name="Bao dưới dòng điện",
        line=dict(color=COL["green"], width=2, dash="dash"),
        hovertemplate="Vị trí chuẩn hóa: %{x:.4f}<br>Bao dưới: %{y:.5f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=sol.d_lambda, y=i_abs, mode="lines", name="Biên độ dòng điện",
        line=dict(color=COL["violet"], width=3.2),
        hovertemplate="Vị trí chuẩn hóa: %{x:.4f}<br>Dòng điện chuẩn hóa: %{y:.5f}<extra></extra>"))
    if show_voltage_reference:
        fig.add_trace(go.Scatter(
            x=sol.d_lambda, y=np.abs(sol.v_total), mode="lines",
            name="Điện áp (đối chiếu)",
            line=dict(color=COL["blue"], width=1.8, dash="dot")))
    style(fig, "Đường bao biên độ dòng điện", 440)
    add_fraction_axis(fig)
    fig.update_yaxes(title="Biên độ chuẩn hóa", rangemode="tozero")
    return fig


def true_voltage_nodes(sol: LineSolution) -> NDArray[np.float64]:
    """Các điểm có điện áp bằng không ở *mọi thời điểm*, đơn vị d/λ.

    Điểm ngược pha không tự động là nút: nó chỉ là nút đúng nghĩa khi
    hai sóng có cùng mô-đun. Trên đường dây suy hao, chỉ có thể còn
    nút điện áp tại chính tải ngắn mạch d=0 (nếu nằm trong đoạn).
    Không đánh dấu cực tiểu khác không như một nút sóng thực sự.
    """
    cfg = sol.cfg
    if not np.isclose(abs(cfg.gamma_load), 1.0, atol=1e-12, rtol=0):
        return np.array([], dtype=float)
    candidates = phase_points(cfg)["min"]
    if cfg.alpha_np_m > 0:
        candidates = candidates[np.isclose(candidates, 0, atol=1e-12 * cfg.wavelength_m, rtol=0)]
    if not len(candidates):
        return np.array([], dtype=float)
    # Kiểm chứng trực tiếp biên độ phasor tại mỗi điểm trước khi đánh dấu.
    vv = (np.exp(-cfg.gamma * (cfg.length_m - candidates))
          + cfg.gamma_load * np.exp(-cfg.gamma * (cfg.length_m + candidates)))
    tolerance = 1e-9 * max(1.0, float(np.max(np.abs(sol.v_total))))
    return candidates[np.abs(vv) <= tolerance] / cfg.wavelength_m


def plot_gamma(sol):
    fig=go.Figure()
    a=np.linspace(0,2*np.pi,200)
    fig.add_trace(go.Scatter(x=np.cos(a),y=np.sin(a),mode="lines",name="Vòng |Γ| = 1",
                             line=dict(color="#6d7886",dash="dot")))
    fig.add_trace(go.Scatter(x=sol.gamma_d.real,y=sol.gamma_d.imag,
                             mode="lines",name="Quỹ đạo Γ(d)",line=dict(color=COL["cyan"],width=3)))
    fig.add_trace(go.Scatter(x=[sol.gamma_d[0].real,sol.gamma_d[-1].real],
                             y=[sol.gamma_d[0].imag,sol.gamma_d[-1].imag],mode="markers",
                             name="Tải và nguồn",marker=dict(size=[13,11],color=[COL["coral"],COL["violet"]],
                                                          line=dict(width=1,color="white"))))
    style(fig,"Quỹ đạo hệ số phản xạ Γ(d)",430)
    fig.update_xaxes(title="Re{Γ}",range=[-1.13,1.13],zeroline=True,scaleanchor="y",scaleratio=1)
    fig.update_yaxes(title="Im{Γ}",range=[-1.13,1.13],zeroline=True)
    fig.update_layout(hovermode="closest", legend=dict(y=-.32))
    return fig


def plot_zin(sol):
    vals=sol.zin
    real=vals.real.copy(); imag=vals.imag.copy()
    finite=(np.isfinite(real) & np.isfinite(imag) & (np.abs(vals) <= sol.cfg.z0_ohm*30))
    real[~finite]=np.nan; imag[~finite]=np.nan
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=sol.d_lambda,y=real,mode="lines",name="Re(Zᵢₙ) (Ω)",
                             line=dict(color=COL["blue"],width=2.6)))
    fig.add_trace(go.Scatter(x=sol.d_lambda,y=imag,mode="lines",name="Im(Zᵢₙ) (Ω)",
                             line=dict(color=COL["coral"],width=2)))
    fig.add_hline(y=sol.cfg.z0_ohm,line=dict(color="#637385",dash="dash",width=1))
    style(fig,"Trở kháng vào theo vị trí",430)
    add_fraction_axis(fig)
    fig.update_yaxes(title="Trở kháng (Ω)")
    return fig


def plot_vswr(sol):
    fig=go.Figure()
    val=np.asarray(sol.vswr_local).copy()
    val[~np.isfinite(val)]=np.nan
    fig.add_trace(go.Scatter(x=sol.d_lambda,y=val,mode="lines",name="VSWR cục bộ",
                             line=dict(color=COL["green"],width=3),fill="tozeroy",fillcolor="rgba(87,239,220,.10)"))
    style(fig,"Tỷ số sóng đứng cục bộ VSWR(d)",395)
    add_fraction_axis(fig)
    fig.update_yaxes(title="VSWR(d)",rangemode="tozero")
    return fig


def plot_motion(sol, show_nodes: bool = True):
    """Sóng thực theo thời gian; điểm nút được xác nhận vật lý trước khi tô sáng.

    Các frame Plotly có cùng số trace với khung đầu tiên; marker nút cố
    định ở y=0, không được di chuyển theo pha thời gian.
    """
    x=sol.d_lambda
    vmax=max(0.2,float(np.max(np.abs(sol.v_total)))*1.08,
             float(np.max(np.abs(sol.v_plus)))*1.08)
    nodes = true_voltage_nodes(sol) if show_nodes else np.array([], dtype=float)
    phases=np.linspace(0,2*np.pi,49)[:-1]
    def traces(phase):
        e=np.exp(1j*phase)
        data = [go.Scatter(x=x,y=np.real(sol.v_plus*e),mode="lines",name="Sóng tới",
                           line=dict(color=COL["cyan"],width=2,dash="dash")),
                go.Scatter(x=x,y=np.real(sol.v_minus*e),mode="lines",name="Sóng phản xạ",
                           line=dict(color=COL["amber"],width=2,dash="dot")),
                go.Scatter(x=x,y=np.real(sol.v_total*e),mode="lines",name="Sóng tổng hợp",
                           line=dict(color=COL["blue"],width=3.3))]
        if len(nodes):
            data.append(go.Scatter(
                x=nodes, y=np.zeros(len(nodes)), mode="markers", name="Nút điện áp (V = 0)",
                marker=dict(symbol="circle-open", size=11, color=COL["coral"],
                            line=dict(width=2.5, color=COL["coral"])),
                hovertemplate="Nút điện áp tại vị trí chuẩn hóa: %{x:.4f}<br>v(d,t) = 0 với mọi t<extra></extra>"))
        return data
    frames=[go.Frame(name=str(k),data=traces(p),layout=go.Layout(
                       annotations=[dict(text=f"Thời gian: {k/48:.2f} chu kỳ",xref="paper",yref="paper",
                                         x=.99,y=1.11,showarrow=False,font=dict(color=COL["muted"],size=13))]))
            for k,p in enumerate(phases)]
    fig=go.Figure(data=traces(0),frames=frames)
    style(fig,"Giao thoa sóng theo thời gian",445)
    if len(nodes):
        for point in nodes:
            fig.add_vline(x=float(point), line_width=1, line_dash="dot",
                          line_color=COL["coral"], opacity=0.32)
    fig.update_xaxes(range=[0,sol.cfg.length_wavelengths])
    add_fraction_axis(fig)
    fig.update_yaxes(title="Điện áp tức thời chuẩn hóa",range=[-vmax,vmax])
    fig.update_layout(
        annotations=[dict(text="Thời gian: 0,00 chu kỳ",xref="paper",yref="paper",x=.99,y=1.11,
                          showarrow=False,font=dict(color=COL["muted"],size=13))],
        updatemenus=[dict(type="buttons",direction="left",showactive=False,x=0,y=1.19,
                          buttons=[dict(label="▶ Phát",method="animate",args=[None,
                                   {"frame":{"duration":65,"redraw":True},"transition":{"duration":0},
                                    "fromcurrent":True,"mode":"immediate"}]),
                                   dict(label="⏸ Dừng",method="animate",args=[[None],
                                   {"frame":{"duration":0,"redraw":False},"transition":{"duration":0},
                                    "mode":"immediate"}])])],
        sliders=[dict(active=0,y=-.49,x=.08,len=.85,pad=dict(t=16),
                      currentvalue=dict(prefix="Pha thời gian: ",visible=False),
                      steps=[dict(args=[[str(k)],{"frame":{"duration":0,"redraw":True},
                                                    "transition":{"duration":0},"mode":"immediate"}],
                                  label=f"{k/48:.2f}" if k%8==0 else "",method="animate")
                             for k in range(48)])],
        legend=dict(orientation="h", x=.02, y=1.05, yanchor="bottom"),
        margin=dict(l=78,r=25,t=140,b=150))
    return fig


def plot_spacetime(sol):
    time=np.linspace(0,1,100)
    yy=np.real(np.exp(1j*2*np.pi*time[:,None])*sol.v_total[None,:])
    fig=go.Figure(data=go.Heatmap(z=yy,x=sol.d_lambda,y=time,
                                  colorscale=[[0,"#7D72F9"],[0.28,"#4AEBD7"],[0.5,"#18202A"],
                                              [0.72,"#B7D95D"],[1,"#F4FF62"]],zmid=0,colorbar=dict(title="v(t)")))
    style(fig,"Bản đồ sóng tổng hợp theo không gian và thời gian",390)
    add_fraction_axis(fig)
    add_fraction_axis(fig, axis="y", numerator="t", denominator="T")
    fig.update_layout(hovermode="closest", margin=dict(l=100, r=45, t=76, b=105), showlegend=False)
    return fig


def plot_case(cases):
    fig=make_subplots(rows=2,cols=2,subplot_titles=list(cases.keys()),vertical_spacing=.16,
                      horizontal_spacing=.11)
    for idx, (name,conf) in enumerate(cases.items()):
        sol=solve(conf)
        row,col=idx//2+1,idx%2+1
        fig.add_trace(go.Scatter(x=sol.d_lambda,y=np.abs(sol.v_total),mode="lines",
                        name=name,line=dict(color=[COL["blue"],COL["coral"],COL["violet"],COL["green"]][idx],width=2.4),
                        showlegend=False),row=row,col=col)
        fig.update_xaxes(title_text="Khoảng cách chuẩn hóa",row=row,col=col)
        fig.update_yaxes(title_text="|V|",row=row,col=col,rangemode="tozero")
    style(fig,"So sánh bốn kịch bản",620)
    fig.update_layout(margin=dict(l=32,r=24,t=75,b=40))
    return fig


def plot_single_case(conf: LineConfig, color: str, height: int = 335) -> go.Figure:
    """One responsive scenario chart. Data and physics are the same as plot_case()."""
    sol_case = solve(conf)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sol_case.d_lambda, y=np.abs(sol_case.v_total), mode="lines",
        name="Biên độ điện áp", line=dict(color=color, width=2.8),
        hovertemplate="Vị trí chuẩn hóa: %{x:.3f}<br>|V| = %{y:.4f}<extra></extra>",
    ))
    style(fig, None, height)
    fig.update_xaxes(fixedrange=False)
    add_fraction_axis(fig)
    fig.update_yaxes(title="|V| (chuẩn hóa)", rangemode="tozero", fixedrange=False)
    fig.update_layout(showlegend=False, margin=dict(l=68, r=24, t=25, b=98))
    return fig


# MÃ QR TRUY CẬP WEB ---------------------------------------------------------
# QR chỉ mã hóa URL do người dùng cấu hình; không tự đoán địa chỉ localhost.
# Mọi hàm trong khối này chỉ phục vụ giao diện, không ảnh hưởng phép tính VSWR.


def valid_web_address(address: str) -> str:
    """Chỉ chấp nhận một đường dẫn HTTP(S) đầy đủ, không có thông tin đăng nhập."""
    address = (address or "").strip()
    try:
        parsed = urlsplit(address)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            return ""
        _ = parsed.port  # Reject invalid ports.
    except ValueError:
        return ""
    return address


def is_localhost_url(address: str) -> bool:
    host = (urlsplit(address).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


@st.cache_data(show_spinner=False)
def qr_png_bytes(url: str) -> bytes:
    """Tạo QR ngay trên máy chủ, không gửi URL sang dịch vụ tạo QR bên ngoài."""
    import qrcode
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=9,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    result = qr.make_image(fill_color="#102333", back_color="#ffffff")
    output = BytesIO()
    result.save(output, format="PNG")
    return output.getvalue()


# SIDEBAR --------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-brand"><strong>WaveLab</strong><small>Mô phỏng đường dây siêu cao tần</small></div>', unsafe_allow_html=True)
    with st.expander("Mã QR truy cập Web", expanded=bool(WEB_APP_URL.strip())):
        st.caption("Dán liên kết công khai của WaveLab sau khi triển khai lên đám mây.")
        entered_url = st.text_input(
            "Địa chỉ Web (URL)",
            value=WEB_APP_URL,
            placeholder="https://ten-ung-dung.streamlit.app/",
            key="wl_qr_url_input",
            help="Nhập tại đây để xem thử QR trong phiên hiện tại. "
                 "Muốn QR hiện mặc định cho tất cả người truy cập, "
                 "dán URL vào biến WEB_APP_URL ở đầu file Python.",
        )
        active_qr_url = valid_web_address(entered_url)
        if not entered_url.strip():
            st.info("Chưa có liên kết Web công khai. Khi triển khai xong, hãy dán URL vào ô phía trên.")
        elif not active_qr_url:
            st.warning("Đường dẫn không hợp lệ. Cần nhập đầy đủ URL bắt đầu bằng https:// hoặc http://.")
        elif is_localhost_url(active_qr_url):
            st.warning("Địa chỉ localhost không thể mở từ điện thoại khác. Hãy dùng URL công khai sau khi triển khai.")
        else:
            try:
                png_data = qr_png_bytes(active_qr_url)
            except ImportError:
                st.info('Cài thêm thư viện mã QR: python -m pip install "qrcode[pil]"')
            else:
                left, center, right = st.columns([1, 5, 1])
                with center:
                    st.image(png_data, width=185, alt="Mã QR dẫn đến WaveLab VSWR")
                st.caption("Mã QR dẫn đến:")
                st.code(active_qr_url, language=None, wrap_lines=True)
                st.download_button(
                    "Tải mã QR (.png)", data=png_data,
                    file_name="WaveLab_VSWR_QR.png", mime="image/png",
                    use_container_width=True, key="wl_qr_download",
                )
        st.caption("Lưu ý: URL nhập trong ô chỉ áp dụng cho phiên đang mở; "
                   "điền WEB_APP_URL trong mã nguồn để cố định link cho mọi người dùng.")
    st.markdown('<div class="side-head">1. Thông số đường truyền</div>', unsafe_allow_html=True)
    freq_ghz=st.number_input("Tần số f (GHz)",min_value=.10,max_value=60.0,value=1.0,step=.1,
                              format="%.2f",help="f > 0")
    vp_mill=st.number_input("Vận tốc pha vₚ (10⁸ m/s)",min_value=.10,max_value=2.99792458,
                            value=2.0,step=.05,format="%.4f",help="Mô hình TEM: 0 < vₚ ≤ c₀")
    z0=st.number_input("Trở kháng đặc tính Z₀ (Ω)",min_value=1.0,max_value=1000.0,
                       value=50.0,step=5.0)
    leng=st.slider("Chiều dài đường dây (λ)",min_value=.1,max_value=8.0,value=2.0,step=.1)
    alpha=st.number_input("Hệ số suy hao α (Np/m)",min_value=0.0,max_value=20.0,value=0.0,
                          step=.05,format="%.3f",help="α=0: lý tưởng; α>0: mô hình đường dây suy hao thấp")
    st.markdown('<div class="side-head">2. Điều kiện tải</div>', unsafe_allow_html=True)
    label_map={"Tải tổng quát":"general","Tải phối hợp (Zₗ = Z₀)":"matched",
               "Tải ngắn mạch (Zₗ = 0)":"short","Tải hở mạch (Zₗ → ∞)":"open",
               "Tải thuần phản kháng (Zₗ = jXₗ)":"reactive"}
    label=st.radio("Chuyển chế độ tải",list(label_map.keys()),index=0,label_visibility="collapsed")
    mode=label_map[label]
    if mode=="general":
        rl=st.number_input("Điện trở tải Rₗ (Ω)",min_value=0.0,max_value=5000.0,
                           value=50.0,step=5.0)
        xl=st.number_input("Điện kháng tải Xₗ (Ω)",min_value=-5000.0,max_value=5000.0,
                           value=50.0,step=5.0)
    elif mode=="reactive":
        xl=st.number_input("Điện kháng tải Xₗ (Ω), khác 0",min_value=-5000.0,max_value=5000.0,
                           value=50.0,step=5.0)
        rl=0.0
    elif mode=="matched":
        rl,xl=z0,0.0
        st.caption("Zₗ = Z₀.")
    else:
        rl,xl=0.0,0.0
        st.caption("Sử dụng điều kiện biên lý tưởng.")
    st.markdown('<div class="side-head">3. Độ phân giải</div>', unsafe_allow_html=True)
    samples=st.slider("Số điểm lấy mẫu N",201,5001,1601,step=200,
                      help="Tăng N để giảm bước lưới không gian và sai số dò cực trị.")
    st.markdown('<div class="side-head">4. Hiển thị đồ thị</div>', unsafe_allow_html=True)
    chart_theme = st.radio("Giao diện biểu đồ", ["Nền tối", "Nền sáng"],
                           key="wl_chart_theme", horizontal=True,
                           help="Chỉ đổi phong cách biểu đồ; không đổi công thức hay dữ liệu.")
    st.caption("Nền sáng phù hợp khi chụp đồ thị cho báo cáo.")
    with st.expander("Hướng dẫn sử dụng", expanded=False):
        st.markdown("""1. Nhập tham số và chọn chế độ tải.
2. Chọn mục cần khảo sát.
3. Dùng nút **Phát** trong phần Sóng và giao thoa; xuất CSV tại phần Kiểm chứng.""")

# Bảng màu theo nền biểu đồ; không sửa các mảng dữ liệu hoặc hàm tính.
if st.session_state.get("wl_chart_theme") == "Nền sáng":
    COL.update({"cyan": "#008C86", "blue": "#4159AD", "amber": "#AA6A03",
                "coral": "#BD4761", "violet": "#6F50AD", "green": "#0A816E",
                "ink": "#1B2C3F", "muted": "#45596D"})

try:
    cfg=LineConfig(f_hz=freq_ghz*1e9,vp_mps=vp_mill*1e8,z0_ohm=z0,
                   length_wavelengths=leng,alpha_np_m=alpha,mode=mode,
                   r_load_ohm=rl,x_load_ohm=xl,samples=samples)
    sol=solve(cfg)
except (ValueError,ZeroDivisionError,FloatingPointError) as exc:
    st.error(f"Không thể mô phỏng với bộ tham số này: {exc}")
    st.stop()

rho=abs(cfg.gamma_load)
vswr_load=float(safe_vswr(rho))
vswr_in=float(safe_vswr(abs(sol.gamma_d[-1])))
rloss=return_loss_db(rho)
mloss=mismatch_loss_db(rho)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">WAVELAB · PHÒNG THÍ NGHIỆM SIÊU CAO TẦN</div>
  <div class="hero-title">Mô phỏng <span>tỷ số sóng đứng VSWR</span></div>
  <div class="hero-sub">Khảo sát sự phản xạ trên đường truyền, phân bố điện áp, trở kháng vào và sự hình thành sóng đứng. Điều chỉnh thông số ở thanh bên để quan sát kết quả.</div>
  <div class="hero-features">
    <div class="hero-feature"><span class="hero-feature-index">01</span><span class="hero-feature-label">Phân bố điện áp</span></div>
    <div class="hero-feature"><span class="hero-feature-index">02</span><span class="hero-feature-label">Sóng &amp; giao thoa</span></div>
    <div class="hero-feature"><span class="hero-feature-index">03</span><span class="hero-feature-label">Hệ số phản xạ Γ</span></div>
    <div class="hero-feature"><span class="hero-feature-index">04</span><span class="hero-feature-label">Kiểm chứng số</span></div>
  </div>
</div>
""",unsafe_allow_html=True)

# Điều hướng 5 nút thực, đảm bảo mỗi mục là một khối độc lập.
# Đây chỉ là state của giao diện; không thay đổi các hàm tính toán.
PAGE_CHOICES = (
    ("overview", "01 · Tổng quan", "Phân bố điện áp và VSWR"),
    ("waves", "02 · Giao thoa", "Sóng tới và sóng phản xạ"),
    ("impedance", "03 · Trở kháng", "Hệ số phản xạ và trở kháng vào"),
    ("scenarios", "04 · Kịch bản", "Các kịch bản khảo sát"),
    ("verify", "05 · Kiểm chứng", "Đối chiếu và xuất dữ liệu"),
)

if "wl_active_page" not in st.session_state:
    st.session_state["wl_active_page"] = "overview"


def switch_page(page_id: str) -> None:
    st.session_state["wl_active_page"] = page_id


st.markdown('<div class="nav-kicker">Chọn nội dung mô phỏng</div>',
            unsafe_allow_html=True)
nav_columns = st.columns(5, gap="small")
for column, (page_id, page_label, page_help) in zip(nav_columns, PAGE_CHOICES):
    with column:
        st.button(
            page_label,
            key=f"wl_page_{page_id}",
            type="primary" if st.session_state["wl_active_page"] == page_id else "secondary",
            use_container_width=True,
            on_click=switch_page,
            args=(page_id,),
        )
active_page = st.session_state["wl_active_page"]


st.markdown(f"""<div class="info-rail">
 <div class="info-chip"><b>f</b> = {freq_ghz:.2f} GHz</div>
 <div class="info-chip"><b>Z₀</b> = {z0:g} Ω</div>
 <div class="info-chip"><b>L</b> = {leng:g} λ</div>
 <div class="info-chip"><b>α</b> = {alpha:g} Np/m</div>
 <div class="info-chip"><b>Chế độ:</b> {label}</div>
</div>""",unsafe_allow_html=True)

if rho < 1e-13:
    st.caption("Tải phối hợp: không có sóng phản xạ.")
elif rho >= 1-1e-13:
    st.caption("Phản xạ toàn phần: VSWR tại tải tiến tới vô hạn.")
else:
    st.caption("Phản xạ một phần: có sóng tới và sóng phản xạ.")

if alpha > 0:
    st.info("**Đường dây có suy hao:** VSWR(d) được tính từ tỷ số hai đường bao cục bộ tại cùng vị trí, không dùng tỷ số cực đại và cực tiểu biên độ trên toàn tuyến.")
if alpha>0 and alpha/cfg.beta_rad_m >= 0.1:
    st.warning(r"Tỷ số $\frac{\alpha}{\beta}$ tương đối lớn: giả thiết đường dây suy hao thấp với $Z_0$ thuần trở có thể không còn phù hợp. Hãy giảm $\alpha$ hoặc sử dụng mô hình RLGC tổng quát (ngoài phạm vi chương trình).")

st.markdown('<div class="nav-kicker">Thông số phản xạ</div>', unsafe_allow_html=True)
m=st.columns(5)
for col, title, value, foot in zip(m,
    ["Hệ số phản xạ |Γₗ|", "VSWR tại tải", "VSWR tại nguồn", "Suy hao phản xạ RL", "Tổn hao không phối hợp (ML)"],
    [ffloat(rho),ffloat(vswr_load,3),ffloat(vswr_in,3),ffloat(rloss,2)+" dB",ffloat(mloss,2)+" dB"],
    ["Tại tải (d = 0)","Tại tải (d = 0)","Tại nguồn (d = L)", "Tại tải · dB","Tại tải · dB"]):
    col.metric(title,value,help=foot)


if active_page == "overview":
    section("Phân bố điện áp", "Biên độ điện áp, đường bao và VSWR dọc đường truyền.")
    display_marks = st.toggle("Hiển thị vị trí đồng pha và các điểm cực trị", value=True,
                              help="Thay đổi phần hiển thị, không ảnh hưởng đến kết quả tính toán.")
    st.plotly_chart(plot_envelope(sol,show_marks=display_marks),use_container_width=True,config=PLOTLY_CONFIG)
    a,b=st.columns([1.05,.95],gap="large")
    with a:
        st.plotly_chart(plot_vswr(sol),use_container_width=True,config=PLOTLY_CONFIG)
    with b:
        st.markdown("#### Thông số kỹ thuật")
        results={
            "Bước sóng λ":f"{cfg.wavelength_m:.6g} m",
            "Chiều dài vật lý L":f"{cfg.length_m:.6g} m",
            "β":f"{cfg.beta_rad_m:.6g} rad/m",
            "γ = α + jβ":compact_complex(cfg.gamma,5)+" /m",
            "Tải Zᴸ":compact_complex(cfg.load_impedance,3)+" Ω" if cfg.load_impedance is not None else "Hở mạch lý tưởng",
            "Γᴸ":compact_complex(cfg.gamma_load,5),
            "Góc pha Γᴸ": "Không xác định (Γ=0)" if rho<1e-13 else f"{np.degrees(np.angle(cfg.gamma_load)):.3f}°",
            "Zᵢₙ tại nguồn": compact_complex(sol.zin[-1],3)+" Ω",
            "Công suất phản xạ tại tải":f"{100*rho*rho:.3f}% công suất tới tải",
            "Tỷ số công suất sóng tới (tải so với nguồn)":f"{100*np.exp(-2*alpha*cfg.length_m):.3f}%",
        }
        st.dataframe(pd.DataFrame([{"Đại lượng":key,"Giá trị":value} for key,value in results.items()]),
                     hide_index=True,use_container_width=True)
    st.caption("Các tỷ số công suất được xác định với Z₀ thuần trở, chưa xét nguồn phát và mạch phối hợp.")

if active_page == "waves":
    section("Sóng tới và sóng phản xạ", "Quan sát giao thoa của hai sóng truyền ngược chiều trong miền thời gian.")
    show_true_nodes = st.toggle("Đánh dấu nút điện áp thực (biên độ bằng 0)", value=True,
                                help="Chỉ đánh dấu những điểm có điện áp bằng 0 ở mọi thời điểm; không nhầm với cực tiểu khác không.")
    node_positions = true_voltage_nodes(sol) if show_true_nodes else np.array([], dtype=float)
    st.plotly_chart(plot_motion(sol, show_nodes=show_true_nodes),
                    use_container_width=True, config=PLOTLY_CONFIG)
    if show_true_nodes:
        if len(node_positions):
            node_text = ", ".join(f"{d:.3f}λ" for d in node_positions[:7])
            if len(node_positions) > 7:
                node_text += f", … ({len(node_positions)} điểm)"
            st.caption(f"Nút điện áp xác định được (tính từ tải): {node_text}. Dấu tròn và đường thẳng chấm thể hiện nút đứng yên.")
        else:
            st.caption("Không có nút điện áp thực trong đoạn khảo sát. Các cực tiểu khác không không được đánh dấu thành nút sóng.")
    st.caption("Sóng tới truyền về tải (d giảm), sóng phản xạ truyền về nguồn (d tăng). Sử dụng nút Phát hoặc Dừng để điều khiển hoạt ảnh.")
    ca,cb=st.columns(2,gap="large")
    with ca:
        show_volt_reference = st.toggle("Hiển thị điện áp để đối chiếu", value=False)
        st.plotly_chart(plot_current(sol, show_voltage_reference=show_volt_reference),
                        use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("Dòng điện chuẩn hóa theo biên độ sóng tới tại nguồn. Hai đường bao giới hạn được xác định từ mô-đun sóng tới và sóng phản xạ.")
        with st.expander("Công thức đường bao dòng điện", expanded=False):
            st.latex(r"\widetilde I(d)=\frac{Z_0|I(d)|}{|V_{\mathrm{in}}^+|}")
            st.latex(r"\widetilde I_{\mathrm{trên}}(d)=\frac{|V^+(d)|+|V^-(d)|}{|V_{\mathrm{in}}^+|}")
            st.latex(r"\widetilde I_{\mathrm{dưới}}(d)=\frac{\left||V^+(d)|-|V^-(d)|\right|}{|V_{\mathrm{in}}^+|}")
            st.caption("Giả thiết Z₀ thuần trở; suy ra từ quan hệ I = (V⁺ − V⁻)/Z₀. Pozar, Microwave Engineering (2012), chương 2.")
    with cb:
        st.plotly_chart(plot_spacetime(sol),use_container_width=True,config=PLOTLY_CONFIG)
    st.caption("Tải phối hợp không tạo sóng đứng, dù đường truyền có suy hao.")

if active_page == "impedance":
    section("Hệ số phản xạ và trở kháng vào", "Khoảng cách d đo từ tải về phía nguồn.")
    ca,cb=st.columns(2,gap="large")
    with ca:
        st.plotly_chart(plot_gamma(sol),use_container_width=True,config=PLOTLY_CONFIG)
    with cb:
        st.plotly_chart(plot_zin(sol),use_container_width=True,config=PLOTLY_CONFIG)
    st.caption("Giới hạn hiển thị: |Zᵢₙ| ≤ 30Z₀; các giá trị lớn hơn hoặc vô hạn được ẩn khỏi đồ thị.")
    st.markdown("#### Vị trí đồng pha và cực trị")
    phase=phase_points(cfg)
    ext=stationary_points(cfg)
    st.dataframe(pd.DataFrame([
        {"Loại vị trí":"Đồng pha (Γ thực dương)","Số điểm":len(phase["max"]),
         "3 tọa độ đầu (chuẩn hóa theo λ)":", ".join(f"{v/cfg.wavelength_m:.4f}" for v in phase["max"][:3]) or "—"},
        {"Loại vị trí":"Ngược pha (Γ thực âm)","Số điểm":len(phase["min"]),
         "3 tọa độ đầu (chuẩn hóa theo λ)":", ".join(f"{v/cfg.wavelength_m:.4f}" for v in phase["min"][:3]) or "—"},
        {"Loại vị trí":"Cực đại |V| thực tế, trong đoạn", "Số điểm":len(ext["max"]),
         "3 tọa độ đầu (chuẩn hóa theo λ)":", ".join(f"{v/cfg.wavelength_m:.4f}" for v in ext["max"][:3]) or "—"},
        {"Loại vị trí":"Cực tiểu |V| thực tế, trong đoạn", "Số điểm":len(ext["min"]),
         "3 tọa độ đầu (chuẩn hóa theo λ)":", ".join(f"{v/cfg.wavelength_m:.4f}" for v in ext["min"][:3]) or "—"},
    ]),hide_index=True,use_container_width=True)
    dcrit=critical_distance(cfg)
    if dcrit is not None:
        st.caption(f"Điều kiện cần cho cực trị: d < d_crit = {dcrit/cfg.wavelength_m:.4f} λ. Điều kiện này **không đủ**; nghiệm còn phụ thuộc pha và chiều dài hữu hạn.")

if active_page == "scenarios":
    section("So sánh các kịch bản", "Bốn chế độ khảo sát theo đề cương.")
    # Keep f, vp, and Z0 fixed at canonical values from the original outline.
    base=LineConfig(f_hz=1e9,vp_mps=2e8,z0_ohm=50,length_wavelengths=2,
                    samples=1001,alpha_np_m=0)
    cases={
        "1 · Tải phối hợp (50 Ω)": replace(base,mode="matched"),
        "2 · Tải ngắn mạch": replace(base,mode="short"),
        "3 · Tải phức (50 + j50 Ω)": replace(base,mode="general",r_load_ohm=50,x_load_ohm=50),
        "4 · Đường dây có suy hao": replace(base,mode="general",r_load_ohm=25,x_load_ohm=35,
                                               alpha_np_m=.4,length_wavelengths=4),
    }
    scenario_items = list(cases.items())
    scenario_colors = [COL["blue"], COL["coral"], COL["violet"], COL["green"]]
    for row_start in (0, 2):
        scenario_cols = st.columns(2, gap="medium")
        for col_idx, column in enumerate(scenario_cols):
            idx = row_start + col_idx
            scenario_name, scenario_cfg = scenario_items[idx]
            with column:
                with st.container(border=True):
                    st.markdown(f'<div class="scenario-title">{scenario_name}</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(plot_single_case(scenario_cfg, scenario_colors[idx]),
                                    use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown("#### Trường hợp mở rộng")
    subs={
        "2b · Tải hở mạch":replace(base,mode="open"),
        "3a · Tải 50 + j50 Ω; L = 0,5λ":replace(base,mode="general",r_load_ohm=50,x_load_ohm=50,length_wavelengths=.5),
        "3b · Tải 50 + j50 Ω; L = 0,2λ":replace(base,mode="general",r_load_ohm=50,x_load_ohm=50,length_wavelengths=.2),
    }
    sc=st.columns(3)
    for c,(nm,cf) in zip(sc,subs.items()):
        s=solve(cf)
        with c:
            with st.container(border=True):
                st.markdown(f'<div class="scenario-title">{nm}</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_single_case(cf, COL["cyan"], height=320),
                                use_container_width=True, config=PLOTLY_CONFIG)
    rows=[]
    for name,cfg_item in {**cases,**subs}.items():
        si=solve(cfg_item)
        rows.append({"Kịch bản":name,"Zᴸ (Ω)":compact_complex(cfg_item.load_impedance,2),
                     "α (Np/m)":cfg_item.alpha_np_m,"VSWR tại tải":ffloat(float(safe_vswr(abs(cfg_item.gamma_load))),3),
                     "VSWR tại nguồn":ffloat(float(si.vswr_local[-1]),3)})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

if active_page == "verify":
    section("Kiểm chứng số", "Đối chiếu nghiệm giải tích với phương pháp ma trận truyền ABCD.")
    with st.expander("Quy trình thuật toán tương ứng mục 3.1", expanded=False):
        st.markdown("""**Nhập tham số:** tần số, vận tốc pha, trở kháng đặc tính, trở kháng tải, chiều dài và suy hao.

**Kiểm tra điều kiện biên:** giới hạn tham số; tải phối hợp, hở mạch, ngắn mạch, thuần phản kháng và tải tổng quát.

**Tính đại lượng trung gian:** bước sóng, hằng số pha, hằng số truyền và hệ số phản xạ tại tải.

**Rời rạc hóa và giải nghiệm:** điện áp, dòng điện, hệ số phản xạ và trở kháng tại từng tọa độ.

**Trình bày kết quả:** VSWR, đường bao điện áp/dòng điện, animation và 4 nhóm kịch bản.

**Kiểm chứng số:** so sánh nghiệm sóng với ma trận ABCD; đánh giá cực trị, bước lưới và xuất CSV.""")
    st.caption("Điện áp và dòng điện được tính bằng nghiệm sóng và ghép ma trận ABCD theo từng đoạn Δd.")
    stats=verification(sol)
    c1,c2,c3=st.columns(3)
    c1.metric("Sai số điện áp ABCD (tương đối)",f"{stats['voltage_abcd_rel']:.2e}")
    c2.metric("Sai số dòng điện ABCD (tương đối)",f"{stats['current_abcd_rel']:.2e}")
    c3.metric("Sai số tuyệt đối Γ",f"{stats['gamma_abcd_abs']:.2e}")
    st.caption("Sai số chuẩn hóa theo biên độ lớn nhất; dòng điện nhân Z₀. Đây là kiểm chứng tính toán, không phải đo thực nghiệm.")
    st.markdown("#### Sai số xác định cực trị")
    peaks=numerical_grid_peaks(sol)
    exact=stationary_points(cfg)
    tab=[]
    for nm,k in [("Cực đại", "max"),("Cực tiểu","min")]:
        match=stats["extrema_counts"]["matched"][k]
        errs=stats["extrema_errors_m"][k]
        tab.append({"Loại":"|V| "+nm,"Số nghiệm đạo hàm":len(exact[k]),
                    "Số điểm lưới phát hiện":len(peaks[k]),"Số ghép đúng cặp":match,
                    "Sai số ghép lớn nhất (m)":f"{max(errs):.3e}" if errs else "—"})
    st.dataframe(pd.DataFrame(tab),hide_index=True,use_container_width=True)
    step_m = sol.grid_step_m
    if step_m >= 0.01:
        step_value, step_unit = step_m, "m"
    elif step_m >= 1e-5:
        step_value, step_unit = step_m * 1000, "mm"
    elif step_m >= 1e-8:
        step_value, step_unit = step_m * 1e6, "µm"
    else:
        step_value, step_unit = step_m * 1e9, "nm"
    step_txt = f"{step_value:.4f}".rstrip("0").rstrip(".").replace(".", ",")
    step_lambda_txt = f"{step_m/cfg.wavelength_m:.7f}".rstrip("0").rstrip(".").replace(".", ",")
    st.markdown("**Bước lưới không gian**")
    st.latex(r"\Delta d = \frac{L}{N-1}")
    st.markdown(
        f'<div class="grid-step"><span class="grid-number">{step_txt} {step_unit}</span>'
        f' &nbsp;=&nbsp; <span class="grid-number">{step_lambda_txt} λ</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(r"Nghiệm đạo hàm được tìm bằng Brent trên khoảng đổi dấu; điểm lưới được dò bằng so sánh với lân cận. Không giả định sai số dò đỉnh luôn $\leq \frac{\Delta d}{2}$. Các nghiệm sát biên hoặc dao động bị lấy mẫu quá thưa có thể không phát hiện được.")
    if math.isinf(vswr_load):
        st.info("VSWR tại tải vô hạn: không tính sai số tương đối VSWR theo mẫu số ∞. Thay vào đó kiểm chứng Γ và phân bố sóng.")
    st.markdown("#### Phương trình xác định cực trị biên độ")
    st.latex(r"\sin(2\beta d-\theta_L)=\frac{\alpha}{2\beta}\left(\frac{1}{|\Gamma(d)|}-|\Gamma(d)|\right)")
    st.caption(r"Đây là phương trình suy ra từ đạo hàm bình phương biên độ điện áp; sau khi tìm nghiệm cần phân loại cực đại, cực tiểu và kiểm tra điều kiện $0<d<L$.")
    st.markdown("#### Dữ liệu và tài liệu tham khảo")
    export=pd.DataFrame({"d_m":sol.d_m,"d_lambda":sol.d_lambda,"V_abs_norm":np.abs(sol.v_total),
         "I_abs_norm":np.abs(sol.i_total_norm),"Gamma_real":sol.gamma_d.real,"Gamma_imag":sol.gamma_d.imag,
         "VSWR_local":sol.vswr_local,"Zin_real_ohm":sol.zin.real,"Zin_imag_ohm":sol.zin.imag})
    st.download_button("Tải dữ liệu mô phỏng (.csv)",export.to_csv(index=False).encode("utf-8-sig"),
                       file_name="vswr_simulation.csv",mime="text/csv")
    st.markdown("**Tài liệu tham khảo cho công thức gốc:**")
    st.markdown("""
- D. M. Pozar, *Microwave Engineering*, 4th ed. (2012), Chương 2, các mục 2.1, 2.3, 2.7 (đặc biệt (2.89)–(2.91)).
- M. Steer, [*Microwave and RF Design II: Transmission Lines*](https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Electronics/Microwave_and_RF_Design_II_-_Transmission_Lines_(Steer)), các mục 2.3 và 2.5.
- S. J. Orfanidis, [*Electromagnetic Waves and Antennas*, Chương 11](https://eceweb1.rutgers.edu/~orfanidi/ewa/ch11.pdf), mục 11.7 (ma trận truyền), mục 11.12 (sóng đứng).
""")
    st.markdown("<div class='note'><b>Giới hạn mô hình:</b> Đường truyền TEM đồng nhất, Z₀ thuần trở; trường hợp α > 0 dùng xấp xỉ suy hao thấp. Không mô phỏng đầy đủ mô hình RLGC với Z₀ phức.</div>",unsafe_allow_html=True)
