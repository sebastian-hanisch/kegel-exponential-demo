"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kxp_constants as C
import kxp_forms as F


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=False):
    """Legende über dem Diagramm; bei Diagrammen mit Unterüberschriften darunter (sonst liegt sie auf den Überschriften)."""
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _positive(values, floor=1e-12):
    return [max(float(v), floor) for v in values]


def build_shapes(points: np.ndarray, a) -> go.Figure:
    """Punktwolke, die gewählte Kugel (Umriss), die Vergleichsformen Raute, Kugel und Würfel (gestrichelt, gleiche Mitte nur für den Würfel mit eigener Mitte)."""
    fig = go.Figure()
    s = a.settings
    sh = a.form.shape
    for label, p, color in (("Raute (p = 1)", 1.0, C.COLORS["diamond"]), ("Kugel (p = 2)", 2.0, C.COLORS["ball"])):
        other = F.solve_form("lp", points, p=p, tol=C.TOL).shape
        x, y = F.outline(other)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(color=color, width=1.5, dash="dot")))
    lo, hi = points.min(axis=0), points.max(axis=0)
    mid, half = (lo + hi) / 2.0, float(np.max(hi - lo)) / 2.0
    fig.add_trace(go.Scatter(x=[mid[0] - half, mid[0] + half, mid[0] + half, mid[0] - half, mid[0] - half], y=[mid[1] - half, mid[1] - half, mid[1] + half, mid[1] + half, mid[1] - half],
                             mode="lines", name="Würfel (p = ∞)", line=dict(color=C.COLORS["cube"], width=1.5, dash="dot")))
    x, y = F.outline(sh)
    name = f"ℓ{s.p:g}-Kugel" if s.family == "lp" else f"log-sum-exp-Kugel (β = {s.beta:g})"
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=C.COLORS[s.family], width=3)))
    dist = F.gauge(sh.family, points - sh.center, sh.p, sh.beta)
    on = dist >= sh.radius * (1.0 - 1e-5)
    fig.add_trace(go.Scatter(x=points[~on, 0], y=points[~on, 1], mode="markers", name="Punkte", marker=dict(size=7, color="#555555")))
    fig.add_trace(go.Scatter(x=points[on, 0], y=points[on, 1], mode="markers", name="Stützpunkte", marker=dict(size=11, color=C.COLOR_BAD, symbol="circle-open", line=dict(width=2))))
    fig.add_trace(go.Scatter(x=[sh.center[0]], y=[sh.center[1]], mode="markers", name="Mitte", marker=dict(size=9, color=C.COLORS[s.family], symbol="x")))
    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1), yaxis=dict(scaleanchor="x", scaleratio=1))
    return _base(fig, height=460)


def build_barriers(res: dict) -> go.Figure:
    """t · Fehler gegen t für die beiden Formulierungen derselben Kugel (logarithmische t-Achse)."""
    fig = go.Figure()
    for key, name, color in (("soc", f"Lorentz-Kegel (ν = {res['soc']['nu']})", C.COLORS["soc"]), ("pow", f"Potenzkegel a = 1/2 (ν = {res['pow']['nu']})", C.COLORS["lp"])):
        path = res[key]["path"]
        fig.add_trace(go.Scatter(x=[r["t"] for r in path], y=[r["m"] for r in path], mode="lines+markers", name=name, line=dict(color=color, width=3 if key == "pow" else 2),
                                 hovertemplate="t = %{x:.0e}<br>t · Fehler = %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="t", type="log", exponentformat="power")
    fig.update_yaxes(title_text="t · Fehler", rangemode="tozero")
    return _base(fig)


def build_lp(rows: list, cube: float, diamond: float, ball: float) -> go.Figure:
    """Radius gegen p (logarithmische p-Achse) mit dem Würfelradius als Linie und Raute und Kugel als Marken."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["p"] for r in rows], y=[r["radius"] for r in rows], mode="lines+markers", name="ℓp-Kugel (Potenzkegel)", line=dict(color=C.COLORS["lp"], width=3)))
    fig.add_trace(go.Scatter(x=[rows[0]["p"], rows[-1]["p"]], y=[cube, cube], mode="lines", name="Würfel (p = ∞)", line=dict(color=C.COLORS["cube"], dash="dot")))
    fig.add_trace(go.Scatter(x=[1.0, 2.0], y=[diamond, ball], mode="markers", name="Raute und Kugel", marker=dict(size=12, color=[C.COLORS["diamond"], C.COLORS["ball"]], symbol="diamond")))
    fig.update_xaxes(title_text="p", type="log", tickvals=[1, 2, 4, 8, 16, 32, 64, 128], ticktext=["1", "2", "4", "8", "16", "32", "64", "128"])
    fig.update_yaxes(title_text="Radius R", rangemode="tozero")
    return _base(fig)


def build_lse(rows: list) -> go.Figure:
    """Abstand zum Würfelradius gegen β (beide logarithmisch) mit der Schranke log(2d)/β."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["beta"] for r in rows], y=_positive(r["above_cube"] for r in rows), mode="lines+markers", name="gemessen (Boden 10⁻¹²)", line=dict(color=C.COLORS["lse"], width=3)))
    fig.add_trace(go.Scatter(x=[r["beta"] for r in rows], y=[r["bound"] for r in rows], mode="lines", name="Schranke log(2d)/β", line=dict(color=C.COLORS["bound"], dash="dash")))
    fig.add_hline(y=1e-7, line=dict(color="#888888", dash="dot"))
    fig.add_annotation(x=1.0, xref="paper", y=math.log10(1e-7), yref="y", text="Genauigkeit der Rechnung", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=11, color="#888888"))
    betas = [r["beta"] for r in rows]
    fig.update_xaxes(title_text="β", type="log", tickvals=betas, ticktext=[f"{b:g}" for b in betas])
    fig.update_yaxes(title_text="Radius minus Würfelradius", type="log", exponentformat="power", dtick=4)
    return _base(fig)


def build_symmetry(rows: list) -> go.Figure:
    """Dimension der Automorphismengruppe je Kegel; die Marke bei 3 trennt transitiv wirkende (homogene) Kegel von den anderen."""
    fig = go.Figure(go.Bar(x=[r["cone"] for r in rows], y=[r["dim"] for r in rows], marker_color=[C.COLOR_OK if r["homogeneous"] else C.COLOR_BAD for r in rows], text=[str(r["dim"]) for r in rows], textposition="outside",
                           hovertemplate="%{x}: Dimension %{y}<extra></extra>"))
    fig.add_hline(y=3, line=dict(color="#888888", dash="dash"), annotation_text="mindestens 3 nötig", annotation_position="top right")
    fig.update_yaxes(title_text="Dimension der Gruppe", range=[0, 5.2], dtick=1)
    fig.update_xaxes(tickangle=-20)
    return _base(fig, height=360)


def build_nu(rows: list, mu: list) -> go.Figure:
    """Links Newton-Schritte gegen ν für beide Familien, rechts gegen den Schrittfaktor μ."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Newton-Schritte gegen die Punktzahl (ν wächst mit)", "Newton-Schritte gegen den Schrittfaktor μ"))
    for fam, name in (("lp", "ℓp-Kugel (Potenzkegel)"), ("lse", "log-sum-exp-Kugel (Exponentialkegel)")):
        fig.add_trace(go.Scatter(x=[r["n"] for r in rows], y=[r[fam]["newton"] for r in rows], mode="lines+markers", name=name, line=dict(color=C.COLORS[fam], width=3),
                                 customdata=[r[fam]["nu"] for r in rows], hovertemplate="n = %{x}, ν = %{customdata}<br>Newton-Schritte %{y}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[r["mu"] for r in mu], y=[r["newton"] for r in mu], mode="lines+markers", name="gewählte Form", line=dict(color=C.COLORS["soc"], width=3)), row=1, col=2)
    ns = [r["n"] for r in rows]
    mus = [r["mu"] for r in mu]
    fig.update_xaxes(title_text="Punkte n (ν = 7n bzw. 13n)", type="log", tickvals=ns, ticktext=[str(v) for v in ns], row=1, col=1)
    fig.update_xaxes(title_text="μ", type="log", tickvals=mus, ticktext=[f"{m:g}" for m in mus], row=1, col=2)
    fig.update_yaxes(title_text="Newton-Schritte", rangemode="tozero")
    return _base(fig, legend_below=True)
