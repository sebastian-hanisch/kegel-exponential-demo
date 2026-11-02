"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein, Diagramme bauen an den Rändern."""

import numpy as np
import pytest

import kxp_constants as C
import kxp_evaluation as ev
import kxp_forms as F
import kxp_visualization as V


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def a():
    return ev.analyse(ev.Settings(n=12))


def test_shapes_chart_has_the_outlines_the_points_the_support_and_the_center(a):
    fig = V.build_shapes(a.cloud.points, a)
    names = [t.name for t in fig.data]
    assert names == ["Raute (p = 1)", "Kugel (p = 2)", "Würfel (p = ∞)", "ℓ3-Kugel", "Punkte", "Stützpunkte", "Mitte"] and _locked(fig)
    assert fig.layout.yaxis.scaleanchor == "x" and fig.layout.xaxis.scaleanchor == "y"
    assert len(fig.data[4].x) + len(fig.data[5].x) == 12 and len(fig.data[5].x) == ev.support_count(a.cloud.points, a.form.shape)
    # der Umriss der gewählten Kugel ist der Rand des Maßes
    x, y = np.array(fig.data[3].x), np.array(fig.data[3].y)
    assert np.allclose(F.gauge("lp", np.stack([x, y], axis=1) - a.form.shape.center, 3.0, 10.0), a.form.shape.radius, atol=1e-6)
    cube = fig.data[2]
    assert max(cube.x) - min(cube.x) == pytest.approx(2.0 * a.cube) or max(cube.y) - min(cube.y) == pytest.approx(2.0 * a.cube)       # Seitenlänge des Würfels = größte Spannweite


def test_shapes_chart_for_the_lse_family_names_beta():
    b = ev.analyse(ev.Settings(family="lse", beta=3.0, n=10))
    fig = V.build_shapes(b.cloud.points, b)
    assert fig.data[3].name == "log-sum-exp-Kugel (β = 3)"


def test_barriers_chart_has_both_formulations_and_a_log_t_axis():
    res = ev.two_barriers(ev.Settings(n=10))
    fig = V.build_barriers(res)
    assert _locked(fig) and fig.layout.xaxis.type == "log" and len(fig.data) == 2 and fig.data[0].name.startswith("Lorentz-Kegel (ν = ") and "Potenzkegel a = 1/2 (ν = " in fig.data[1].name
    assert list(fig.data[1].x) == [r["t"] for r in res["pow"]["path"]] and fig.data[1].y[-1] == pytest.approx(res["pow"]["path"][-1]["m"])


def test_lp_chart_has_the_curve_the_cube_line_and_the_two_markers(a):
    rows = ev.lp_family(ev.Settings(n=10), ps=(1.0, 2.0, 8.0))
    fig = V.build_lp(rows, a.cube, a.others["diamond"], a.others["ball"])
    assert _locked(fig) and fig.layout.xaxis.type == "log" and len(fig.data) == 3 and list(fig.data[0].x) == [1.0, 2.0, 8.0] and list(fig.data[2].x) == [1.0, 2.0]


def test_lse_chart_floors_the_distance_for_the_log_axis():
    rows = ev.lse_family(ev.Settings(n=10), betas=(0.5, 10.0, 1000.0))
    fig = V.build_lse(rows)
    assert _locked(fig) and fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log" and all(y > 0 for y in fig.data[0].y) and len(fig.data) == 2
    assert list(fig.data[1].y) == [r["bound"] for r in rows]


def test_symmetry_chart_colors_homogeneous_cones_differently():
    rows = ev.symmetry_rows(200)
    fig = V.build_symmetry(rows)
    colors = list(fig.data[0].marker.color)
    assert _locked(fig) and list(fig.data[0].y) == [3, 4, 4, 2, 2] and colors[:3] == [C.COLOR_OK] * 3 and colors[3:] == [C.COLOR_BAD] * 2


def test_nu_chart_has_both_families_left_and_the_step_factor_right():
    rows = ev.nu_rows(ns=(4, 6), seeds=(100000,))
    mu = ev.mu_rows(ev.Settings(n=10), mus=(2.0, 8.0))
    fig = V.build_nu(rows, mu)
    assert _locked(fig) and len(fig.data) == 3 and fig.layout.xaxis.type == "log" and list(fig.data[0].x) == [r["n"] for r in rows] and list(fig.data[0].customdata) == [r["lp"]["nu"] for r in rows] and list(fig.data[2].x) == [2.0, 8.0]
