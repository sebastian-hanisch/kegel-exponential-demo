"""Die Formen (kxp_forms): Handwerte, Rückumformung, Umriss, Vergleich mit dem Lorentz-Kern."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kxp_barrier as B
import kxp_forms as F
from kxp_evaluation import cube_radius
from kxp_scenario import make_cloud


@pytest.mark.parametrize("p,expected", [(1.0, 3.0), (2.0, math.sqrt(5.0)), (3.0, 9.0 ** (1.0 / 3.0)), (4.0, 17.0 ** 0.25)])
def test_lp_ball_of_two_antipodal_points_by_hand(antipodal, p, expected):
    """Mittelpunkt 0 und Radius ‖(1, 2)‖_p = (1 + 2^p)^(1/p)."""
    fr = F.solve_form("lp", antipodal, p=p, tol=1e-11)
    assert fr.result.value == pytest.approx(expected, abs=1e-7) and fr.shape.center == pytest.approx([0.0, 0.0], abs=1e-6) and not fr.original_problems


@pytest.mark.parametrize("beta", [0.5, 1.0, 5.0])
def test_lse_ball_of_two_antipodal_points_by_hand(antipodal, beta):
    """R = (1/β) log(e^β + e^(2β) + e^(−β) + e^(−2β)) im Mittelpunkt 0."""
    expected = math.log(math.exp(beta) + math.exp(2 * beta) + math.exp(-beta) + math.exp(-2 * beta)) / beta
    fr = F.solve_form("lse", antipodal, beta=beta, tol=1e-11)
    assert fr.result.value == pytest.approx(expected, abs=1e-7) and fr.shape.center == pytest.approx([0.0, 0.0], abs=1e-6) and not fr.original_problems


def test_p_equals_one_two_and_large_give_diamond_ball_and_cube():
    pts = make_cloud(12, 3.0, 30, 100000).points
    cube = cube_radius(pts)
    r1, r2, rbig = (F.solve_form("lp", pts, p=p).result.value for p in (1.0, 2.0, 128.0))
    assert r1 > r2 > rbig >= cube - 1e-7 and rbig == pytest.approx(cube, abs=1e-3)
    # p = 1: kleinste l1-Kugel gleich dem LP (Gegenprobe im Orakel-Test); hier: jeder Punkt im l1-Abstand höchstens R
    sh = F.solve_form("lp", pts, p=1.0).shape
    assert np.abs(pts - sh.center).sum(axis=1).max() <= r1 * (1 + 1e-6)


def test_the_power_cone_ball_equals_the_lorentz_ball_of_the_kernel_with_other_nu_and_t_times_error():
    pts = make_cloud(20, 4.0, 45, 35).points
    soc = F.build_soc_ball(pts)
    ref = K.solve_barrier(soc, tol=1e-12).value
    fr = F.solve_form("lp", pts, p=2.0)
    assert fr.result.value == pytest.approx(ref, abs=1e-7)
    assert fr.problem.nu == 140 and K.barrier_parameter(soc.cones) == 40          # ν gehört zur Formulierung


def test_lp_radius_falls_with_p_and_lse_radius_falls_with_beta():
    pts = make_cloud(14, 3.0, 25, 100003).points
    rp = [F.solve_form("lp", pts, p=p).result.value for p in (1.0, 1.5, 2.0, 4.0, 16.0)]
    rb = [F.solve_form("lse", pts, beta=b).result.value for b in (0.5, 1.0, 3.0, 10.0)]
    assert all(b < a for a, b in zip(rp, rp[1:])) and all(b < a for a, b in zip(rb, rb[1:]))
    cube = cube_radius(pts)
    assert all(r >= cube - 1e-7 for r in rp + rb)                                  # beide Familien liegen über dem Würfel


def test_the_lse_ball_is_within_log_2d_over_beta_of_the_cube():
    pts = make_cloud(14, 3.0, 25, 100003).points
    cube = cube_radius(pts)
    for beta in (0.5, 2.0, 10.0):
        r = F.solve_form("lse", pts, beta=beta).result.value
        assert cube - 1e-7 <= r <= cube + math.log(4.0) / beta + 1e-7


def test_gauge_by_hand():
    V = np.array([[3.0, 4.0]])
    assert F.gauge("lp", V, 2.0, 1.0)[0] == pytest.approx(5.0) and F.gauge("lp", V, 1.0, 1.0)[0] == pytest.approx(7.0)
    assert F.gauge("lse", V, 1.0, 2.0)[0] == pytest.approx(math.log(math.exp(6) + math.exp(8) + math.exp(-6) + math.exp(-8)) / 2.0)


def test_outline_points_satisfy_the_gauge_equation():
    for fam, p, beta in (("lp", 3.0, 1.0), ("lp", 1.0, 1.0), ("lse", 2.0, 2.0)):
        sh = F.Shape(fam, np.array([1.0, -2.0]), 2.5, p, beta)
        x, y = F.outline(sh, 120)
        assert np.allclose(F.gauge(fam, np.stack([x - 1.0, y + 2.0], axis=1), p, beta), 2.5, atol=1e-9)
        assert np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])                # geschlossener Umriss


def test_original_violations_detects_a_point_outside_and_a_wrong_value(antipodal):
    fr = F.solve_form("lp", antipodal, p=2.0)
    assert F.original_violations("lp", antipodal, fr.result.x, fr.result.value, p=2.0) == []
    x = fr.result.x.copy()
    x[2] *= 0.5                                                                    # Radius halbiert: Punkte liegen außerhalb
    assert any("außerhalb" in m for m in F.original_violations("lp", antipodal, x, fr.result.value, p=2.0))
    assert any("Zielwert" in m for m in F.original_violations("lp", antipodal, fr.result.x, fr.result.value * 2.0, p=2.0))


def test_programs_have_a_strictly_feasible_start_and_the_documented_shape():
    pts = make_cloud(6, 2.0, 10, 1).points
    for prob, nu, rows, nv in ((F.build_lp_ball(pts, 3.0), 6 * 7, 6 * 2 * 3 + 6, 3 + 12), (F.build_lse_ball(pts, 2.0), 6 * 13, 6 * 4 * 3 + 6, 3 + 24)):
        assert B.barrier(prob, B.slack(prob, prob.x0), hess=False)[1] is not None
        assert prob.nu == nu and prob.A.shape == (rows, nv) and prob.c.shape == (nv,)


def test_invalid_input_is_refused():
    pts = make_cloud(6, 2.0, 10, 1).points
    with pytest.raises(ValueError, match="mindestens 1"):
        F.build_form("lp", pts, p=0.5)
    with pytest.raises(ValueError, match="positiv"):
        F.build_form("lse", pts, beta=0.0)
    with pytest.raises(ValueError, match="unbekannte Familie"):
        F.build_form("kreis", pts)
    with pytest.raises(ValueError, match="nicht leere"):
        F.build_form("lp", np.zeros(5))
