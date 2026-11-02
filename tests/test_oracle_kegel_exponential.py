"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): die Barrierenmethode über Exponential- und Potenzkegel gegen einen anderen Rechenweg.

- cvxpy mit Clarabel (primal-dualer Löser mit eigener Behandlung der nichtsymmetrischen Kegel): Radius der ℓp-Kugeln (`pnorm(..., approx=False)` = Potenzkegel) und der log-sum-exp-Kugeln
- die Gegenstücke aus den früheren Stücken: p = 2 gleich der Lorentz-Kugel des Kerns; p = 1 gleich dem LP der Raute (scipy); großes p und großes β gleich dem Würfel (geschlossen)
- Eigenschaften, die ein Fehler in beiden Rechenwegen nicht überlebt (Monotonie, Dualität, Schranke ν/t)

Fehlt cvxpy im Test-venv, überspringen sich die cvxpy-Tests (pytest.importorskip), sie raten nie."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kxp_barrier as B
import kxp_evaluation as ev
import kxp_forms as F


def _cloud(seed, n, d=2):
    pts = np.random.default_rng(seed).normal(0.0, 1.0, (n, d))
    return pts - pts.mean(axis=0)


def _cvxpy():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp, dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_lp_balls_equal_the_cvxpy_value_with_power_cones():
    cp, kw = _cvxpy()
    for seed in range(5):
        d = 2 + seed % 2
        pts = _cloud(seed, 8 + seed % 4, d)
        for p in (1.5, 2.0, 3.0, 4.0, 8.0):
            c, R = cp.Variable(d), cp.Variable()
            cp.Problem(cp.Minimize(R), [cp.pnorm(pt - c, p, approx=False) <= R for pt in pts]).solve(**kw)
            fr = F.solve_form("lp", pts, p=p, tol=1e-10)
            assert fr.result.value == pytest.approx(float(R.value), rel=1e-6), (seed, p)
            assert fr.result.value - float(R.value) <= fr.result.gap_bound + 1e-7, (seed, p)            # die bewiesene Lücke deckt den Abstand zu cvxpy


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_lse_balls_equal_the_cvxpy_value_with_exponential_cones():
    cp, kw = _cvxpy()
    for seed in range(5):
        d = 2 + seed % 2
        pts = _cloud(seed, 8 + seed % 4, d)
        for beta in (0.5, 1.0, 3.0, 10.0):
            c, R = cp.Variable(d), cp.Variable()
            cons = [cp.log_sum_exp(cp.hstack([beta * (pt - c), -beta * (pt - c)])) <= beta * R for pt in pts]
            cp.Problem(cp.Minimize(R), cons).solve(**kw)
            fr = F.solve_form("lse", pts, beta=beta, tol=1e-10)
            assert fr.result.value == pytest.approx(float(R.value), rel=1e-6), (seed, beta)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_cube_radius_is_the_cvxpy_value_and_the_limit_of_both_families():
    cp, kw = _cvxpy()
    pts = _cloud(3, 14)
    c, R = cp.Variable(2), cp.Variable()
    cp.Problem(cp.Minimize(R), [cp.norm(pt - c, "inf") <= R for pt in pts]).solve(**kw)
    cube = float(R.value)
    assert ev.cube_radius(pts) == pytest.approx(cube, rel=1e-7)                       # geschlossene Form gegen cvxpy
    assert F.solve_form("lp", pts, p=128.0).result.value == pytest.approx(cube, abs=5e-3)
    assert F.solve_form("lse", pts, beta=1000.0).result.value == pytest.approx(cube, abs=1e-5)


def test_p_equal_two_is_the_lorentz_ball_of_the_kernel():
    for seed in range(4):
        pts = _cloud(seed, 10 + seed)
        ref = K.solve_barrier(F.build_soc_ball(pts), tol=1e-12).value
        assert F.solve_form("lp", pts, p=2.0, tol=1e-10).result.value == pytest.approx(ref, abs=1e-7), seed


def test_p_equal_one_is_the_linear_program_of_the_diamond():
    linprog = pytest.importorskip("scipy.optimize").linprog
    for seed in range(4):
        pts = _cloud(seed, 9 + seed)
        n, d = pts.shape
        # Variablen (c, R, t_ij): t_ij >= |x_ij − c_j|, sum_j t_ij <= R
        nv = d + 1 + n * d
        A, b = [], []
        for i in range(n):
            for j in range(d):
                k = d + 1 + i * d + j
                for sg in (1.0, -1.0):
                    row = np.zeros(nv)
                    row[j] = -sg
                    row[k] = -1.0
                    A.append(row)
                    b.append(-sg * pts[i, j])
            row = np.zeros(nv)
            row[d] = -1.0
            row[d + 1 + i * d: d + 1 + (i + 1) * d] = 1.0
            A.append(row)
            b.append(0.0)
        cost = np.zeros(nv)
        cost[d] = 1.0
        res = linprog(cost, A_ub=np.array(A), b_ub=np.array(b), bounds=[(None, None)] * d + [(0, None)] + [(0, None)] * (n * d), method="highs")
        assert res.status == 0
        assert F.solve_form("lp", pts, p=1.0, tol=1e-10).result.value == pytest.approx(float(res.fun), abs=1e-6), seed


def test_the_bound_covers_the_error_at_every_path_point_of_both_families():
    pts = _cloud(2, 12)
    for prob in (F.build_lp_ball(pts, 3.0), F.build_lse_ball(pts, 4.0)):
        p_star = B.solve(prob, tol=1e-11).value
        for pt in B.trace_path(prob, (0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5)):
            assert -1e-9 <= pt["value"] - p_star <= pt["bound"] * (1 + 1e-6) + 1e-9


def test_scaling_the_points_scales_the_radius_and_keeps_the_newton_steps_close():
    pts = _cloud(2, 10)
    for fam, kw in (("lp", {"p": 3.0}), ("lse", {})):
        a = F.solve_form(fam, pts, **kw, **({"beta": 5.0} if fam == "lse" else {}))
        b = F.solve_form(fam, 3.0 * pts, **kw, **({"beta": 5.0 / 3.0} if fam == "lse" else {}))
        assert b.result.value == pytest.approx(3.0 * a.result.value, rel=1e-6) and abs(a.result.newton_steps - b.result.newton_steps) <= 12, fam


def test_the_dual_point_from_the_path_gives_a_dual_bound_at_moderate_t_and_degrades_at_the_very_end():
    """Nach scharfem Nachzentrieren liegen Primal c^T x und Dual −b^T z um etwa s^T z = ν/t auseinander, solange t mäßig ist; am äußersten Pfadende (t über 10^9) verliert z durch die
    Konditionierung an Genauigkeit (dasselbe hat Stück 4 am Pfad der Orthant-, Lorentz- und PSD-Programme gemessen). Ohne Nachzentrieren (Toleranz 1e-6) ist A^T z + c nur etwa 1e-3 genau."""
    pts = _cloud(1, 9)
    prob = F.build_lp_ball(pts, 3.0)
    mid = B.solve(prob, tol=1e-5)
    plain = B.dual_from_path(prob, mid.x, mid.t)
    x, z = B.dual_point(prob, mid)
    s = B.slack(prob, x)
    nu_t = prob.nu / mid.t
    assert B.dual_feasible_blocks(prob, z) and float(s @ z) == pytest.approx(nu_t, rel=1e-8)
    assert abs(float(prob.c @ x + prob.b @ z) - nu_t) <= 0.1 * nu_t and np.abs(prob.A.T @ z + prob.c).max() < 1e-6
    assert np.abs(prob.A.T @ plain + prob.c).max() >= np.abs(prob.A.T @ z + prob.c).max()      # das Nachzentrieren verbessert, nie umgekehrt
    end = B.solve(prob, tol=1e-10)
    x_end, z_end = B.dual_point(prob, end)
    assert B.dual_feasible_blocks(prob, z_end) and abs(float(prob.c @ x_end + prob.b @ z_end)) < 1e-3          # trägt noch eine grobe Dualschranke, aber kein Zertifikat auf 1e-10
