"""Die Barrierenmethode (kxp_barrier): Mini-Programme von Hand, Pfad, Dual aus dem Pfad, Genauigkeit, Randfälle."""

import math

import numpy as np
import pytest

import kxp_barrier as B
import kxp_cones as C
import kxp_forms as F
from kxp_scenario import make_cloud


def test_the_two_hand_programs_reach_their_optimum_within_the_proven_gap(minis):
    for name, (prob, opt) in minis.items():
        r = B.solve(prob, tol=1e-10)
        assert r.centered and r.value - opt >= -1e-9 and r.value - opt <= r.gap_bound * (1 + 1e-6) + 1e-9, name      # nu/t deckt den Fehler
        assert r.value == pytest.approx(opt, abs=1e-6), name


def test_structure_nu_and_rows(minis):
    exp, _ = minis["exp"]
    assert exp.nu == 3 + 1 and exp.rows == 4 and minis["pow"][0].nu == 3 and minis["pow"][0].rows == 3
    assert B.nu_of((("e", None, 5), ("p", 0.3, 2), ("l", None, 7))) == 15 + 6 + 7 and B.rows_of((("e", None, 5), ("p", 0.3, 2), ("l", None, 7))) == 15 + 6 + 7


def test_the_path_of_the_exponential_program_in_closed_form(minis):
    """min z, e^x <= z, x >= 1: auf dem Pfad stationär in x und z. Mit s = (x, 1, z): ψ = log z − x. t = −∂/∂z-Bedingung: t − (1/ψ)(1/z) ... nachgerechnet numerisch gegen die Stationarität."""
    prob, _ = minis["exp"]
    for t in (1.0, 10.0, 1e3):
        x, steps, lam2, _ = B.center(prob, prob.x0.copy(), t, tol=1e-14, max_steps=200)
        s = B.slack(prob, x)
        _, g, _ = B.barrier(prob, s)
        grad = t * prob.c - prob.A.T @ g
        assert np.abs(grad).max() < 1e-6 * (1 + t)                                  # Gradient von f_t verschwindet
        assert lam2 < 1e-10


def test_newton_matrix_equals_the_finite_difference_hessian_of_the_barrier_part():
    pts = make_cloud(5, 2.0, 20, 3).points
    for prob in (F.build_lp_ball(pts, 3.0), F.build_lse_ball(pts, 4.0)):
        x = prob.x0 + 0.01 * np.random.default_rng(0).normal(size=prob.x0.shape)
        _, _, hs = B.barrier(prob, B.slack(prob, x))
        M = B.newton_matrix(prob, hs)

        def grad_x(xx):
            _, g, _ = B.barrier(prob, B.slack(prob, xx))
            return -prob.A.T @ g
        h = 1e-6
        fd = np.array([(grad_x(x + h * e) - grad_x(x - h * e)) / (2 * h) for e in np.eye(len(x))])
        assert np.allclose(M, M.T, rtol=1e-9) and np.allclose(M, fd, rtol=1e-4, atol=1e-5 * (1 + np.abs(M).max()))
        assert np.linalg.eigvalsh(M).min() > 0


def test_centering_decreases_the_function_and_ends_with_a_small_decrement():
    prob = F.build_lp_ball(make_cloud(8, 2.0, 10, 5).points, 3.0)
    t = B.initial_t(prob)
    val0 = t * float(prob.c @ prob.x0) + B.barrier(prob, B.slack(prob, prob.x0))[0]
    x, steps, lam2, lengths = B.center(prob, prob.x0.copy(), t)
    val1 = t * float(prob.c @ x) + B.barrier(prob, B.slack(prob, x))[0]
    assert val1 < val0 and lam2 < 1e-6 and steps == len(lengths) and all(0 < a <= 1.0 for a in lengths)
    assert lengths[-1] == 1.0                                                      # am Ende volle Schritte (quadratische Phase)


def test_the_dual_point_from_the_path_is_dual_feasible_and_complementary():
    pts = make_cloud(10, 3.0, 30, 7).points
    for prob in (F.build_lp_ball(pts, 3.0), F.build_lse_ball(pts, 5.0)):
        res = B.solve(prob, tol=1e-9)
        z = B.dual_from_path(prob, res.x, res.t)
        s = B.slack(prob, res.x)
        assert B.dual_feasible_blocks(prob, z)
        assert float(s @ z) == pytest.approx(prob.nu / res.t, rel=1e-8)             # ⟨∇φ, s⟩ = −ν
        assert np.abs(prob.A.T @ z + prob.c).max() < 1e-4 * (1.0 + np.abs(z).max())      # A^T z = −c bis zur Zentrierungsgenauigkeit


def test_the_dual_check_rejects_a_point_outside_the_dual_cone(minis):
    prob, _ = minis["exp"]
    assert not B.dual_feasible_blocks(prob, np.array([1.0, 1.0, 1.0, 1.0]))      # u > 0 liegt nicht im Dualkegel des Exponentialkegels
    assert B.dual_feasible_blocks(prob, np.array([-1.0, 1.0, 5.0, 1.0]))


def test_the_power_cone_dual_is_the_scaled_cone():
    prob = B.Program(np.zeros(1), np.zeros((3, 1)), np.array([1.0, 1.0, 0.0]), (("p", 0.25, 1),), np.zeros(1))
    a = 0.25
    inside = np.array([a * 2.0, (1 - a) * 2.0, 1.9])               # (u/a)^a (v/(1−a))^(1−a) = 2 > |w|
    outside = np.array([a * 2.0, (1 - a) * 2.0, 2.1])
    assert B.dual_feasible_blocks(prob, inside) and not B.dual_feasible_blocks(prob, outside)


def test_accuracy_is_relative_to_the_start_value_so_the_scale_does_not_matter():
    pts = make_cloud(10, 3.0, 30, 11).points
    base = B.solve(F.build_lp_ball(pts, 3.0), tol=1e-9)
    for f in (1e-6, 1e3, 1e6):
        r = B.solve(F.build_lp_ball(f * pts, 3.0), tol=1e-9)
        assert r.value == pytest.approx(f * base.value, rel=1e-6) and r.centered and abs(r.newton_steps - base.newton_steps) <= 8


def test_a_start_outside_the_cone_is_refused():
    prob = B.Program(np.array([0.0, 1.0]), np.array([[-1.0, 0.0], [0.0, 0.0], [0.0, -1.0]]), np.array([0.0, 1.0, 0.0]), (("e", None, 1),), np.array([2.0, 1.0]))     # z = 1 < e²
    with pytest.raises(ValueError, match="streng im Inneren"):
        B.solve(prob)


def test_the_rounding_stop_reports_an_honest_centering_flag():
    """Bei sehr hohem t kann die Zentrierung ihre Toleranz nicht mehr erreichen: das Ergebnis meldet es (centered), statt 100 Schritte zu verschwenden."""
    prob = F.build_lse_ball(make_cloud(20, 4.0, 45, 35).points, 5.0)
    res = B.solve(prob, tol=1e-12)
    assert res.newton_steps < 400 and isinstance(res.centered, bool)
    if not res.centered:
        assert res.lam2 > 1e-6
    assert res.value == pytest.approx(B.solve(prob, tol=1e-9).value, abs=1e-6)


def test_trace_path_is_monotone_and_the_bound_covers_the_error():
    prob = F.build_lp_ball(make_cloud(12, 3.0, 30, 100001).points, 3.0)
    p_star = B.solve(prob, tol=1e-11).value
    path = B.trace_path(prob, (1.0, 10.0, 100.0, 1e3, 1e4, 1e5))
    assert all(b["value"] <= a["value"] + 1e-12 for a, b in zip(path, path[1:])) and all(p["centered"] for p in path)
    assert all(0 <= p["value"] - p_star <= p["bound"] * (1 + 1e-6) + 1e-9 for p in path)


def test_active_count_of_the_ball_counts_the_scalar_constraints_of_the_support_points(antipodal):
    prob = F.build_lp_ball(antipodal, 2.0)
    res = B.solve(prob, tol=1e-12)
    t = 1e7
    path = B.trace_path(prob, (1.0, 1e2, 1e4, 1e6, t))
    # zwei Stützpunkte; je Stützpunkt zwei Potenzkegel und die Summenzeile aktiv: 3 · 2
    assert B.active_count(prob, path[-1]["x"], t) == 6
    assert res.value == pytest.approx(math.sqrt(5.0), abs=1e-8)
