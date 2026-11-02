"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import math

import numpy as np
import pytest

import kxp_barrier as B
import kxp_evaluation as ev
import kxp_forms as F
from kxp_scenario import make_cloud


def test_a_single_point_has_radius_zero_in_both_families():
    one = np.array([[3.0, -2.0]])
    pts = one - one.mean(axis=0)
    for fam, kw in (("lp", {"p": 3.0}), ("lse", {"beta": 5.0})):
        fr = F.solve_form(fam, pts, **kw)
        r = fr.result.value
        assert not fr.original_problems
        # lp: R -> 0; lse: R = (1/β) log(4) bei Mittelpunkt auf dem Punkt? nein: ein Punkt im Mittelpunkt hat Maß 0 für lp, bei lse log(2d)/β
        assert (r < 1e-6) if fam == "lp" else (r == pytest.approx(math.log(4.0) / 5.0, abs=1e-6))


def test_two_points_one_and_three_dimensions_run_through_both_families():
    p1 = np.array([[-2.0], [1.0], [3.0]])
    p3 = np.random.default_rng(5).normal(0, 1, (9, 3))
    for pts in (p1 - p1.mean(), p3 - p3.mean(axis=0)):
        for fam, kw in (("lp", {"p": 3.0}), ("lse", {"beta": 3.0})):
            fr = F.solve_form(fam, pts, **kw)
            assert not fr.original_problems and fr.result.value > 0, (fam, pts.shape)
    # d = 1: ℓp-Kugel gleich der halbe Spannweite für jedes p
    assert F.solve_form("lp", p1 - p1.mean(), p=3.0).result.value == pytest.approx(2.5, abs=1e-6)


def test_extreme_exponents_and_sharpness_still_solve():
    pts = make_cloud(12, 3.0, 20, 100002).points
    for p in (1.05, 128.0):
        fr = F.solve_form("lp", pts, p=p)
        assert not fr.original_problems and fr.result.value > 0, p
    for beta in (0.01, 1e5):
        fr = F.solve_form("lse", pts, beta=beta)
        assert not fr.original_problems, beta
    assert F.solve_form("lse", pts, beta=1e5).result.value == pytest.approx(ev.cube_radius(pts), abs=1e-4)


def test_duplicated_points_change_nothing():
    base = make_cloud(10, 3.0, 30, 100000).points
    twice = np.vstack([base, base])
    for fam, kw in (("lp", {"p": 3.0}), ("lse", {"beta": 4.0})):
        a, b = F.solve_form(fam, base, **kw).result.value, F.solve_form(fam, twice, **kw).result.value
        assert b == pytest.approx(a, abs=1e-6), fam


def test_scaling_the_data_scales_the_radius_with_lp_and_with_lse_when_beta_scales_inversely():
    pts = make_cloud(12, 3.0, 20, 100002).points
    base_lp = F.solve_form("lp", pts, p=3.0).result.value
    base_lse = F.solve_form("lse", pts, beta=4.0).result.value
    for f in (1e-6, 1e3, 1e6):
        assert F.solve_form("lp", f * pts, p=3.0).result.value == pytest.approx(f * base_lp, rel=1e-6)
        assert F.solve_form("lse", f * pts, beta=4.0 / f).result.value == pytest.approx(f * base_lse, rel=1e-6)


def test_the_scale_experiment_shows_the_absolute_tolerance_failing_and_the_relative_one_holding():
    rows = ev.scale_rows(ev.Settings(n=12))
    assert all(r["rel_centered"] for r in rows) and all(r["rel_newton"] < 80 for r in rows)
    big = rows[-1]
    assert (not big["abs_centered"]) and big["abs_newton"] > 100                      # absolute Genauigkeit bei 10⁶-fachen Daten: Rundungsgrenze (CPU-unabhängig weit darüber)


def test_every_experiment_runs_at_the_smallest_and_largest_settings():
    for n in (8, 40):
        for stretch, angle in ((1.2, 0), (6.0, 90)):
            for fam in ("lp", "lse"):
                s = ev.Settings(family=fam, n=n, stretch=stretch, angle=angle, seed=1)
                a = ev.analyse(s)
                assert ev.verdict_run(a)[0] in {"success", "warning"} and ev.deviation(a) < 1e-5
                assert ev.mu_rows(s, mus=(8.0,))[0]["newton"] > 0
    assert ev.two_barriers(ev.Settings(n=8))["support"] >= 2


def test_symmetry_probe_is_stable_in_the_number_of_points():
    for count in (200, 400, 800):
        dims = [r["dim"] for r in ev.symmetry_rows(count)]
        assert dims == [3, 4, 4, 2, 2], count


def test_support_count_and_cube_radius_by_hand():
    pts = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.5], [0.0, -0.5]])
    assert ev.cube_radius(pts) == 1.0                                          # größte Spannweite 2 / 2
    sh = F.Shape("lp", np.zeros(2), 1.0, 2.0)
    assert ev.support_count(pts, sh) == 2
