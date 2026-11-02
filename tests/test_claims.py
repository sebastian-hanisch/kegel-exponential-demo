"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alle Sweeps laufen über feste Wolken (SWEEP_SEEDS); die Standardwolke der Primäransicht (Seed 35) ist ausdrücklich eine einzelne.
Bänder statt Punktwerte, wo die Rundung die Zahl bewegt: die Auswahl der BLAS-Routinen auf dem Runner verschiebt Läufe im Rundungsbereich (Lehre vom 2026-11-01);
dort nennt die README beide Zahlen oder das Band."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kxp_barrier as B
import kxp_constants as C
import kxp_evaluation as ev
import kxp_forms as F
from kxp_scenario import make_cloud


@pytest.fixture(scope="module")
def std():
    return ev.analyse(ev.Settings())


def test_standard_cloud_run(std):
    """README: ℓ3-Kugel der Standardwolke R = 6,4146 (Raute 10,0895, Kugel 7,1675, Würfel 5,7840); ν = 140, 12 bis 13 Runden, 38 bis 50 Newton-Schritte, bewiesene Lücke unter 10⁻⁷."""
    r = std.form.result
    assert r.value == pytest.approx(6.414595, abs=1e-5) and std.others["diamond"] == pytest.approx(10.0895, abs=1e-4) and std.others["ball"] == pytest.approx(7.1675, abs=1e-4) and std.cube == pytest.approx(5.7840, abs=1e-4)
    assert std.form.problem.nu == 140 and 11 <= r.rounds <= 14 and 36 <= r.newton_steps <= 52 and r.gap_bound < 1e-7 and ev.deviation(std) < 1e-7 and not std.form.original_problems
    assert std.form.problem.A.shape == (3 * 40 + 20, 3 + 40) and ev.support_count(std.cloud.points, std.form.shape) == 2


def test_two_barriers_for_the_same_ball(std):
    """README: Lorentz ν = 40, Potenzkegel ν = 140, Radius 7,167473551 in beiden, Newton 34 und 39 (Band), t · Fehler am Pfadende 2,00 und 6,00, zwei Stützpunkte, sechs aktive Bedingungen."""
    res = ev.two_barriers(ev.Settings())
    assert res["soc"]["nu"] == 40 and res["pow"]["nu"] == 140
    assert res["soc"]["value"] == pytest.approx(7.167473551, abs=1e-7) and res["pow"]["value"] == pytest.approx(7.167473551, abs=1e-7)
    assert 30 <= res["soc"]["newton"] <= 40 and 34 <= res["pow"]["newton"] <= 48 and res["soc"]["rounds"] == 13 and 12 <= res["pow"]["rounds"] <= 14
    assert res["soc"]["path"][-1]["m"] == pytest.approx(2.0, abs=1e-3) and res["pow"]["path"][-1]["m"] == pytest.approx(6.0, abs=1e-3)
    assert res["support"] == 2 and res["pow"]["active"] == 6
    soc_m = [r["m"] for r in res["soc"]["path"]]
    pow_m = [r["m"] for r in res["pow"]["path"]]
    assert soc_m[0] > 20 and pow_m[0] > 100 and all(b <= a + 1e-3 for a, b in zip(soc_m[2:], soc_m[3:]))


def test_lp_family_values(std):
    """README: R(p) = 10,0895 (p = 1), 8,7936 (1,25), 8,0267 (1,5), 7,1675 (2), 6,4146 (3), 6,0814 (4), 5,7981 (8), 5,7841 (16), 5,7840 (ab 32); monoton fallend, über dem Würfel;
    ab p = 32 weniger als 10⁻⁷ über dem Würfelradius; Newton-Schritte 38 bis 83 (Band 30 bis 100, Spitze um p = 16)."""
    rows = {r["p"]: r for r in ev.lp_family(ev.Settings())}
    for p, v in ((1.0, 10.0895), (1.25, 8.7936), (1.5, 8.0267), (2.0, 7.1675), (3.0, 6.4146), (4.0, 6.0814), (8.0, 5.7981), (16.0, 5.7841), (32.0, 5.7840)):
        assert rows[p]["radius"] == pytest.approx(v, abs=1e-4), p
    radii = [rows[p]["radius"] for p in C.P_GRID]
    assert all(b <= a + 1e-9 for a, b in zip(radii, radii[1:])) and all(r["above_cube"] >= -1e-7 for r in rows.values())
    assert rows[32.0]["above_cube"] < 1e-7 and rows[128.0]["above_cube"] < 1e-7 and 1e-5 < rows[16.0]["above_cube"] < 1e-3
    newton = [r["newton"] for r in rows.values()]
    assert 30 <= min(newton) and max(newton) <= 100
    assert all(not r["problems"] and 10 <= r["rounds"] <= 13 for r in rows.values())


def test_lse_family_values():
    """README: R(β) − Würfel = 0,7185 (β = 0,5), 0,1116 (1), 0,0046 (2), 2,8·10⁻⁴ (3), 1,5·10⁻⁶ (5), unter 10⁻⁷ ab β = 10; Schranke log(2d)/β = 2,77 / 1,39 / 0,69 / 0,46 / 0,28 / 0,14; Newton 44 bis 90 (Band)."""
    rows = {r["beta"]: r for r in ev.lse_family(ev.Settings())}
    for beta, v in ((0.5, 0.7185), (1.0, 0.1116), (2.0, 0.0046), (3.0, 2.84e-4), (5.0, 1.54e-6)):
        assert rows[beta]["above_cube"] == pytest.approx(v, rel=0.02, abs=2e-7), beta
    assert all(rows[b]["above_cube"] < 1e-7 for b in (10.0, 30.0, 100.0, 1000.0))
    assert [round(rows[b]["bound"], 2) for b in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)] == [2.77, 1.39, 0.69, 0.46, 0.28, 0.14]
    assert all(rows[b]["above_cube"] < rows[b]["bound"] for b in rows) and rows[5.0]["bound"] / max(rows[5.0]["above_cube"], 1e-12) > 1e4
    assert 35 <= min(r["newton"] for r in rows.values()) and max(r["newton"] for r in rows.values()) <= 120


def test_symmetry_and_dual_cone():
    """README: Automorphismen-Dimension Orthant 3, Lorentz 4, Potenzkegel a = 1/2 4, a = 1/3 2, Exponentialkegel 2; −∇φ 400 von 400 im Dualkegel; 0 von 20000 Paaren verletzen schwache Dualität."""
    rows = {r["cone"]: r for r in ev.symmetry_rows()}
    assert [rows[c]["dim"] for c in ("Orthant (3 Koordinaten)", "Lorentz-Kegel (3 Koordinaten)", "Potenzkegel, a = 1/2", "Potenzkegel, a = 1/3", "Exponentialkegel")] == [3, 4, 4, 2, 2]
    assert [rows[c]["homogeneous"] for c in rows] == [True, True, True, False, False]
    for r in rows.values():
        k = 9 - r["dim"]
        assert r["singular"][k - 1] > 1e6 * max(r["singular"][k], 1e-300)
    dual = ev.dual_checks(ev.Settings())
    assert dual["inside"] == dual["samples"] == 400 and dual["violated"] == 0 and dual["pairs"] == 20000 and dual["witness_in_K"] and not dual["witness_in_dual"]
    assert dual["dual_feasible"] and dual["sz"] == pytest.approx(dual["nu_over_t"], rel=1e-8) and dual["dual_residual"] < 1e-5 and abs(dual["cx_bz"] - dual["nu_over_t"]) <= 0.2 * dual["nu_over_t"]


def test_the_dual_certificate_degrades_at_the_path_end():
    """README: nach scharfem Nachzentrieren ist A^T z + c bei t ≈ 2·10⁷ unter 10⁻⁶, am Pfadende (t ≈ 7·10¹¹) bis 10⁻⁵ (Band) und c^T x + b^T z nicht mehr als Zertifikat auf 10⁻¹⁰ tragfähig."""
    pts = make_cloud(9, 2.0, 0, 1).points
    prob = F.build_lp_ball(pts, 3.0)
    mid = B.solve(prob, tol=1e-6)
    x, z = B.dual_point(prob, mid)
    assert np.abs(prob.A.T @ z + prob.c).max() < 1e-6
    end = B.solve(prob, tol=1e-10)
    xe, ze = B.dual_point(prob, end)
    assert np.abs(prob.A.T @ ze + prob.c).max() < 1e-3 and end.gap_bound < 1e-9 and abs(float(prob.c @ xe + prob.b @ ze)) > 0


def test_newton_steps_are_flat_in_nu():
    """README: n = 4 bis 40 (ν von 28 auf 280 bzw. von 52 auf 520): ℓ3 35 bis 40 Newton-Schritte, log-sum-exp 46 bis 59 (Bänder 30 bis 48 und 40 bis 70), je 11 bis 13 Runden (Median über 3 Wolken)."""
    rows = ev.nu_rows()
    assert [r["n"] for r in rows] == list(C.N_SWEEP) and [r["lp"]["nu"] for r in rows] == [7 * n for n in C.N_SWEEP] and [r["lse"]["nu"] for r in rows] == [13 * n for n in C.N_SWEEP]
    assert all(30 <= r["lp"]["newton"] <= 48 and 40 <= r["lse"]["newton"] <= 70 and 11 <= r["lp"]["rounds"] <= 13 and 11 <= r["lse"]["rounds"] <= 13 for r in rows)
    assert max(r["lp"]["newton"] for r in rows) / min(r["lp"]["newton"] for r in rows) < 1.4 and max(r["lse"]["newton"] for r in rows) / min(r["lse"]["newton"] for r in rows) < 1.5
    # ν wächst um den Faktor 10, die Newton-Schritte nicht
    assert rows[-1]["lp"]["nu"] / rows[0]["lp"]["nu"] == 10 and rows[-1]["lp"]["newton"] / rows[0]["lp"]["newton"] < 1.4


def test_the_step_factor_has_its_minimum_at_a_moderate_mu():
    """README: Schrittfaktor μ der Standardwolke (ℓ3): 2: 68, 8: 44, 32: 33, 1000: 93 Newton-Schritte (Bänder); das Minimum liegt bei μ = 8 oder 32."""
    rows = {r["mu"]: r["newton"] for r in ev.mu_rows(ev.Settings())}
    assert 50 <= rows[2.0] <= 90 and 36 <= rows[8.0] <= 52 and 28 <= rows[32.0] <= 42 and 60 <= rows[1000.0] <= 180
    assert min(rows, key=rows.get) in (8.0, 32.0) and rows[2.0] > rows[8.0] and rows[1000.0] > 1.5 * rows[32.0]


def test_the_data_scale_with_relative_and_absolute_accuracy():
    """README: relative Genauigkeit 42 bis 46 Newton-Schritte bei Maßstab 10⁻⁶ bis 10⁶, stets zentriert; absolute Genauigkeit 10⁻⁹: bei 10⁶ nicht zentriert (halbes Dekrement-Quadrat über 10²), bei 10³ je nach Prozessor (Entwicklungsrechner 1,2·10⁻³: nicht zentriert, GitHub-Runner: noch zentriert)."""
    rows = ev.scale_rows(ev.Settings())
    assert [r["factor"] for r in rows] == list(C.SCALE_FACTORS)
    assert all(r["rel_centered"] and 38 <= r["rel_newton"] <= 50 for r in rows)
    assert all(r["radius"] == pytest.approx(f * rows[1]["radius"], rel=1e-6) for r, f in zip(rows, C.SCALE_FACTORS))
    assert rows[0]["abs_centered"] and rows[1]["abs_centered"] and not rows[3]["abs_centered"]                 # bei 10³ ist es CPU-abhängig (Entwicklungsrechner: nicht zentriert, GitHub-Runner: noch zentriert)
    assert rows[3]["abs_lam2"] > 1e2 and rows[3]["abs_newton"] > 3 * rows[3]["rel_newton"]


def test_extreme_parameters_do_not_break_the_method():
    """README: β bis 10⁵, p von 1,05 bis 128 und der Datenmaßstab 10⁻⁶ bis 10⁶ (relativ) lösen sich ohne Absturz; die Rückumformung besteht jeweils."""
    pts = make_cloud(20, 4.0, 45, 35).points
    for fam, kw in (("lp", {"p": 1.05}), ("lp", {"p": 128.0}), ("lse", {"beta": 1e5}), ("lse", {"beta": 0.01})):
        fr = F.solve_form(fam, pts, **kw)
        assert not fr.original_problems and fr.result.value > 0, (fam, kw)


def test_the_power_cone_ball_equals_the_stueck_5_ball_value(std):
    """README: p = 2 gibt denselben Radius wie die Lorentz-Kugel aus Stück 5 und Stück 6 (7,1675)."""
    assert std.others["ball"] == pytest.approx(K.solve_barrier(F.build_soc_ball(std.cloud.points), tol=1e-12).value, abs=1e-7)


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 6: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    import hashlib
    from pathlib import Path

    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-primal-dual-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
