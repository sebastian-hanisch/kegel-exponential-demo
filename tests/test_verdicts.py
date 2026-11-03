"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

from types import SimpleNamespace

import numpy as np

import kxp_evaluation as ev
import kxp_forms as F


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich"
    assert ev.sci(9.99e-3, 1) == "1,0·10⁻²" and ev.num(0.5) == "0,5000" and ev.num(1e-6) == "1,00·10⁻⁶" and ev.num(7.167473, 6) == "7,167473"
    assert ev.p_text(3.0) == "3" and ev.p_text(1.5) == "1,5" and ev.p_text(1.25) == "1,25" and ev.beta_text(10.0) == "10" and ev.beta_text(0.5) == "0,5"
    assert ev.above_text(1e-9) == "weniger als 1·10⁻⁷" and ev.above_text(2e-4) == "2,0·10⁻⁴" and ev.over_text(2e-4) == "um 2,0·10⁻⁴" and ev.over_text(1e-9) == "weniger als 1·10⁻⁷"
    assert ev._range(5, 5, " Runden") == "5 Runden" and ev._range(5, 8) == "5 bis 8"


def _analysis(family="lp", p=3.0, beta=10.0, centered=True, problems=None, nu=140, value=6.414595, newton=44, rounds=12):
    result = SimpleNamespace(value=value, gap_bound=3.5e-9, newton_steps=newton, rounds=rounds, centered=centered, lam2=4.4e-6)
    form = SimpleNamespace(problem=SimpleNamespace(nu=nu), result=result, original_problems=problems or [])
    return SimpleNamespace(settings=SimpleNamespace(family=family, p=p, beta=beta), form=form, reference=value + 2.0e-11, others={"diamond": 10.0895, "ball": 7.1675}, cube=5.784)


def test_run_verdict_names_radius_comparison_nu_rounds_newton_and_gap():
    kind, text = ev.verdict_run(_analysis())
    assert kind == "success"
    for part in ("ℓp-Kugel (Potenzkegel)** (p = 3, a = 1/3)", "Radius **6,414595**", "Raute 10,0895, Kugel 7,1675, Würfel 5,7840", "ν = **140** (Potenzkegel je drei, Zeilen je eins)",
                 "**12 Runden mit zusammen 44 Newton-Schritten**", "Lücke 3,5·10⁻⁹", "Abweichung vom engeren Referenzwert 2,0·10⁻¹¹", "Die Rückumformung besteht."):
        assert part in text, part
    kind, text = ev.verdict_run(_analysis("lse", beta=0.5, nu=260))
    assert "log-sum-exp-Kugel (Exponentialkegel)** (β = 0,5)" in text and "Exponentialkegel je drei" in text


def test_run_verdict_warns_when_the_last_centering_ended_at_the_rounding_limit_and_errors_when_the_form_fails():
    kind, text = ev.verdict_run(_analysis(centered=False))
    assert kind == "warning" and "Rundungsgrenze (halbes Dekrement-Quadrat 4,4·10⁻⁶)" in text and "gilt streng nur für zentrierte Punkte" in text
    kind, text = ev.verdict_run(_analysis(problems=["ein Punkt liegt außerhalb"]))
    assert kind == "error" and "besteht nicht: ein Punkt liegt außerhalb" in text


def _barriers():
    soc = {"nu": 40, "value": 7.167473551, "newton": 34, "rounds": 13, "path": [{"t": 1e7, "error": 2.0e-7, "m": 2.0}]}
    pw = {"nu": 140, "value": 7.167473552, "newton": 39, "rounds": 13, "path": [{"t": 1e7, "error": 6.0e-7, "m": 6.0}], "active": 6}
    return {"soc": soc, "pow": pw, "support": 2, "p_star": 7.167473551}


def test_barriers_verdict_says_nu_and_m_belong_to_the_formulation():
    kind, text = ev.verdict_barriers(_barriers())
    assert kind == "info"
    for part in ("7,167474 und 7,167474", "als Lorentz-Kegel ist ν = 40, als Potenzkegel a = 1/2 ist ν = 140", "Newton-Schritte 34 und 39, beide in 13 und 13 Runden", "(t = 1·10⁷)",
                 "`t · Fehler` = 2,00 für den Lorentz-Kegel und **6,00** für den Potenzkegel", "2 Stützpunkte, und 6 Bedingungen sind aktiv", "gehören zur Formulierung", "gedrehter Lorentz-Kegel"):
        assert part in text, part


def _lp_rows(bad=(), loose=()):
    return [{"p": p, "radius": r, "above_cube": r - 5.784, "nu": 140, "newton": nw, "rounds": 12, "centered": p not in loose, "problems": ["x"] if p in bad else []}
            for p, r, nw in ((1.0, 10.0895, 34), (2.0, 7.1675, 39), (128.0, 5.784, 40))]


def test_lp_verdict_names_the_trend_the_cube_and_flags_problems_and_uncentered_ends():
    kind, text = ev.verdict_lp(_lp_rows(), 5.784)
    assert kind == "info"
    for part in ("fällt monoton von 10,0895 (p = 1, Raute) über 7,1675 (p = 2, Kugel) auf 5,784000 (p = 128)", "Würfel liegt bei 5,784000 (Abstand weniger als 1·10⁻⁷)", "Newton-Schritte 34 bis 40"):
        assert part in text, part
    assert "Nicht zentriert am Ende (Rundungsgrenze): p = 2." in ev.verdict_lp(_lp_rows(loose=(2.0,)), 5.784)[1]
    kind, text = ev.verdict_lp(_lp_rows(bad=(2.0,)), 5.784)
    assert kind == "warning" and "besteht nicht für p = 2" in text
    rows = _lp_rows()
    rows[1]["radius"] = 12.0
    assert "nicht monoton" in ev.verdict_lp(rows, 5.784)[1]


def _lse_rows(loose=()):
    return [{"beta": b, "radius": 5.784 + a, "above_cube": a, "bound": np.log(4.0) / b, "nu": 260, "newton": nw, "rounds": 12, "centered": b not in loose, "problems": []}
            for b, a, nw in ((0.5, 0.7185, 46), (10.0, 1e-9, 51), (1000.0, 1e-9, 50))]


def test_lse_verdict_compares_the_distance_with_the_bound():
    kind, text = ev.verdict_lse(_lse_rows())
    assert kind == "info"
    for part in ("für β = 0,5 um 0,7185 über dem Würfelradius", "für β = 10 weniger als 1·10⁻⁷ und für β = 1000 weniger als 1·10⁻⁷", "sagt 2,77, 0,139 und 0,0014", "bei β = 0,5 um den Faktor 4 zu grob und bei größerem β um Größenordnungen", "Newton-Schritte 46 bis 51"):
        assert part in text, part
    assert "β = 10." in ev.verdict_lse(_lse_rows(loose=(10.0,)))[1] and "Nicht zentriert am Ende" in ev.verdict_lse(_lse_rows(loose=(10.0,)))[1]


def test_symmetry_verdict_names_the_dimensions_and_the_dual_facts():
    rows = [{"cone": c, "dim": d, "homogeneous": d >= 3, "singular": []} for c, d in (("Orthant (3 Koordinaten)", 3), ("Lorentz-Kegel (3 Koordinaten)", 4), ("Potenzkegel, a = 1/2", 4), ("Potenzkegel, a = 1/3", 2), ("Exponentialkegel", 2))]
    dual = {"samples": 400, "inside": 400, "pairs": 20000, "violated": 0, "witness_in_K": True, "witness_in_dual": False, "sz": 1.3e-5, "nu_over_t": 1.3e-5}
    kind, text = ev.verdict_symmetry(rows, dual)
    assert kind == "info"
    for part in ("Orthant 3, Lorentz-Kegel 4, Potenzkegel a = 1/2 **4**", "Potenzkegel a = 1/3 **2**, Exponentialkegel **2**", "mindestens 3", "keine NT-Skalierung", "ein Punkt liegt in K, nicht in K*",
                 "für 400 von 400 Punkten", "0 von 20000 Paaren verletzen", "`sᵀz` = 1,3·10⁻⁵ bei ν/t = 1,3·10⁻⁵"):
        assert part in text, part
    assert "der Zeuge trennt K und K* nicht" in ev.verdict_symmetry(rows, {**dual, "witness_in_dual": True})[1]


def test_nu_verdict_names_the_ranges_and_the_best_mu():
    rows = [{"n": 4, "lp": {"nu": 28, "newton": 35.0, "rounds": 12.0}, "lse": {"nu": 52, "newton": 46.0, "rounds": 12.0}},
            {"n": 40, "lp": {"nu": 280, "newton": 40.0, "rounds": 12.0}, "lse": {"nu": 520, "newton": 59.0, "rounds": 12.0}}]
    mu = [{"mu": 2.0, "newton": 68}, {"mu": 8.0, "newton": 44}, {"mu": 32.0, "newton": 33}, {"mu": 1000.0, "newton": 130}]
    kind, text = ev.verdict_nu(rows, mu)
    assert kind == "info"
    for part in ("n = 4 bis n = 40", "ν der ℓp-Kugel von 28 auf 280, der log-sum-exp-Kugel von 52 auf 520", "35 bis 40 (ℓp) und 46 bis 59 (log-sum-exp), je 12 Runden", "am wenigsten Newton-Schritte bei μ = 32 (33)", "bei μ = 2 68", "bei μ = 1000 130"):
        assert part in text, part


def test_scale_verdict_contrasts_relative_and_absolute_accuracy():
    rows = [{"factor": f, "rel_newton": nw, "rel_centered": True, "abs_lam2": lam, "abs_centered": ok}
            for f, nw, lam, ok in ((1e-6, 44, 3e-10, True), (1.0, 45, 1e-7, True), (1e6, 44, 5.7e4, False))]
    kind, text = ev.verdict_scale(rows)
    assert kind == "info"
    for part in ("von 1·10⁻⁶ bis 1·10⁶", "mit **relativer** Genauigkeit 44 bis 45 Newton-Schritte und stets zentriert", "bei dem Faktor 1·10⁶ nicht zentriert (halbes Dekrement-Quadrat 5,7·10⁴)", "Rundungsgrenze der Rechnung"):
        assert part in text, part
    rows[-1]["abs_centered"] = True
    assert "noch zentriert" in ev.verdict_scale(rows)[1]


def test_family_names_and_cone_names_are_complete():
    assert set(F.FAMILY_NAMES) == set(F.CONE_NAMES) == set(F.FAMILIES)
