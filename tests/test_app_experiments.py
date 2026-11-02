"""Pfade der App, die der Rauchtest nicht erreicht: jede Familie, die fünf Experimente im Seiteninhalt, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kxp_constants as C
import kxp_forms as F

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        widget = at.sidebar.selectbox if key == "family_select" else at.sidebar.number_input if key == "seed_input" else at.sidebar.select_slider if key in ("p_select", "beta_select") else at.sidebar.slider
        widget(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def _count(text):
    return int(re.match(r"(\d+)", text).group(1))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Barriere-Parameter ν"] == "140" and 30 <= _count(values["Newton-Schritte"]) <= 60 and values["Stützpunkte"] == "2" and values["Radius R"] == "6,4146"
    assert values["Bewiesene Lücke ν/t"].endswith(("⁻⁷", "⁻⁸", "⁻⁹", "⁻¹⁰"))
    text = _texts(at)
    for part in ("Die **ℓp-Kugel (Potenzkegel)** (p = 3, a = 1/3)", "Dieselbe Kugel, derselbe Wert", "Der Radius fällt", "Die log-sum-exp-Kugel liegt", "Dimension der Automorphismengruppe", "Von n = 4 bis n = 40 Punkten",
                 "Datenmaßstab von 1·10⁻⁶ bis 1·10⁶"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) >= 7 and not at.error


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("family", F.FAMILIES)
def test_every_family_renders_without_error(family):
    at = _app(family_select=family)
    assert not at.error
    assert any(f"Die **{F.FAMILY_NAMES[family]}**" in s.value for s in list(at.success) + list(at.warning))


@pytest.mark.parametrize("p", C.P_CHOICES)
def test_every_exponent_renders(p):
    at = _app(p_select=p)
    assert not at.error and any("Rückumformung besteht" in s.value for s in list(at.success) + list(at.warning))        # bei Rundungsgrenze am Pfadende ist das Verdikt eine Warnung


@pytest.mark.parametrize("beta", C.BETA_CHOICES)
def test_every_sharpness_renders(beta):
    at = _app(family_select="lse", beta_select=beta)
    assert not at.error and not at.exception


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_controls_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"n_slider": pick(C.N_MIN, C.N_MAX), "stretch_slider": pick(C.STRETCH_MIN, C.STRETCH_MAX), "angle_slider": pick(C.ANGLE_MIN, C.ANGLE_MAX),
             "p_select": pick(C.P_CHOICES[0], C.P_CHOICES[-1]), "beta_select": pick(C.BETA_CHOICES[0], C.BETA_CHOICES[-1])}
    at = _app(**state)
    assert not at.exception and not at.error


def test_the_slider_corners_work_in_both_families():
    """Regression aus Stück 4 (Eckfall-Test fand dort einen Absturz): Streckung 6, 45 Grad, 20 Punkte in jeder Familie."""
    for family in F.FAMILIES:
        at = _app(family_select=family, stretch_slider=C.STRETCH_MAX)
        assert not at.error, family
