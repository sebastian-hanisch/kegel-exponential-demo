"""Exponential- und Potenzkegel – wenn der Kegel nicht selbstdual ist - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 7 der Reihe „Konvexe und konische Optimierung“: Zwei dreidimensionale Kegel, die nicht symmetrisch sind (Exponentialkegel, Potenzkegel mit a ≠ 1/2), mit einer Barriere (ν = 3) in der Barrierenmethode aus Stück 5.
Die Demo zeigt, was sich überträgt (Barriere, Dualpunkt aus dem Pfad) und was nicht (die Nesterov–Todd-Skalierung aus Stück 6), und dass ν und `t · Fehler` zur Formulierung gehören, nicht zum Problem.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kxp_constants as C
import kxp_forms as A
from kxp_evaluation import (
    Settings,
    analyse,
    beta_text,
    de,
    dual_checks,
    lp_family,
    lse_family,
    mu_rows,
    nu_rows,
    num,
    p_text,
    scale_rows,
    sci,
    support_count,
    symmetry_rows,
    two_barriers,
    verdict_barriers,
    verdict_lp,
    verdict_lse,
    verdict_nu,
    verdict_run,
    verdict_scale,
    verdict_symmetry,
)
from kxp_layout import dunkles_design, kopf_css
from kxp_pdf_export import build_pdf
from kxp_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kxp_visualization import build_barriers, build_lp, build_lse, build_nu, build_shapes, build_symmetry

st.set_page_config(page_title="Exponential- und Potenzkegel – Sebastian Hanisch", layout="wide")


def _cloud_key(settings):
    """Ein Experiment, das nur von der Wolke abhängt, rechnet nicht neu, wenn Familie, p oder β wechseln."""
    return Settings(C.DEFAULT_FAMILY, settings.n, settings.stretch, settings.angle, settings.seed, C.DEFAULT_P, C.DEFAULT_BETA)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _barriers(key):
    return two_barriers(key)


@st.cache_data(show_spinner=False)
def _lp(key):
    return lp_family(key)


@st.cache_data(show_spinner=False)
def _lse(key):
    return lse_family(key)


@st.cache_data(show_spinner=False)
def _symmetry():
    return symmetry_rows()


@st.cache_data(show_spinner=False)
def _dual(settings):
    return dual_checks(settings)


@st.cache_data(show_spinner=False)
def _sweeps():
    return nu_rows()


@st.cache_data(show_spinner=False)
def _mu(settings):
    return mu_rows(settings)


@st.cache_data(show_spinner=False)
def _scale(settings):
    return scale_rows(settings)


def _above(x):
    return "< 1·10⁻⁷" if x < 1e-7 else de(x, 4) if x >= 1e-3 else sci(x)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kxp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Exponential- und Potenzkegel – wenn der Kegel nicht selbstdual ist")
st.markdown(
    """
**Die Barrierenmethode aus Stück 5 überträgt sich auf zwei neue Kegel, die NT-Skalierung aus Stück 6 nicht.** Der **Exponentialkegel** `y e^(x/y) ≤ z` und der **Potenzkegel** `u^a v^(1−a) ≥ |w|` haben je drei Koordinaten und eine
Barriere mit ν = 3, und die Barrierenmethode läuft im Prinzip unverändert (eigene Implementierung `kxp_barrier.py`). Sie sind aber **nicht symmetrisch**: ihre Automorphismengruppe ist zu klein, ihr Dualkegel ist ein anderer Kegel, und für das Beispiel der Reihe,
die kleinste umschließende Kugel einer Punktwolke, geben sie zwei Familien: die **ℓp-Kugeln** (Potenzkegel, zwischen Raute, Kugel und Würfel aus Stück 2) und die **log-sum-exp-Kugel** (Exponentialkegel, ein glatter Würfel).
Stück 7 der Reihe „Konvexe und konische Optimierung“.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Zwei Familien:** `‖x_i − c‖_p ≤ R` wird über Potenzkegel `(s_ij, R, x_ij − c_j)` mit `a = 1/p` und die Zeile `R − Σ_j s_ij ≥ 0` geschrieben; `log Σ exp(β(±(x_ij − c_j))) ≤ β R` über Exponentialkegel `(β(±(x_ij − c_j)) − βR, 1, z_ijk)` und `Σ z ≤ 1`.
2. **Barriere je Kegel:** Exponentialkegel `−log(y log(z/y) − x) − log y − log z`, Potenzkegel `−log(u^(2a) v^(2−2a) − w²) − (1 − a) log u − a log v`; beide logarithmisch homogen mit `ν = 3`.
3. **Barrierenmethode wie in Stück 5:** Newton mit Rücksetzen auf `t cᵀx + φ(b − A x)`, dann `t ← 8 t`, bis `ν/t` unter einer **relativen** Genauigkeit liegt.
4. **Dual aus dem Pfad:** `z = −∇φ(s)/t` liegt im **Dualkegel** `K*`, mit `sᵀz = ν/t`. Dieser Kegel ist ein anderer als `K`.
5. **Was fehlt:** Eine Nesterov–Todd-Skalierung `W` mit `W K = K` gibt es nicht, weil die Automorphismen nicht ausreichen (Experiment 4). Ein nichtsymmetrisches Primal-Dual ist nicht gebaut.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Form**")
    family = st.selectbox("Familie", A.FAMILIES, key="family_select", format_func=A.FAMILY_NAMES.get,
                          help="Welche der beiden Familien das Hauptdiagramm zeigt; die Experimente 2 und 3 vergleichen immer die ganze Familie.")
    p = st.select_slider("Exponent p (nur ℓp-Kugel)", options=C.P_CHOICES, key="p_select", format_func=p_text,
                         help="Potenzkegel mit a = 1/p: p = 1 ist die Raute, p = 2 die Kugel, ein großes p fast der Würfel.")
    beta = st.select_slider("Schärfe β (nur log-sum-exp-Kugel)", options=C.BETA_CHOICES, key="beta_select", format_func=beta_text,
                            help="Je größer β, desto näher liegt die Kugel am Würfel.")
    st.markdown("**Punktwolke**")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help="Wie viele Punkte die Wolke hat.")
    stretch = st.slider("Streckung", *bounds("stretch_slider"), step=0.1, key="stretch_slider", help="Wie viel breiter die Wolke entlang ihrer Längsachse ist als quer dazu.")
    angle = st.slider("Drehung (Grad)", *bounds("angle_slider"), key="angle_slider", help="Winkel der Längsachse gegen die x-Achse.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"family_select": family, "n_slider": int(n), "stretch_slider": float(stretch), "angle_slider": int(angle), "seed_input": int(seed), "p_select": float(p), "beta_select": float(beta)})

settings = Settings(family, int(n), float(stretch), int(angle), int(seed), float(p), float(beta))
a = _analysis(settings)
res = a.form.result

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("Radius R", num(res.value, 4), help=f"Radius der gewählten Kugel. Raute {num(a.others['diamond'], 4)}, Kugel {num(a.others['ball'], 4)}, Würfel {num(a.cube, 4)}.")
m2.metric("Barriere-Parameter ν", f"{a.form.problem.nu}", help=f"{A.CONE_NAMES[family]} zählen je 3, die Summenzeilen je 1.")
m3.metric("Newton-Schritte", f"{res.newton_steps}", help=f"In {res.rounds} äußeren Runden mit Schrittfaktor 8.")
m4.metric("Bewiesene Lücke ν/t", sci(res.gap_bound), help="Das Optimum liegt höchstens so weit unter dem Wert (gilt für zentrierte Punkte).")
m5.metric("Stützpunkte", f"{support_count(a.cloud.points, a.form.shape)}", help="Punkte auf dem Rand der Kugel.")
st.plotly_chart(build_shapes(a.cloud.points, a), width="stretch", key="main_chart")
st.caption("Die gewählte Kugel (dicke Linie) mit Raute, Kugel und Würfel zum Vergleich (gepunktet); Stützpunkte rot umrandet. Für p = 2 fällt die ℓp-Kugel mit der Kugel zusammen.")

if st.button("📄 PDF der aktuellen Wolke erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegel-exponential.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was ändert sich, wenn der Kegel nicht selbstdual ist?")
verdict(verdict_run(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Dieselbe Kugel, zwei Barrieren")
st.caption("Die Kugel (p = 2) der Wolke einmal als Lorentz-Kegel (Kern der Reihe) und einmal als Potenzkegel mit a = 1/2: derselbe Radius, aber ein anderes ν.")
bar = _barriers(_cloud_key(settings))
st.plotly_chart(build_barriers(bar), width="stretch", key="barriers_chart")
st.dataframe(pd.DataFrame([{"Formulierung": name, "ν": r["nu"], "Radius": num(r["value"], 8), "Newton-Schritte": r["newton"], "Runden": r["rounds"], "t · Fehler am Pfadende": f"{r['path'][-1]['m']:.2f}".replace(".", ",")}
                           for name, r in (("Lorentz-Kegel", bar["soc"]), ("Potenzkegel a = 1/2", bar["pow"]))]), hide_index=True, width="stretch")
verdict(verdict_barriers(bar))

st.markdown("---")

st.subheader("🔬 Experiment 2: Die ℓp-Familie")
st.caption("Der Radius der kleinsten ℓp-Kugel für p von 1 (Raute) über 2 (Kugel) bis 128 (fast der Würfel), alle als Potenzkegel gerechnet.")
lp_rows = _lp(_cloud_key(settings))
st.plotly_chart(build_lp(lp_rows, a.cube, a.others["diamond"], a.others["ball"]), width="stretch", key="lp_chart")
st.dataframe(pd.DataFrame([{"p": p_text(r["p"]), "Radius": num(r["radius"], 6), "über dem Würfel": _above(r["above_cube"]), "ν": r["nu"], "Newton-Schritte": r["newton"],
                            "Runden": r["rounds"], "zentriert": "ja" if r["centered"] else "nein (Rundungsgrenze)"} for r in lp_rows]), hide_index=True, width="stretch")
verdict(verdict_lp(lp_rows, a.cube))

st.markdown("---")

st.subheader("🔬 Experiment 3: Der glatte Würfel")
st.caption("Die log-sum-exp-Kugel mit wachsender Schärfe β gegen den Würfelradius, alle als Exponentialkegel gerechnet.")
lse_rows = _lse(_cloud_key(settings))
st.plotly_chart(build_lse(lse_rows), width="stretch", key="lse_chart")
st.dataframe(pd.DataFrame([{"β": beta_text(r["beta"]), "Radius": num(r["radius"], 6), "über dem Würfel": _above(r["above_cube"]), "Schranke log(2d)/β": num(r["bound"], 4), "ν": r["nu"],
                            "Newton-Schritte": r["newton"], "zentriert": "ja" if r["centered"] else "nein (Rundungsgrenze)"} for r in lse_rows]), hide_index=True, width="stretch")
verdict(verdict_lse(lse_rows))

st.markdown("---")

st.subheader("🔬 Experiment 4: Nicht symmetrisch")
st.caption("Die Dimension der Automorphismengruppe (lineare Abbildungen `A` mit `A K = K`), live aus 400 Randpunkten je Kegel gerechnet, und was der Dualkegel am Pfad tut.")
sym = _symmetry()
st.plotly_chart(build_symmetry(sym), width="stretch", key="sym_chart")
dual = _dual(settings)
st.dataframe(pd.DataFrame([{"Kegel": r["cone"], "Dimension der Gruppe": r["dim"], "transitiv möglich (≥ 3)": "ja" if r["homogeneous"] else "nein",
                            "Singulärwerte unter 10⁻⁶ (relativ)": sum(v < 1e-6 * r["singular"][0] for v in r["singular"])}
                           for r in sym]), hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"Prüfung": "−∇φ(s) im Inneren des Dualkegels", "Ergebnis": f"{dual['inside']} von {dual['samples']} Punkten"},
                           {"Prüfung": "schwache Dualität sᵀz ≥ 0 für Paare aus K × K*", "Ergebnis": f"{dual['violated']} Verletzungen bei {dual['pairs']} Paaren"},
                           {"Prüfung": f"Punkt {tuple(round(v, 3) for v in dual['witness'])}", "Ergebnis": f"in K: {'ja' if dual['witness_in_K'] else 'nein'}; in K*: {'ja' if dual['witness_in_dual'] else 'nein'}"},
                           {"Prüfung": f"Pfad der gewählten Wolke bei t = {sci(dual['t'], 1)}", "Ergebnis": f"ν/t = {sci(dual['nu_over_t'])}, sᵀz = {sci(dual['sz'])}, cᵀx + bᵀz = {sci(dual['cx_bz'])}, ‖Aᵀz + c‖ = {sci(dual['dual_residual'])}, z in K*: {'ja' if dual['dual_feasible'] else 'nein'}"}]),
             hide_index=True, width="stretch")
verdict(verdict_symmetry(sym, dual))

st.markdown("---")

st.subheader("🔬 Experiment 5: Newton-Schritte gegen ν, Schrittfaktor und Maßstab")
st.caption(f"Beide Familien auf {C.N_SWEEP[0]} bis {C.N_SWEEP[-1]} Punkten (Median über {len(C.SWEEP_SEEDS)} feste Wolken, ℓ3-Kugel und β = 10), der Schrittfaktor μ der gewählten Form, und dieselbe Wolke in anderem Datenmaßstab.")
nrows = _sweeps()
mrows = _mu(settings)
st.plotly_chart(build_nu(nrows, mrows), width="stretch", key="nu_chart")
st.dataframe(pd.DataFrame([{"Punkte": r["n"], "ν (ℓ3)": r["lp"]["nu"], "Newton (ℓ3)": r["lp"]["newton"], "ν (log-sum-exp)": r["lse"]["nu"], "Newton (log-sum-exp)": r["lse"]["newton"], "Runden": r["lp"]["rounds"]} for r in nrows]),
             hide_index=True, width="stretch")
st.dataframe(pd.DataFrame([{"Schrittfaktor μ": de(r["mu"], 0), "Newton-Schritte": r["newton"], "Runden": r["rounds"], "zentriert": "ja" if r["centered"] else "nein (Rundungsgrenze)"} for r in mrows]),
             hide_index=True, width="stretch")
verdict(verdict_nu(nrows, mrows))
srows = _scale(settings)
st.dataframe(pd.DataFrame([{"Datenmaßstab": sci(r["factor"], 0), "Radius": num(r["radius"], 6), "relativ: Newton": r["rel_newton"], "relativ: zentriert": "ja" if r["rel_centered"] else "nein",
                            "absolut 10⁻⁹: Newton": r["abs_newton"], "absolut: halbes Dekrement-Quadrat": sci(r["abs_lam2"]), "absolut: zentriert": "ja" if r["abs_centered"] else "nein"} for r in srows]), hide_index=True, width="stretch")
verdict(verdict_scale(srows))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die NT-Skalierung aus Stück 6 überträgt sich auf jeden Kegel** | Nein: Exponentialkegel und Potenzkegel (a ≠ 1/2) haben nur eine zweidimensionale Automorphismengruppe, ein dreidimensionaler Kegel braucht mindestens 3 (Experiment 4). Das nichtsymmetrische Primal-Dual (Skajaa–Ye, Clarabel) ist hier nicht gebaut. | – |
| **ν und `t · Fehler` sind Eigenschaften des Problems** | Sie gehören zur Formulierung: dieselbe Kugel hat als Lorentz-Kegel ν = 40 und `t · Fehler → 2`, als Potenzkegel ν = 140 und `→ 6` (Experiment 1). | – |
| **Der glatte Würfel braucht ein großes β, damit er dem Würfel gleicht** | Auf dieser Wolke genügt β = 10 (Abstand unter 10⁻⁷); die Schranke `log(2d)/β` sagt 0,14 und ist hier ab β = 2 um zwei und mehr Größenordnungen zu grob (bei β = 0,5 nur um den Faktor 4). Ein Messbefund dieser Wolken, kein Satz. | – |
| **Die Schranke `ν/t` gilt an jedem Pfadende** | Nur für zentrierte Punkte. Bei großem `t` endet die Zentrierung an der Rundungsgrenze (je nach Prozessor bei einzelnen β oder p); die Tabellen melden „zentriert: nein“, der Radius stimmt trotzdem. | – |
| **Eine absolute Genauigkeit passt zu jedem Maßstab** | Nein: bei 10⁶-fach größeren Daten erreicht `ν/t ≤ 10⁻⁹` sicher die Rundungsgrenze, bei 1000-fach je nach Prozessor (Experiment 5); die Demo rechnet mit relativer Genauigkeit. | – |
| **Die Beispiele stehen für Anwendungen** | Nur die beiden Kugel-Familien sind gerechnet; logistische Regression, Entropie-Maximierung und geometrische Programme (die Standardanwendungen) sind nicht gebaut. | – |
| **Iterationen sind Zeit** | Ein Newton-Schritt löst ein dichtes System der Größe der Variablen; die Demo zählt Schritte, keine Sekunden. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kegelprogramm.** $\min c^\top x$ mit $s = b - Ax \in K$, $K$ ein Produkt aus Orthanten, Exponentialkegeln und Potenzkegeln; Dual $\max -b^\top z$ mit $A^\top z = -c$, $z \in K^*$.

**Exponentialkegel.** $K_{\exp} = \operatorname{cl}\{(x, y, z): y > 0,\ y e^{x/y} \le z\}$. Barriere $\varphi(x, y, z) = -\log\big(y \log(z/y) - x\big) - \log y - \log z$, logarithmisch homogen mit $\nu = 3$:
$\varphi(\lambda s) = \varphi(s) - 3 \log \lambda$, also $\langle \nabla\varphi(s), s\rangle = -3$, $\nabla^2\varphi(s)\, s = -\nabla\varphi(s)$ und $\langle \nabla\varphi, (\nabla^2\varphi)^{-1}\nabla\varphi\rangle = 3$ (so prüft der Test die Barriere).
Dualkegel $K_{\exp}^* = \{(u, v, w): u < 0,\ -u\, e^{v/u} \le e\, w\} \cup \{(0, v, w): v, w \ge 0\}$.

**Potenzkegel.** $P_a = \{(u, v, w): u, v \ge 0,\ u^a v^{1-a} \ge |w|\}$ mit $0 < a < 1$; Barriere $\varphi = -\log(u^{2a} v^{2-2a} - w^2) - (1-a)\log u - a \log v$, $\nu = 3$. Für $a = 1/2$ ist $P_a$ der gedrehte Lorentz-Kegel $uv \ge w^2$.
Dualkegel $P_a^* = \{(u, v, w): (u/a)^a (v/(1-a))^{1-a} \ge |w|\}$.

**Die beiden Kugel-Familien.** ℓp-Kugel: $|x_{ij} - c_j| \le s_{ij}^{1/p} R^{1-1/p}$ ist $(s_{ij}, R, x_{ij} - c_j) \in P_{1/p}$; mit $\sum_j s_{ij} \le R$ folgt $\sum_j |x_{ij} - c_j|^p \le R^p$.
Log-sum-exp-Kugel: $e^{\beta(\pm(x_{ij} - c_j) - R)} \le z_{ijk}$ ist $(\beta(\pm(x_{ij} - c_j)) - \beta R, 1, z_{ijk}) \in K_{\exp}$; mit $\sum_k z_{ijk} \le 1$ folgt $\tfrac1\beta \log \sum_{j,\pm} e^{\beta(\pm(x_{ij} - c_j))} \le R$. Für $\beta \to \infty$ nähert sie sich dem Würfel von oben; der Abstand ist höchstens $\log(2d)/\beta$.

**Dual aus dem Pfad.** Am zentralen Punkt $x(t)$ verschwindet der Gradient von $t\,c^\top x + \varphi(b - Ax)$: $t c - A^\top \nabla\varphi(s) = 0$. Mit $z = -\nabla\varphi(s)/t$ heißt das $A^\top z = -c$; eine Barriere erfüllt $-\nabla\varphi(s) \in \operatorname{int} K^*$, also $z \in \operatorname{int} K^*$, und wegen $\langle \nabla\varphi(s), s\rangle = -\nu$ gilt $s^\top z = \nu/t$.

**Automorphismen.** Eine lineare Abbildung $A$ mit $A K = K$ erfüllt am glatten Rand $n(b)^\top A b = 0$ ($n$ Normale): die Lie-Algebra der Gruppe ist der Nullraum dieser linearen Bedingungen an vielen Randpunkten. Wirkt die Gruppe transitiv auf dem dreidimensionalen Inneren, hat sie mindestens Dimension 3 (Orthant 3, Lorentz-Kegel 4); der Exponentialkegel und $P_a$ mit $a \ne 1/2$ haben 2. Eine NT-Skalierung $W$ mit $W K = K$ und $W z = W^{-1} s$ für beliebige $s \in K$, $z \in K^*$ braucht diese Transitivität.

**Literatur.** P. R. Chares: *Cones and interior-point algorithms for structured convex optimization involving powers and exponentials*, Dissertation, Université catholique de Louvain (Promotionsjahr 2009 laut Mathematics Genealogy Project, die PDF-Datei auf der Seite des Betreuers trägt 2007; die Barriere des Potenzkegels mit ν = 3).
A. Skajaa und Y. Ye: *A homogeneous interior-point algorithm for nonsymmetric convex conic optimization*, Mathematical Programming 150 (2), 2015, S. 391–422 (Seitenangabe aus einer Verzeichnisseite, nicht aus dem Verlagsexemplar).
S. Akle Serrano: *Algorithms for unsymmetric cone optimization and an implementation for problems with the exponential cone*, Dissertation, Stanford University, 2015.
P. J. Goulart und Y. Chen: *Clarabel: An interior-point solver for conic programs with quadratic objectives*, arXiv:2405.12762, 2024 (Löser mit Exponential- und Potenzkegel, Gegenprobe der Tests).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
