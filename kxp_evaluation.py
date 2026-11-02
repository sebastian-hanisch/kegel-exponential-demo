"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median

import numpy as np

import kegel_kern as K
import kxp_barrier as B
import kxp_constants as C
import kxp_cones as CO
import kxp_forms as F
from kxp_scenario import Cloud, make_cloud


@dataclass(frozen=True)
class Settings:
    family: str = C.DEFAULT_FAMILY
    n: int = C.DEFAULT_N
    stretch: float = C.DEFAULT_STRETCH
    angle: int = C.DEFAULT_ANGLE
    seed: int = C.DEFAULT_SEED
    p: float = C.DEFAULT_P
    beta: float = C.DEFAULT_BETA


@dataclass(frozen=True, eq=False)
class Analysis:
    settings: Settings
    cloud: Cloud
    form: F.FormResult
    reference: float         # dieselbe Form mit Genauigkeit 1e-13 (relativ): Vergleichswert für die Abweichung
    cube: float              # Radius des kleinsten Würfels (geschlossen: halbe größte Spannweite)
    others: dict             # Radien der Vergleichsformen: 'diamond' (p = 1), 'ball' (p = 2)


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def num(x: float, digits: int = 4) -> str:
    return sci(x, 2) if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e5) else de(x, digits)


ACCURACY = 1e-7      # unterhalb davon trägt der Abstand zum Würfel nur die Genauigkeit der Rechnung (Lücke etwa 3·10⁻⁹)


def above_text(x: float) -> str:
    return f"weniger als {sci(ACCURACY, 0)}" if x < ACCURACY else sci(x)


def over_text(x: float) -> str:
    return above_text(x) if x < ACCURACY else f"um {sci(x)}"


def p_text(p: float) -> str:
    return str(int(p)) if float(p).is_integer() else de(p, 2).rstrip("0").rstrip(",")


def beta_text(beta: float) -> str:
    return str(int(beta)) if float(beta).is_integer() else de(beta, 1)


# ----------------------------------------------------------------------------- Analyse der eingestellten Wolke

def cube_radius(points: np.ndarray) -> float:
    """Kleinster Würfel (achsenparallel, l-unendlich-Kugel): die halbe größte Spannweite über die Koordinaten."""
    return float(np.max(points.max(axis=0) - points.min(axis=0)) / 2.0)


def analyse(s: Settings) -> Analysis:
    cloud = make_cloud(s.n, s.stretch, s.angle, s.seed)
    form = F.solve_form(s.family, cloud.points, p=s.p, beta=s.beta, tol=C.TOL)
    ref = B.solve(form.problem, tol=1e-12).value
    others = {"diamond": F.solve_form("lp", cloud.points, p=1.0, tol=C.TOL).result.value, "ball": F.solve_form("lp", cloud.points, p=2.0, tol=C.TOL).result.value}
    return Analysis(s, cloud, form, ref, cube_radius(cloud.points), others)


def deviation(a: Analysis) -> float:
    return abs(a.form.result.value - a.reference)


def support_count(points: np.ndarray, shape: F.Shape, rel: float = 1e-5) -> int:
    """Punkte auf dem Rand der Kugel (Maß >= (1 - rel) R)."""
    dist = F.gauge(shape.family, points - shape.center, shape.p, shape.beta)
    return int(np.sum(dist >= shape.radius * (1.0 - rel)))


# ----------------------------------------------------------------------------- Experiment 1: dieselbe Kugel, zwei Barrieren

def _soc_path(points: np.ndarray, ts, p_star: float) -> list:
    prob = F.build_soc_ball(points)
    x = prob.x0.copy()
    out = []
    for t in ts:
        x, _, _ = K.center(prob, x, float(t), tol=1e-12, max_iter=200)
        out.append({"t": float(t), "error": float(prob.c @ x) - p_star})
    return out


def two_barriers(s: Settings) -> dict:
    """Die Kugel (p = 2) der Wolke mit dem Lorentz-Kegel (Kern, nu = 2 je Punkt) und mit dem Potenzkegel a = 1/2 (nu = 7 je Punkt)."""
    pts = make_cloud(s.n, s.stretch, s.angle, s.seed).points
    soc_prob = F.build_soc_ball(pts)
    soc = K.solve_barrier(soc_prob, tol=1e-9)
    p_star = K.solve_barrier(soc_prob, tol=1e-13).value
    pow_prob = F.build_lp_ball(pts, 2.0)
    pw = B.solve(pow_prob, tol=1e-9 / abs(float(pow_prob.c @ pow_prob.x0)))      # absolut 1e-9 wie der Kern
    soc_path = _soc_path(pts, C.PATH_TS, p_star)
    pow_path = B.trace_path(pow_prob, C.PATH_TS)
    last_t = pow_path[-1]["t"]
    shape = F.decode("lp", pts, pw.x, 2.0)
    return {"p_star": p_star, "soc": {"nu": K.barrier_parameter(soc_prob.cones), "value": soc.value, "newton": soc.newton_steps, "rounds": soc.rounds,
                                       "path": [{**r, "m": r["t"] * r["error"]} for r in soc_path]},
            "pow": {"nu": pow_prob.nu, "value": pw.value, "newton": pw.newton_steps, "rounds": pw.rounds,
                    "path": [{"t": r["t"], "error": r["value"] - p_star, "m": r["t"] * (r["value"] - p_star)} for r in pow_path],
                    "active": B.active_count(pow_prob, pow_path[-1]["x"], last_t)},
            "support": support_count(pts, shape)}


# ----------------------------------------------------------------------------- Experiment 2 und 3: die beiden Familien

def lp_family(s: Settings, ps=C.P_GRID) -> list:
    pts = make_cloud(s.n, s.stretch, s.angle, s.seed).points
    cube = cube_radius(pts)
    rows = []
    for p in ps:
        fr = F.solve_form("lp", pts, p=p, tol=C.TOL)
        rows.append({"p": p, "radius": fr.result.value, "above_cube": fr.result.value - cube, "nu": fr.problem.nu, "newton": fr.result.newton_steps, "rounds": fr.result.rounds, "centered": fr.result.centered, "problems": fr.original_problems})
    return rows


def lse_family(s: Settings, betas=C.BETA_GRID) -> list:
    pts = make_cloud(s.n, s.stretch, s.angle, s.seed).points
    cube = cube_radius(pts)
    d = pts.shape[1]
    rows = []
    for beta in betas:
        fr = F.solve_form("lse", pts, beta=beta, tol=C.TOL)
        rows.append({"beta": beta, "radius": fr.result.value, "above_cube": fr.result.value - cube, "bound": math.log(2.0 * d) / beta, "nu": fr.problem.nu, "newton": fr.result.newton_steps,
                     "rounds": fr.result.rounds, "centered": fr.result.centered, "problems": fr.original_problems})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: nicht symmetrisch

KINDS = (("Orthant (3 Koordinaten)", "l", None), ("Lorentz-Kegel (3 Koordinaten)", "q", None), ("Potenzkegel, a = 1/2", "p", 0.5), ("Potenzkegel, a = 1/3", "p", 1.0 / 3.0), ("Exponentialkegel", "e", None))


def symmetry_rows(count: int = C.SYMMETRY_POINTS) -> list:
    """Dimension der Automorphismengruppe je Kegel (Nullraum an `count` Randpunkten) und die Singulärwerte der Probe."""
    rows = []
    for name, kind, par in KINDS:
        dim, sv = CO.automorphism_dimension(kind, par, count)
        rows.append({"cone": name, "dim": dim, "homogeneous": dim >= 3, "singular": [float(v) for v in sv]})
    return rows


def dual_checks(s: Settings, samples: int = C.DUAL_SAMPLES, pairs: int = C.PAIR_SAMPLES, seed: int = 0) -> dict:
    """Der Dualkegel des Exponentialkegels ist ein anderer; -grad phi(s) liegt in seinem Inneren; schwache Dualität; der Dualpunkt am Pfad der gewählten Wolke."""
    rng = np.random.default_rng(seed)
    y = np.exp(rng.normal(size=samples))
    x = rng.normal(size=samples)
    z = y * np.exp(x / y) * np.exp(np.abs(rng.normal(size=samples)) + 0.05)
    _, g, _ = CO.exp_grad_hess(np.stack([x, y, z], axis=1))
    inside = int(CO.dual_exp_inside(-g).sum())
    yy = np.exp(rng.normal(size=pairs))
    xx = rng.normal(size=pairs)
    zz = yy * np.exp(xx / yy) * np.exp(np.abs(rng.normal(size=pairs)))
    uu = -np.exp(rng.normal(size=pairs))
    vv = rng.normal(size=pairs)
    ww = (-uu * np.exp(vv / uu) / math.e) * np.exp(np.abs(rng.normal(size=pairs)))
    violated = int(np.sum(xx * uu + yy * vv + zz * ww < -1e-12))
    pts = make_cloud(s.n, s.stretch, s.angle, s.seed).points
    prob = F.build_lse_ball(pts, s.beta) if s.family == "lse" else F.build_lp_ball(pts, s.p)
    res = B.solve(prob, tol=1e-6)
    xc, zpath = B.dual_point(prob, res)
    sl = B.slack(prob, xc)
    witness = (1.0, 1.0, 1.5 * math.e)
    return {"samples": samples, "inside": inside, "pairs": pairs, "violated": violated, "witness": witness, "witness_in_K": CO.exp_member(witness), "witness_in_dual": CO.dual_exp_member(witness),
            "nu": prob.nu, "t": res.t, "nu_over_t": prob.nu / res.t, "sz": float(sl @ zpath), "cx_bz": float(prob.c @ xc + prob.b @ zpath), "dual_residual": float(np.abs(prob.A.T @ zpath + prob.c).max()),
            "dual_feasible": B.dual_feasible_blocks(prob, zpath), "cone": F.CONE_NAMES[s.family]}


# ----------------------------------------------------------------------------- Experiment 5: Newton-Schritte gegen ν, Schrittfaktor μ, Datenmaßstab

def nu_rows(ns=C.N_SWEEP, seeds=C.SWEEP_SEEDS, p: float = 3.0, beta: float = 10.0) -> list:
    rows = []
    for n in ns:
        row = {"n": n}
        for fam in ("lp", "lse"):
            newton, rounds, nus = [], [], []
            for seed in seeds:
                pts = make_cloud(n, 3.0, 30, seed).points
                fr = F.solve_form(fam, pts, p=p, beta=beta, tol=C.TOL)
                newton.append(fr.result.newton_steps)
                rounds.append(fr.result.rounds)
                nus.append(fr.problem.nu)
            row[fam] = {"nu": nus[0], "newton": float(median(newton)), "rounds": float(median(rounds))}
        rows.append(row)
    return rows


def mu_rows(s: Settings, mus=C.MUS) -> list:
    pts = make_cloud(s.n, s.stretch, s.angle, s.seed).points
    prob = F.build_form(s.family, pts, p=s.p, beta=s.beta)
    rows = []
    for mu in mus:
        r = B.solve(prob, tol=C.TOL, mu=mu)
        rows.append({"mu": mu, "newton": r.newton_steps, "rounds": r.rounds, "centered": r.centered})
    return rows


def scale_rows(s: Settings, factors=C.SCALE_FACTORS) -> list:
    """Dieselbe Wolke mit dem Faktor f skaliert: relative gegen absolute Genauigkeit (nu/t <= 1e-9 absolut)."""
    pts = make_cloud(s.n, s.stretch, s.angle, s.seed).points
    rows = []
    for f in factors:
        prob = F.build_form(s.family, f * pts, p=s.p, beta=s.beta / f if s.family == "lse" else s.beta)
        rel = B.solve(prob, tol=C.TOL)
        absolute = B.solve(prob, tol=1e-9 / max(abs(float(prob.c @ prob.x0)), 1e-300))
        rows.append({"factor": f, "radius": rel.value, "rel_newton": rel.newton_steps, "rel_rounds": rel.rounds, "rel_centered": rel.centered,
                     "abs_newton": absolute.newton_steps, "abs_rounds": absolute.rounds, "abs_lam2": absolute.lam2, "abs_centered": absolute.centered})
    return rows


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_run(a: Analysis) -> tuple:
    s, r = a.settings, a.form.result
    fam = F.FAMILY_NAMES[s.family]
    par = f"p = {p_text(s.p)}, a = 1/{p_text(s.p)}" if s.family == "lp" else f"β = {beta_text(s.beta)}"
    text = (f"Die **{fam}** ({par}) hat den Radius **{num(r.value, 6)}** (Raute {num(a.others['diamond'], 4)}, Kugel {num(a.others['ball'], 4)}, Würfel {num(a.cube, 4)}). "
            f"Das Programm hat ν = **{a.form.problem.nu}** ({F.CONE_NAMES[s.family]} je drei, Zeilen je eins), die Barrierenmethode braucht **{r.rounds} Runden mit zusammen {r.newton_steps} Newton-Schritten** "
            f"bis zur bewiesenen Lücke {sci(r.gap_bound)} (Abweichung vom engeren Referenzwert {sci(deviation(a))}).")
    if a.form.original_problems:
        return "error", text + " Die Rückumformung besteht nicht: " + "; ".join(a.form.original_problems) + "."
    if not r.centered:
        return "warning", (text + f" Die letzte Zentrierung endete an der Rundungsgrenze (halbes Dekrement-Quadrat {sci(r.lam2)}): die Schranke ν/t gilt streng nur für zentrierte Punkte, der Radius stimmt trotzdem "
                           f"(Abweichung {sci(deviation(a))}). Die Rückumformung besteht.")
    return "success", text + " Die Rückumformung besteht."


def verdict_barriers(res: dict) -> tuple:
    soc, pw = res["soc"], res["pow"]
    last_soc, last_pow = soc["path"][-1], pw["path"][-1]
    return "info", (f"Dieselbe Kugel, derselbe Wert ({num(soc['value'], 6)} und {num(pw['value'], 6)}): als Lorentz-Kegel ist ν = {soc['nu']}, als Potenzkegel a = 1/2 ist ν = {pw['nu']}. "
                    f"Newton-Schritte {soc['newton']} und {pw['newton']}, beide in {soc['rounds']} und {pw['rounds']} Runden. Am Pfadende (t = {sci(last_pow['t'], 0)}) gilt `t · Fehler` = {de(last_soc['m'], 2)} für den Lorentz-Kegel "
                    f"und **{de(last_pow['m'], 2)}** für den Potenzkegel; die Kugel hat {res['support']} Stützpunkte, und {pw['active']} Bedingungen sind aktiv. "
                    f"ν und `m` gehören zur Formulierung, nicht zum Problem; der Potenzkegel mit a = 1/2 ist ein gedrehter Lorentz-Kegel.")


def verdict_lp(rows: list, cube: float) -> tuple:
    first, last = rows[0], rows[-1]
    mono = all(b["radius"] <= a["radius"] + 1e-9 for a, b in zip(rows, rows[1:]))
    bad = [r for r in rows if r["problems"]]
    loose = [r for r in rows if not r["centered"]]
    text = (f"Der Radius fällt {'monoton' if mono else 'nicht monoton'} von {num(first['radius'], 4)} (p = {p_text(first['p'])}, Raute) über {num(next(r for r in rows if r['p'] == 2.0)['radius'], 4)} (p = 2, Kugel) "
            f"auf {num(last['radius'], 6)} (p = {p_text(last['p'])}); der Würfel liegt bei {num(cube, 6)} (Abstand {above_text(last['above_cube'])}). Newton-Schritte {min(r['newton'] for r in rows)} bis {max(r['newton'] for r in rows)}."
            + (f" Nicht zentriert am Ende (Rundungsgrenze): p = {', '.join(p_text(r['p']) for r in loose)}." if loose else ""))
    return ("warning", text + " Die Rückumformung besteht nicht für p = " + ", ".join(p_text(r["p"]) for r in bad) + ".") if bad else ("info", text)


def verdict_lse(rows: list) -> tuple:
    first = rows[0]
    mid = next(r for r in rows if r["beta"] == 10.0)
    loose = [r for r in rows if not r["centered"]]
    return "info", (f"Die log-sum-exp-Kugel liegt für β = {beta_text(first['beta'])} um {de(first['above_cube'], 4)} über dem Würfelradius, für β = 10 {over_text(mid['above_cube'])} und für β = {beta_text(rows[-1]['beta'])} "
                    f"{over_text(rows[-1]['above_cube'])}. Die Schranke `log(2d)/β` sagt {de(first['bound'], 2)}, {de(mid['bound'], 3)} und {de(rows[-1]['bound'], 4)}: auf dieser Wolke, mit zwei Stützpunkten, ist sie um Größenordnungen zu grob "
                    f"(Messbefund, kein Satz). Newton-Schritte {min(r['newton'] for r in rows)} bis {max(r['newton'] for r in rows)}."
                    + (f" Nicht zentriert am Ende (Rundungsgrenze, CPU-abhängig): β = {', '.join(beta_text(r['beta']) for r in loose)}." if loose else ""))


def verdict_symmetry(rows: list, dual: dict) -> tuple:
    dims = {r["cone"]: r["dim"] for r in rows}
    exp_dim, pow_dim, half = dims["Exponentialkegel"], dims["Potenzkegel, a = 1/3"], dims["Potenzkegel, a = 1/2"]
    witness = "ein Punkt liegt in K, nicht in K*" if dual["witness_in_K"] and not dual["witness_in_dual"] else "der Zeuge trennt K und K* nicht"
    return "info", (f"Dimension der Automorphismengruppe: Orthant {dims['Orthant (3 Koordinaten)']}, Lorentz-Kegel {dims['Lorentz-Kegel (3 Koordinaten)']}, Potenzkegel a = 1/2 **{half}** (ein gedrehter Lorentz-Kegel), "
                    f"Potenzkegel a = 1/3 **{pow_dim}**, Exponentialkegel **{exp_dim}**: ein dreidimensionaler Kegel, auf dessen Innerem die Gruppe transitiv wirkt, braucht mindestens 3. Für die beiden Letzten gibt es keine "
                    f"NT-Skalierung, und der Dualkegel ist ein anderer Kegel ({witness}). Am Pfad geht es trotzdem: −∇φ liegt für {dual['inside']} von {dual['samples']} Punkten im Inneren des Dualkegels, "
                    f"{dual['violated']} von {dual['pairs']} Paaren verletzen die schwache Dualität, und für die gewählte Wolke ist `sᵀz` = {sci(dual['sz'])} bei ν/t = {sci(dual['nu_over_t'])}.")


def _range(lo, hi, unit: str = "") -> str:
    lo, hi = int(lo), int(hi)
    return f"{lo}{unit}" if lo == hi else f"{lo} bis {hi}{unit}"


def verdict_nu(rows: list, mu: list) -> tuple:
    a, b = rows[0], rows[-1]
    lp_n = [r["lp"]["newton"] for r in rows]
    lse_n = [r["lse"]["newton"] for r in rows]
    best = min(mu, key=lambda r: r["newton"])
    return "info", (f"Von n = {a['n']} bis n = {b['n']} Punkten wächst ν der ℓp-Kugel von {a['lp']['nu']} auf {b['lp']['nu']}, der log-sum-exp-Kugel von {a['lse']['nu']} auf {b['lse']['nu']}; die Newton-Schritte bleiben bei "
                    f"{int(min(lp_n))} bis {int(max(lp_n))} (ℓp) und {int(min(lse_n))} bis {int(max(lse_n))} (log-sum-exp), je {_range(min(r[f]['rounds'] for r in rows for f in ('lp', 'lse')), max(r[f]['rounds'] for r in rows for f in ('lp', 'lse')), ' Runden')} (Median über feste Wolken). Der Schrittfaktor μ: am wenigsten Newton-Schritte bei μ = {de(best['mu'], 0)} "
                    f"({best['newton']}), bei μ = {de(mu[0]['mu'], 0)} {mu[0]['newton']}, bei μ = {de(mu[-1]['mu'], 0)} {mu[-1]['newton']}.")


def verdict_scale(rows: list) -> tuple:
    big = rows[-1]
    return "info", (f"Datenmaßstab von {sci(rows[0]['factor'], 0)} bis {sci(big['factor'], 0)}: mit **relativer** Genauigkeit {_range(min(r['rel_newton'] for r in rows), max(r['rel_newton'] for r in rows), ' Newton-Schritte')} und stets "
                    f"zentriert; mit **absoluter** Genauigkeit 10⁻⁹ endet die Zentrierung bei dem Faktor {sci(big['factor'], 0)} {'nicht zentriert (halbes Dekrement-Quadrat ' + sci(big['abs_lam2']) + ')' if not big['abs_centered'] else 'noch zentriert'}: "
                    f"das ist die Rundungsgrenze der Rechnung, kein Fehler des Verfahrens.")
