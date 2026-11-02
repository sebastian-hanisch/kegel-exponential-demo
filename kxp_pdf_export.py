"""PDF-Export der aktuellen Wolke mit der gewählten Kugel (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import kxp_forms as A
from kxp_evaluation import Analysis, beta_text, de, p_text, sci

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "β": "beta", "κ": "kappa", "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
            "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "ℓ": "l", "∞": "inf", "√": "sqrt"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s, r = a.settings, a.form.result
    par = f"p = {p_text(s.p)}" if s.family == "lp" else f"beta = {beta_text(s.beta)}"
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Exponential- und Potenzkegel"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Familie: {A.FAMILY_NAMES[s.family]} ({par}); Wolke: {s.n} Punkte, Streckung {de(s.stretch, 1)}, um {s.angle} Grad gedreht, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    rows = [("Radius R", de(r.value, 6)), ("Raute (p = 1)", de(a.others["diamond"], 6)), ("Kugel (p = 2)", de(a.others["ball"], 6)), ("Wuerfel (p = inf)", de(a.cube, 6)), ("nu", a.form.problem.nu),
            ("Runden", r.rounds), ("Newton-Schritte", r.newton_steps), ("bewiesene Luecke nu/t", sci(r.gap_bound)), ("Abweichung vom engeren Referenzwert", sci(abs(r.value - a.reference))),
            ("zentriert am Ende", "ja" if r.centered else "nein (Rundungsgrenze)")]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(90, 6, "Kennzahl", border=1)
    pdf.cell(60, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, v in rows:
        pdf.cell(90, 5.5, _s(k), border=1)
        pdf.cell(60, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-exponential-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
