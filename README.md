# Exponential- und Potenzkegel – wenn der Kegel nicht selbstdual ist – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-exponential-demo.streamlit.app/)** -->

Stück 7 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst (Orthant, Lorentz-Kegel, PSD-Kegel, jetzt Exponential- und Potenzkegel). Vorgänger sind [konvexitaet-zertifikat-demo](https://github.com/sebastian-hanisch/konvexitaet-zertifikat-demo) (Stück 1), [kegelform-demo](https://github.com/sebastian-hanisch/kegelform-demo) (Stück 2: der Löser als Rechenwerk), [kruemmungsregeln-demo](https://github.com/sebastian-hanisch/kruemmungsregeln-demo) (Stück 3: Krümmungsregeln; dort bekamen `exp`, `log` und `log_sum_exp` ein Urteil, aber kein Lösen), [kegel-dualitaet-demo](https://github.com/sebastian-hanisch/kegel-dualitaet-demo) (Stück 4: das Dual), [kegel-barriere-demo](https://github.com/sebastian-hanisch/kegel-barriere-demo) (Stück 5: die Barrierenmethode) und [kegel-primal-dual-demo](https://github.com/sebastian-hanisch/kegel-primal-dual-demo) (Stück 6: der primal-duale Schritt mit Nesterov–Todd-Skalierung). Nachfolger ist die Projektion auf die Kegel (Stück 8). Die Linien-Spezifikation und die Einbindung in die Website folgen, wenn die Reihe vollständig ist.

## Kernfrage

**Was ändert sich, wenn der Kegel nicht selbstdual ist?** Der **Exponentialkegel** `y e^(x/y) ≤ z` und der **Potenzkegel** `u^a v^(1−a) ≥ |w|` haben je drei Koordinaten und eine logarithmisch homogene Barriere mit ν = 3. Die Barrierenmethode aus Stück 5 läuft mit ihnen im Prinzip **unverändert** (eigene Implementierung `kxp_barrier.py`), und der Dualpunkt `z = −∇φ(s)/t` am Pfad liegt im Dualkegel `K*`, mit `sᵀz = ν/t`. Aber: beide Kegel sind **nicht symmetrisch**. Ihre Automorphismengruppe hat Dimension 2, ein dreidimensionaler Kegel, auf dessen Innerem die Gruppe transitiv wirkt, braucht mindestens 3. Darum gibt es für sie keine Nesterov–Todd-Skalierung, und der primal-duale Schritt aus Stück 6 überträgt sich nicht; der Dualkegel ist ein **anderer** Kegel. Die Demo misst, **was sich überträgt, was nicht, und dass `ν` und die Zahl `m` aus `t · Fehler → m` zur Formulierung gehören, nicht zum Problem.**

Das Beispiel der Reihe wächst von selbst: die kleinste umschließende Form einer Punktwolke (Stück 2) bekommt zwei Familien. Die **ℓp-Kugeln** (Potenzkegel mit `a = 1/p`) verbinden Raute (p = 1), Kugel (p = 2) und Würfel (p → ∞); die **log-sum-exp-Kugel** (Exponentialkegel) ist der glatte Würfel.

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 6, unverändert): Orthant, Lorentz- und PSD-Kegel; hier nur als Gegenprobe der Kugel (p = 2) und für die Fixtures des Kerntests.
- **Kegel** (`kxp_cones.py`): Barrieren, Gradient und Hesse-Matrix beider Kegel, **gestapelt** gerechnet (N Kegel als Matrix `(N, 3)`); `y log(z/y)` statt `exp(x/y)`, damit nichts überläuft; der Dualkegel des Exponentialkegels; die Dimension der Automorphismengruppe als Nullraum der Bedingung `n(b)ᵀ A b = 0` an Randpunkten.
- **Barrierenmethode** (`kxp_barrier.py`): wie in Stück 5 Newton mit Rücksetzen auf `t cᵀx + φ(b − A x)`, dann `t ← 8 t`; über Orthant, Exponential- und Potenzkegel; **relative** Genauigkeit (`ν/t ≤ 10⁻⁹ · max(|cᵀx₀|, |cᵀx|)`), ein Rundungsstopp, der die Zentrierung am Ende ehrlich meldet (`centered`), und der Dualpunkt am Pfad mit Nachzentrieren (`dual_point`).
- **Formen** (`kxp_forms.py`): ℓp-Kugel (`|x_ij − c_j| ≤ s_ij^(1/p) R^(1−1/p)` als Potenzkegel, Summenzeile `R − Σ_j s_ij ≥ 0`), log-sum-exp-Kugel (Exponentialkegel `(β(±(x_ij − c_j)) − βR, 1, z_ijk)`, Summenzeile `1 − Σ z ≥ 0`), die Lorentz-Kugel des Kerns als Gegenprobe, Rückumformung in die Ausgangsaufgabe und Umriss.
- **Kein Eingriff in den Kern:** alles Neue steht in `kxp_*.py`; BLAS-Threads laufen mit einem Thread (`threadpoolctl`), weil die Matrizen klein sind.

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) dieselbe Kugel, zwei Barrieren (Lorentz-Kegel gegen Potenzkegel a = 1/2: ν, Newton-Schritte, `t · Fehler`); (2) die ℓp-Familie (Radius gegen p, von der Raute über die Kugel zum Würfel); (3) der glatte Würfel (Radius gegen β, die Schranke `log(2d)/β` daneben); (4) nicht symmetrisch (die Dimension der Automorphismengruppe je Kegel, live gerechnet, der Dualkegel, der Dualpunkt am Pfad); (5) Newton-Schritte gegen ν, der Schrittfaktor μ und der Datenmaßstab mit relativer und absoluter Genauigkeit.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen der Rundung auf Linux, steht die gemessene Zahl trotzdem hier); Sweeps laufen über feste Wolken `SWEEP_SEEDS` (drei). Die Standardwolke (20 Punkte, Streckung 4, 45 Grad, Seed 35) ist ausdrücklich eine einzelne.

| Frage | Befund |
|---|---|
| Standardwolke, ℓ₃-Kugel (Potenzkegel a = 1/3) | Radius **6,4146** (Raute 10,0895, Kugel 7,1675, Würfel 5,7840), ν = 140 (je Koordinate ein Potenzkegel, je Punkt eine Summenzeile), **12 Runden, 44 Newton-Schritte** bis zur bewiesenen Lücke 3,5·10⁻⁹, zwei Stützpunkte |
| Dieselbe Kugel, zwei Barrieren (p = 2) | Lorentz-Kegel (Stück 5): ν = 40, 34 Newton-Schritte; Potenzkegel a = 1/2: **ν = 140, derselbe Radius 7,167473551**, 39 bis 43 Newton-Schritte (CPU-abhängig; beide in 13 Runden bei absoluter Genauigkeit 10⁻⁹). `t · Fehler` am Pfadende (t = 10⁷): **2,00 für den Lorentz-Kegel, 6,00 für den Potenzkegel**; die Kugel hat 2 Stützpunkte, und 6 Bedingungen sind aktiv (je Stützpunkt zwei Potenzkegel und die Summenzeile). ν und `m` gehören zur Formulierung |
| ℓp-Familie (Standardwolke) | R(p) = **10,0895 (p = 1), 8,7936 (1,25), 8,0267 (1,5), 7,1675 (2), 6,4146 (3), 6,0814 (4), 5,7981 (8), 5,7841 (16), 5,7840 (ab 32)**: monoton fallend, immer über dem Würfel; ab p = 32 weniger als 10⁻⁷ darüber. Newton-Schritte 38 bis 83 (Spitze um p = 16), 11 bis 12 Runden |
| log-sum-exp-Kugel (Standardwolke) | R(β) − Würfelradius = **0,7185 (β = 0,5), 0,1116 (1), 0,0046 (2), 2,8·10⁻⁴ (3), 1,5·10⁻⁶ (5), unter 10⁻⁷ ab β = 10**. Die Schranke `log(2d)/β` sagt 2,77, 1,39, 0,69, 0,46, 0,28, 0,14: ab β = 2 um zwei und mehr Größenordnungen zu grob (bei β = 0,5 nur um den Faktor 4; Messbefund dieser Wolken, kein Satz). Newton-Schritte 44 bis 84 |
| Automorphismen-Dimension (400 Randpunkte je Kegel) | **Orthant 3, Lorentz-Kegel 4, Potenzkegel a = 1/2 4, Potenzkegel a ≠ 1/2 2, Exponentialkegel 2** (klare Lücke zwischen den Singulärwerten: die kleinen unter 10⁻¹⁵, die nächsten über 10⁻³ absolut; beim Exponentialkegel 4·10⁻³ absolut, 2·10⁻⁴ relativ zum größten Singulärwert). Der Potenzkegel mit a = 1/2 **ist** ein gedrehter Lorentz-Kegel. Ein Kegel, auf dessen dreidimensionalem Inneren die Gruppe transitiv wirkt, braucht mindestens 3: Exponential- und Potenzkegel (a ≠ 1/2) haben es nicht, also keine NT-Skalierung |
| Dualkegel und Dualpunkt am Pfad | `K* ≠ K`: der Punkt (1, 1, 1,5 e) liegt in K, nicht in K*. `−∇φ(s)` liegt für **400 von 400** Punkten im Inneren von `K*`, **0 von 20 000** Paaren aus `K × K*` verletzen `sᵀz ≥ 0` (zufällige Punkte: ein Plausibilitätstest mit derselben Formel für `K*`, kein unabhängiger Beweis). Am Pfad der Standardwolke: `sᵀz = ν/t` auf 10⁻⁸ relativ, nach scharfem Nachzentrieren `‖Aᵀz + c‖` unter 10⁻⁵ und `cᵀx + bᵀz` auf 20 % bei `ν/t`. **Ohne** Nachzentrieren (Zentrierungstoleranz 10⁻⁶) ist `Aᵀz + c` nur auf etwa 10⁻³ genau; am äußersten Pfadende (t über 10⁹) trägt der Dualpunkt kein Zertifikat auf 10⁻¹⁰ |
| Newton-Schritte gegen ν (n = 4 bis 40 Punkte, 3 feste Wolken) | ℓ₃-Kugel ν = 28 bis 280: **35 bis 40 Newton-Schritte**; log-sum-exp (β = 10) ν = 52 bis 520: **46 bis 59**; je 12 Runden: flach in ν, wie in Stück 5 |
| Schrittfaktor μ (ℓ₃-Kugel der Standardwolke) | μ = 2: 68, 8: 44, **32: 33**, 1000: 93 Newton-Schritte: das Minimum liegt bei mittlerem μ wie in Stück 5; bei μ = 1000 ist der Aufwand hier dagegen deutlich höher (93, der Lauf endet nicht zentriert) |
| Datenmaßstab 10⁻⁶ bis 10⁶ (ℓ₃-Kugel) | **Relative** Genauigkeit: 42 bis 46 Newton-Schritte, stets zentriert. **Absolute** Genauigkeit `ν/t ≤ 10⁻⁹`: bei Maßstab 10⁶ weit nicht zentriert (halbes Dekrement-Quadrat etwa 10⁴ bis 10⁵, einige Hundert Newton-Schritte), bei 10³ je nach Prozessor (Entwicklungsrechner 1,2·10⁻³: nicht zentriert; GitHub-Runner: noch zentriert): die Rundungsgrenze, kein Fehler des Verfahrens |
| Extreme Parameter | β = 0,01 und 10⁵, p = 1,05 und 128, Maßstab 10⁻⁶ bis 10⁶: kein Absturz, richtige Radien und eine bestehende Rückumformung auf der Standardwolke; **die Methode bricht dort nicht** (ich hatte es erwartet); das Pfadende kann an der Rundungsgrenze enden |

## Befunde und Korrekturen gegenüber dem Plan

- **Kein numerischer Zusammenbruch.** Der Plan nannte als offene Frage, wo die Barrierenmethode über diese Kegel bricht. Gemessen auf der Standardwolke, je eine Form und Wolke: kein Zusammenbruch bei β bis 10⁵, p von 1,05 bis 128 und Datenmaßstab 10⁻⁶ bis 10⁶ (das Pfadende kann an der Rundungsgrenze enden). Die Grenze der Demo ist, was sie nicht baut (das nichtsymmetrische Primal-Dual), nicht eine Rechengrenze.
- **Zentrierung am Pfadende.** Bei der absoluten Genauigkeit des Prototyps (`ν/t ≤ 10⁻⁹`) und hohem `t` erreichte die Zentrierung in einzelnen Läufen ihre Toleranz nicht (Rundung); der Bau hat deshalb die relative Genauigkeit und einen Rundungsstopp, und `centered` steht im Ergebnis, in den Tabellen der Experimente 2 und 3 und im Verdikt. Welche p oder β das trifft, hängt von der CPU ab (auf dem Entwicklungsrechner p = 2 mit halbem Dekrement-Quadrat 4,4·10⁻⁶ bei Toleranz 10⁻⁶).
- **Der Dualpunkt am Pfad braucht Nachzentrieren.** Der Plan nahm an, `z = −∇φ/t` sei am Pfadende ein Zertifikat. Nach der gewöhnlichen Zentrierung (Toleranz 10⁻⁶) liegt `Aᵀz + c` bei etwa `√(2·10⁻⁶)`, weil die Hesse-Matrix am Rand mit `t²` wächst; der Bau zentriert vor der Auswertung scharf nach (`dual_point`). Dasselbe hatte Stück 4 an den anderen Kegeln gemessen.
- **Newton-Schritte höher als in Stück 5.** ν zählt je Kegel 3 (140 für 20 Punkte der ℓp-Kugel), und die absolute Zahl liegt im ν-Sweep bei 35 bis 59 statt 30 bis 34, in den Spitzen bei p = 16 und β = 3 bei 83 und 84. Flach in ν bleibt sie. Die Newton-Zahl hängt außerdem von p und β ab (Spitzen um p = 16 und β = 2 bis 5, CPU-abhängig); sie ist ein Messbefund dieser Wolken, keine Eigenschaft der Kegel.
- **Die Deutung `t · Fehler = 6`** (je Stützpunkt zwei Potenzkegel und die Summenzeile = 3 aktive Bedingungen, bei zwei Stützpunkten 6) steht im Test über die Zahl der aktiven Bedingungen am Pfad (`active_count` über die Dualblöcke); die Zählung ist die des Plans, die Messung bestätigt sie auf der Standardwolke.
- **Quellen:** geprüft per Suche sind Skajaa und Ye (Mathematical Programming 150 (2), 2015, S. 391–422; die Seitenzahlen aus einer Verzeichnisseite, nicht aus dem Verlagsexemplar), Serrano (Dissertation, Stanford University, 2015), Chares (Dissertation, Université catholique de Louvain; Mathematics Genealogy nennt 2009, die PDF-Datei trägt 2007) und Goulart und Chen (Clarabel, arXiv:2405.12762, 2024). Dass Chares die Barriere des Potenzkegels mit ν = 3 angibt, habe ich nicht an der Dissertation selbst gelesen (Beleg: eine spätere Arbeit); ob das Skajaa–Ye-Verfahren dort so beschrieben ist, wie es hier im Satz genannt wird, ebenfalls nicht. Beweise der Quellen beweise ich hier nicht.
- **Port:** 9305 (`launch.json` und Netstat gelesen, frei).

## Ehrliche Grenzen

- **Kein nichtsymmetrisches Primal-Dual.** Die Demo zeigt, dass die NT-Skalierung nicht existiert (Dimension der Gruppe 2 statt mindestens 3) und was am Pfad geht (Dualpunkt in `K*`); sie baut kein Verfahren, das beide Kegel primal-dual löst (Skajaa–Ye, Clarabel nur genannt, Clarabel in den Tests als Gegenprobe).
- **Zwei Kugel-Familien sind keine Anwendungen.** Logistische Regression, Entropie-Maximierung und geometrische Programme, die Standardanwendungen der beiden Kegel, sind nicht gebaut; die ℓp-Kugel mit rationalem p lässt sich auch mit Lorentz-Kegeln schreiben (cvxpy tut es für größere p; hier nicht gemessen, nur dass der Potenzkegel sie mit einem Kegel je Koordinate schreibt).
- **Die Newton-Schritte hängen kaum an ν** ist ein Messbefund dieser Implementierung (Faktor 8 für `t`, Zentrieren auf 10⁻⁶), keine Eigenschaft der Kegel. Zeiten werden nicht gemessen.
- **Rechengrenze:** am äußersten Pfadende endet die Zentrierung in einzelnen Läufen an der Rundungsgrenze; die Tabellen melden „zentriert: nein“, der Radius stimmt trotzdem (Abweichung unter 10⁻⁷). Die Schranke `ν/t` gilt streng nur für zentrierte Punkte.
- **Die Schranke `log(2d)/β`** für den glatten Würfel ist ein Satz, nur ihre Schärfe auf diesen Wolken ist gemessen: mit zwei Stützpunkten liegt der Abstand weit darunter. Auf anderen Wolken kann er näher an der Schranke liegen.
- **Die Dimension der Automorphismengruppe** ist numerisch aus 400 Randpunkten bestimmt (Nullraum mit klarer Singulärwert-Lücke), nicht bewiesen; für den Orthant und den Lorentz-Kegel stimmt sie mit der bekannten Dimension überein, für den Potenzkegel a = 1/2 mit dem gedrehten Lorentz-Kegel.
- **Der Exponentialkegel** wird nur über `y log(z/y) − x > 0` ausgewertet; Punkte mit `y` gegen 0 und extremem `x/y` liegen im Rundungsbereich der Barriere (nicht gemessen).

## Verifikation

- **Kegel** (`tests/test_cones.py`): Barrieren von Hand (`(0, 1, e²)`: `φ = −log 2 − 2`, `∇φ = (½, −3/2, −3e⁻²/2)`; Potenzkegel `(2, 8, 0)`), die Identitäten einer logarithmisch homogenen Barriere (`⟨∇φ, s⟩ = −3`, `∇²φ s = −∇φ`, `⟨∇φ, (∇²φ)⁻¹∇φ⟩ = 3`, `φ(λ s) = φ(s) − 3 log λ`) für beide Kegel und drei Werte von a, Gradient und Hesse-Matrix gegen finite Differenzen in mittleren Punkten und mit zwei Schrittweiten, Hesse-Matrix positiv definit auch am Rand, Punkte außerhalb gleich unendlich, Potenzkegel a = ½ gleich `uv ≥ w²`, der Dualkegel von Hand (Randpunkt `(−1, −1, 1)`), `−∇φ` im Dualkegel, schwache Dualität an 20 000 Paaren, die Dimension der Automorphismengruppe je Kegel (3, 4, 4, 2, 2) mit Lücke zwischen den Singulärwerten.
- **Barrierenmethode** (`tests/test_barrier.py`): zwei Mini-Programme von Hand (`min z, eˣ ≤ z, x ≥ 1`: Optimum e; `min −w, (2, 8, w) ∈ P_½`: Optimum −4), Gradient von `f_t` verschwindet, die Newton-Matrix gleich der Hesse-Matrix per finiten Differenzen (beide Familien), Zentrierung senkt `f_t` und endet mit vollen Schritten, Dualpunkt dual zulässig mit `sᵀz = ν/t`, Dualkegel des Potenzkegels, relative Genauigkeit gegen Maßstab, Start außerhalb abgelehnt, Rundungsstopp, Pfad monoton, aktive Bedingungen.
- **Formen** (`tests/test_forms.py`): zwei antipodale Punkte von Hand (ℓp: `(1 + 2ᵖ)^(1/p)`, log-sum-exp: `(1/β) log(e^β + e^(2β) + e^(−β) + e^(−2β))`), p = 1/2/groß gleich Raute/Kugel/Würfel, p = 2 gleich der Lorentz-Kugel des Kerns bei ν 140 gegen 40, Monotonie, Schranke `log(2d)/β`, Maß und Umriss, Rückumformung erkennt einen Verstoß.
- **Orakel** (`tests/test_oracle_kegel_exponential.py`): ℓp-Kugeln gegen cvxpy mit Clarabel und `pnorm(approx=False)` (5 Wolken, p = 1,5 bis 8), log-sum-exp-Kugeln gegen cvxpy (β = 0,5 bis 10), der Würfelradius gegen cvxpy und die geschlossene Form, p = 2 gegen den Lorentz-Kern, p = 1 gegen das LP (scipy/HiGHS), die bewiesene Lücke deckt den Fehler am Pfad, Dualpunkt am Pfad mäßig und am Ende.
- **Randfälle** (`tests/test_edge_cases.py`): ein Punkt, d = 1 und 3, extreme p und β, doppelte Punkte, Datenmaßstab, die Maßstab-Probe, Eckwerte der Regler, die Probe der Automorphismen unabhängig von der Stichprobenzahl.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: jede Zahl dieser README), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich, jede Familie, jedes p, jedes β, PDF), Presets (die Zahlen der Hilfetexte), Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe.
- **Umfang:** 180 Tests (3 davon, die cvxpy-Orakel gegen Clarabel, überspringen sich ohne cvxpy; mit cvxpy laufen alle 8 Orakel-Tests grün), rund 2,7 Minuten (2026-11-03, Windows).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy und scipy sind Gegenproben im dev-Satz; ohne sie überspringen sich die Orakel-Tests, die sie brauchen. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 6 |
| `kxp_cones.py` | Exponential- und Potenzkegel: Barrieren (gestapelt), Dualkegel, Automorphismen |
| `kxp_barrier.py` | Barrierenmethode über Orthant, Exponential- und Potenzkegel, Pfad, Dualpunkt |
| `kxp_forms.py` | ℓp- und log-sum-exp-Kugel als Kegelprogramm samt Rückumformung und Umriss, Lorentz-Kugel des Kerns |
| `kxp_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kxp_evaluation.py` | Analyse, Experimente, Verdikte, Formatierung |
| `kxp_layout.py` | Fester Kopf (Stylesheet) |
| `kxp_pdf_export.py` | PDF-Export (fpdf2) |
| `kxp_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kxp_scenario.py` | Punktwolken (gestreckt, gedreht) |
| `kxp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Kegel, Barrierenmethode, Formen, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern |

## Bewusst nicht umgesetzt

- **Nichtsymmetrisches Primal-Dual** (Skajaa–Ye, Dahl–Andersen, Clarabel): nur genannt, nicht gebaut.
- **Anwendungen der Kegel:** logistische Regression, Entropie-Maximierung, geometrische Programme (Stück 3 übersetzt Ausdrücke in Kegel; die Anwendung bleibt dort).
- **Ein nichtsymmetrischer Kegel in anderer Barriere** (z. B. Näherung des Exponentialkegels durch Lorentz-Kegel): nicht gemessen.
- **Beweise** der Selbstkonkordanz und der Iterationsschranken: nur als Hintergrund genannt.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html).
