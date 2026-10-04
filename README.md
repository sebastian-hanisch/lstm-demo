# LSTM – wie Gatter den Gradienten retten – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lstm-demo.streamlit.app/)**

Stück 5 der **Neuronale-Netze-Reihe** der "Konzepte"-Reihe im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Stück 4 zeigte: der Gradient eines RNN verschwindet mit wachsender Sequenzlänge $T$. Das
**LSTM** (Hochreiter & Schmidhuber 1997) hat einen **Zellzustand** mit einem eigenen
Rückwärtspfad, der nicht bei jedem Schritt mit einer Aktivierungsableitung multipliziert wird.
Diese Demo zeigt sowohl den strukturellen Beweis dafür **als auch eine echte, überraschende
Einschränkung**: ein nicht-verschwindender Gradient allein macht das Training bei großem $T$
noch nicht zuverlässig – Curriculum Learning schon.

**Einordnung in die Reihe:**

```
Perceptron (WURZEL)                              [gebaut]
 └─ MLP + Backpropagation                        [gebaut]
      ├─ CNN                                     [gebaut]
      └─ RNN                                     [gebaut]
           └─ LSTM                               [DIESES STÜCK]
                └─ Attention/Transformer         [gebaut]
```

**Ergebnis in Kürze:** Strukturell hält das LSTM einen dramatisch stärkeren Gradienten über die
Zeit als das RNN – bei $T=1200$ ist der RNN-Gradient am ersten Zeitschritt praktisch numerisch
verschwunden ($\approx 10^{-85}$), der LSTM-Gradient ist immer noch eine normale, darstellbare
Zahl ($\approx 3\cdot10^{-23}$), ein Unterschied von über 60 Zehnerpotenzen. **Aber**: beim
direkten Kaltstart-Training bei großem $T$ ist das LSTM **nicht** zuverlässiger als das RNN –
ab $T=80$ sogar unzuverlässiger (0 % gegen 17–33 % Erfolgsquote). Beide Architekturen fallen oft
in eine entartete "sage immer dasselbe voraus"-Lösung. **Curriculum Learning** (Bengio et al.
2009 – erst kurze, dann wachsende Sequenzen) löst das für **beide** Architekturen.

## Warum dieses Problem

Stück 4 endete mit einer offenen Frage: was tut man gegen den verschwindenden Gradienten? Die
historische Antwort (Hochreiter 1991, Hochreiter & Schmidhuber 1997) ist ein Zellzustand mit
Gattern, die den Gradienten gezielt durchlassen. Diese Demo prüft diese Antwort ehrlich – inkl.
einer echten Überraschung, die die ursprüngliche Plan-Hypothese widerlegt hat.

## Vorab-Hypothesen (vor der Messung notiert, hier geprüft)

| Hypothese | Ergebnis |
|---|---|
| LSTM-Gradient bleibt über einen viel größeren $T$-Bereich von null verschieden als beim RNN | ✅ bei $T=1200$: RNN $\approx10^{-85}$, LSTM $\approx3\cdot10^{-23}$ (>60 Zehnerpotenzen Unterschied) |
| Gradienten-Check (BPTT durch alle vier Tore) unter $10^{-6}$ | ✅ 1,6·10⁻⁷ |
| LSTM mit Toren fest auf 1 reduziert sich exakt auf $c_t=c_{t-1}+g_t,\ h_t=\tanh(c_t)$ | ✅ identische Ausgabe |
| ❌ **Widerlegt:** LSTM hält beim Kaltstart-Training bis zu einem deutlich größeren $T$ durch als RNN (konkretes Vielfaches) | ❌ **Beim Kaltstart ist das LSTM NICHT zuverlässiger** – ab $T=80$ sogar unzuverlässiger (0 % gegen 17–33 %). Ursache: beide Architekturen fallen oft in eine entartete Konstant-Vorhersage, unabhängig vom Gradientenfluss. |
| GRU (Nebenvergleich) hat ähnliche Reichweite wie LSTM, weniger Parameter | ❌ Bei dieser Skala/kurzem Budget ist GRU sogar **unzuverlässiger** als RNN/LSTM (10 %/0 %/0 % bei T=10/40/80) |
| ⚠️ **Plan-Erweiterung (Aufklärung des Rätsels):** Curriculum Learning behebt die Kaltstart-Schwäche für beide Architekturen ohne klaren LSTM-Vorteil in der Erfolgsquote | ✅ bei $T=150$: Kaltstart RNN 33 %/LSTM 0 % → Curriculum RNN 70 %/LSTM 70 % |

## Befunde (gemessen, keine Behauptungen)

**Gradientennorm am ersten Zeitschritt** (RNN $\|\delta_1\|$ vs. LSTM $\|dc_1\|$, letzter
Zeitschritt bleibt bei beiden $\approx0{,}5$–$1{,}3$):

| T | RNN $\|\delta_1\|$ | LSTM $\|dc_1\|$ |
|---|---|---|
| 10 | 0,348 | 0,120 |
| 40 | 2,86·10⁻⁴ | 0,039 |
| 80 | 1,08·10⁻⁵ | 0,010 |
| 150 | 6,75·10⁻¹¹ | 9,02·10⁻⁵ |
| 300 | 9,92·10⁻²² | 1,20·10⁻⁶ |
| 600 | 5,06·10⁻⁴³ | 5,86·10⁻¹² |
| 1200 | 1,01·10⁻⁸⁵ | 3,35·10⁻²³ |

**Kaltstart-Erfolgsquote** (6 Zufalls-Initialisierungen je $T$, SGD, kein Curriculum):

| T | RNN | LSTM |
|---|---|---|
| 10 | 100 % | 100 % |
| 40 | 67 % | 17 % |
| 80 | 17 % | 0 % |
| 150 | 33 % | 0 % |

**GRU-Nebenvergleich** (Cho et al. 2014, Kaltstart, 10 Initialisierungen): T=10: 10 %, T=40: 0 %,
T=80: 0 % – bei dieser Skala kein Vorteil, im Gegenteil unzuverlässiger als RNN/LSTM.

**Curriculum Learning** (Bengio et al. 2009, Adam, Stufen 10→20→40→80→150, 10 Initialisierungen,
getestet bei $T=150$): RNN 70 %, LSTM 70 % – gegenüber 33 %/0 % beim Kaltstart eine deutliche
Verbesserung für **beide** Architekturen, kein exklusiver LSTM-Vorteil in der Erfolgsquote.

## Modell und Verfahren

- `lstm_scenario.py` – Signal-in-Rauschen-Sequenzgenerator (eigenständige Kopie aus rnn-demo).
- `lstm_model.py` – `RNN`-Referenz (Kopie aus Stück 4), `LSTM` (vier Tore, BPTT von Hand), `GRU`
  (Nebenvergleich), `SGD`/`Adam`, `train_curriculum`.
- `lstm_evaluation.py` – Kaltstart-Sweep, Gradientennorm-Sweep, GRU-Sweep,
  Curriculum-Erfolgsquote, Gradienten-Check, Korrektheits-Kette (Tore fest auf 1).
- `lstm_visualization.py` – Plotly: Sequenz-Anzeige, Erfolgsquote-vs-T (RNN/LSTM), Gradientennorm-
  vs-T (log-Skala), Kaltstart-vs-Curriculum-Vergleich.

## Was die App zeigt

Sequenzlänge, verdeckte Einheiten, Trainingsgröße, Rauschen, Lernrate, Epochen und Seed in der
Sidebar; eine Beispielsequenz; Einzel-Vergleich RNN/LSTM für die aktuelle Konfiguration; der
Kaltstart-Erfolgsquote-Sweep und die Gradientennorm-Kurve als zentrale (überraschende) Befunde;
ein Curriculum-Learning-Vergleich per Knopf (on-demand, da zusätzlich teuer); ein
"📐"-Abschnitt mit Korrektheits-Kette, Gradienten-Check und dem GRU-Nebenvergleich.

## Was nicht funktioniert hat / Grenzen

**Echte, substantielle Plan-Korrektur:** Die ursprüngliche Hypothese ("LSTM hält bis zu einem
deutlich größeren T durch als RNN, konkretes Vielfaches") wurde durch die Vormessung
**widerlegt** – beim direkten Kaltstart-Training (SGD, keine Vorkehrungen) ist das LSTM nicht
zuverlässiger als das RNN, sondern ab $T=80$ sogar unzuverlässiger. Debugging zeigte: beide
Architekturen kollabieren oft auf eine **entartete Konstant-Vorhersage** (alle Scores nahe einem
kleinen konstanten Wert, unabhängig vom Eingang) – ein Optimierungslandschafts-Problem, nicht ein
Gradientenproblem (der Gradienten-Check bestätigt korrekte Ableitungen; die Gradientennorm-Messung
bestätigt, dass der LSTM-Gradient strukturell nicht verschwindet). Mehr Epochen, mehr
Trainingsdaten, ein anderer Optimierer (Adam) oder eine kleinere Gewichtsinitialisierung haben
das Problem NICHT behoben – **Curriculum Learning** (schrittweise wachsendes T während des
Trainings) hingegen schon, für beide Architekturen gleichermaßen. Das ist die interessantere,
ehrlichere Geschichte: ein nicht-verschwindender Gradient ist notwendig, aber nicht hinreichend
für zuverlässiges Training bei großem T.

**Grenzen:** Ein Bit muss über die Zeit getragen werden (einfachste Version der Aufgabe). Der
Curriculum-Vergleich nutzt Adam (nicht SGD) – ein bewusster Kompromiss, um die Rechenzeit
im Rahmen zu halten, nicht um einen Effekt zu verstecken (im Gegensatz zu Stück 4, wo Adam den
eigentlichen Effekt verdeckte: hier geht es nicht um Gradientenmessung, sondern um die
Erfolgsquote nach dem Training).

## Tests

29 Tests, `python -m pytest tests/ -v` (Laufzeit mehrere Minuten wegen der teuren Sweeps):
- `test_scenario.py` – Reproduzierbarkeit, Signalposition, Klassenbalance.
- `test_model.py` – Forward/Backward, Gradienten-Check, Gradientennorm-Vergleich,
  Korrektheits-Kette, Curriculum-Rauchtest.
- `test_evaluation.py` – Sweep-Funktionen mit billigen Parametern (Korrektheit, nicht die
  offiziellen Zahlen).
- `test_claims.py` – jede Zahl oben nachgerechnet, mit Toleranzband (Modul-Fixtures berechnen
  jeden teuren Sweep nur einmal).
- `test_presets.py`, `test_app.py` – Presets, Regler-Extremwerte, Footer.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `lstm_constants.py` | Regler-Grenzen, Presets |
| `lstm_scenario.py` | Sequenzgenerator |
| `lstm_model.py` | RNN, LSTM, GRU, SGD, Adam, Curriculum |
| `lstm_evaluation.py` | Sweeps, Gradienten-Check, Reduktions-Check |
| `lstm_visualization.py` | Plotly-Plots |
| `lstm_presets.py` | Permalink-Sync, Presets |
| `tests/` | pytest-Suite |

## Bewusst nicht umgesetzt

Kein eigenes GRU-Stück (Nebenvergleich, wie im Linien-Scoping festgelegt). Kein
Curriculum-Learning-Regler in der Sidebar (nur ein einzelner "Testen"-Knopf) – das würde die
ohnehin komplexe App überladen. Keine echten Langzeit-Datensätze – bewusst synthetisch, damit
jede Zahl exakt nachrechenbar bleibt.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
streamlit run app.py
```

## Literatur

- Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen.* Diplomarbeit, TU
  München (verschwindender/explodierender Gradient, formal identifiziert).
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8),
  1735–1780.
- Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for
  Statistical Machine Translation.* EMNLP 2014, 1724–1734 (GRU – Nebenvergleich).
- Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009). *Curriculum Learning.*
  Proceedings of ICML 2009.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Neuronale Netze: vom Perceptron zum Transformer](https://sebastianhanisch.net/konzepte-neuronale-netze.html).
