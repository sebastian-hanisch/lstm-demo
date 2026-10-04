"""LSTM — wie Gatter den Gradienten retten

Sebastian Hanisch - Operations Research und Machine Learning

Stück 5 der "Neuronale Netze"-Reihe der "Konzepte"-Reihe:
Perceptron -> MLP+Backpropagation -> {CNN, RNN -> LSTM -> Attention/Transformer}.
Stück 4 zeigte: der Gradient eines RNN verschwindet mit wachsender Sequenzlänge
T. Das LSTM (Hochreiter & Schmidhuber 1997) hat einen Zellzustand mit einem
eigenen Rückwärtspfad, der nicht bei jedem Schritt schrumpft. Diese App zeigt
sowohl den Beweis dafür ALS AUCH eine echte, überraschende Einschränkung: der
nicht-verschwindende Gradient allein reicht nicht, um direktes Training bei
großem T zuverlässig zu machen - Curriculum Learning schon.

Lauffähig mit: streamlit run app.py
"""
import numpy as np
import streamlit as st

import lstm_constants as C
import lstm_evaluation as ev
import lstm_model as m
import lstm_presets as pr
import lstm_scenario as sc
import lstm_visualization as viz

st.set_page_config(page_title="LSTM", layout="wide")


@st.cache_data(show_spinner=False)
def _analyse(T, hidden, n_train, n_test, noise, eta, epochs, seed):
    settings = ev.Settings(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                           eta=eta, epochs=epochs, seed=seed)
    out = ev.analyse(settings)
    return {
        "rnn_acc": out["rnn"]["test_accuracy"], "lstm_acc": out["lstm"]["test_accuracy"],
        "rnn_errs": out["rnn"]["result"].errors_per_epoch,
        "lstm_errs": out["lstm"]["result"].errors_per_epoch,
    }


@st.cache_data(show_spinner=False)
def _t_sweep():
    return ev.t_sweep()


@st.cache_data(show_spinner=False)
def _gradient_norm_sweep():
    return ev.gradient_norm_sweep()


@st.cache_data(show_spinner=False)
def _gradient_check():
    return ev.gradient_check()


@st.cache_data(show_spinner=False)
def _reduction_check():
    out = ev.reduction_check_gates_fixed()
    return out["s_lstm"], out["s_manual"], out["identical"]


@st.cache_data(show_spinner=False)
def _curriculum_rate(model_name):
    cls = m.RNN if model_name == "rnn" else m.LSTM
    return ev.curriculum_success_rate(cls)


@st.cache_data(show_spinner=False)
def _gru_sweep():
    return ev.gru_cold_sweep()


st.title("🧠 LSTM — wie Gatter den Gradienten retten")
st.markdown(
    "Stück 4 zeigte: der Gradient eines RNN verschwindet mit wachsender Sequenzlänge $T$ - "
    "über 10 Zehnerpotenzen bis $T=150$. Das **LSTM** (Hochreiter & Schmidhuber 1997) hat einen "
    "**Zellzustand** $c_t$ mit einem eigenen Rückwärtspfad, der nicht bei jedem Schritt mit "
    "einer Aktivierungsableitung multipliziert wird - solange das Vergessenstor nahe 1 bleibt, "
    "kann der Fehler über viele Zeitschritte fast unverändert zurückfließen."
)
st.caption(
    "Stück 5 (Kind von RNN) der 'Neuronale Netze'-Reihe. Folgestück: "
    "[Attention/Transformer](https://sebastianhanisch-attention-transformer-demo.streamlit.app/) - dieselbe Aufgabe, unabhängig von T."
)

with st.expander("So funktioniert das LSTM", expanded=True):
    st.markdown(
        "1. Drei Tore (Vergessen $f_t$, Eingabe $i_t$, Ausgabe $o_t$) und ein Kandidat $g_t$, "
        "alle aus $x_t$ und $h_{t-1}$ berechnet.\n"
        "2. Zellzustand: $c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$ - eine ADDITION, keine "
        "Multiplikation mit einer schrumpfenden Ableitung.\n"
        "3. Sichtbarer Zustand: $h_t = o_t \\odot \\tanh(c_t)$.\n"
        "4. Rückwärts: $\\partial c_{t-1}/\\partial c_t \\approx f_t$ statt $\\tanh'(z)\\cdot "
        "W_{hh}$ beim RNN - das ist die 'Gradienten-Autobahn'."
    )

st.caption("🎯 Schnellstart – ein Klick lädt ein durchgerechnetes Beispiel:")
preset_cols = st.columns(len(C.PRESETS))
for col, (key, preset) in zip(preset_cols, C.PRESETS.items()):
    with col:
        st.button(preset["label"], help=preset["help"], on_click=pr.apply_preset, args=(key,),
                   use_container_width=True)

st.caption("🔗 Die Adresszeile speichert deine Einstellungen als Permalink.")

pr.load_permalink_settings()
pr.init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    T = st.slider("Sequenzlänge T", C.T_MIN, C.T_MAX, ss["T"], key="widget_T",
                 on_change=pr.store_from_widget, args=("T",))
    ss["T"] = T
    hidden = st.slider("Verdeckte Einheiten", C.HIDDEN_MIN, C.HIDDEN_MAX, ss["hidden"],
                       key="widget_hidden", on_change=pr.store_from_widget, args=("hidden",))
    ss["hidden"] = hidden
    n_train = st.slider("Trainingsbeispiele", C.N_TRAIN_MIN, C.N_TRAIN_MAX, ss["n_train"],
                        step=10, key="widget_n_train", on_change=pr.store_from_widget,
                        args=("n_train",))
    ss["n_train"] = n_train
    noise = st.slider("Rauschanteil", C.NOISE_MIN, C.NOISE_MAX, ss["noise"], step=0.05,
                      key="widget_noise", on_change=pr.store_from_widget, args=("noise",))
    ss["noise"] = noise
    eta = st.slider("Lernrate η", C.ETA_MIN, C.ETA_MAX, ss["eta"], step=0.02,
                    key="widget_eta", on_change=pr.store_from_widget, args=("eta",))
    ss["eta"] = eta
    epochs = st.slider("Epochen", C.EPOCHS_MIN, C.EPOCHS_MAX, ss["epochs"], step=10,
                       key="widget_epochs", on_change=pr.store_from_widget, args=("epochs",))
    ss["epochs"] = epochs
    seed = st.number_input("Seed", value=ss["seed"], step=1, key="widget_seed",
                           on_change=pr.store_from_widget, args=("seed",))
    ss["seed"] = seed
    st.button("🎲 Zufälliger Seed", on_click=pr.randomize_seed)
    n_test = C.N_TEST_DEFAULT

pr.sync_query_params(dict(T=T, hidden=hidden, n_train=n_train, n_test=n_test, noise=noise,
                         eta=eta, epochs=epochs, seed=seed))

st.markdown("---")
st.subheader("📈 Eine Beispielsequenz")
rng_preview = np.random.default_rng(int(seed) + 5000)
seq_preview, label_preview = sc.make_sequence(rng_preview, T, noise_std=noise)
st.plotly_chart(viz.build_sequence_figure(seq_preview, label_preview),
                key=f"seq_{T}_{noise}_{seed}", use_container_width=True)

with st.spinner("Trainiere RNN und LSTM (Kaltstart, kein Curriculum)..."):
    out = _analyse(T, hidden, n_train, n_test, noise, eta, epochs, int(seed))

st.subheader("🎯 Was am Ende steht (Kaltstart, direkt bei diesem T)")
m1, m2, m3 = st.columns(3)
m1.metric("RNN-Testgenauigkeit", f"{out['rnn_acc']*100:.0f}%")
m2.metric("LSTM-Testgenauigkeit", f"{out['lstm_acc']*100:.0f}%")
m3.metric("Sequenzlänge T", f"{T}")

st.markdown("---")
st.subheader("🎯 Kaltstart-Erfolgsquote vs. Sequenzlänge")
with st.spinner("Berechne Erfolgsquote über 6 Zufalls-Initialisierungen je Sequenzlänge, "
               "RNN und LSTM (einmalig, kann bis zu ~2,5 Minuten dauern - lange Sequenzen "
               "sind langsam zu trainieren)..."):
    sweep = _t_sweep()
st.plotly_chart(viz.build_t_sweep_figure(sweep), key="t_sweep_chart", use_container_width=True)
st.warning(
    "⚠️ **Überraschung:** entgegen der naiven Erwartung scheitert das LSTM hier NICHT "
    "seltener als das RNN - ab T=80 sogar öfter (0 % gegen 17-33 %). Ein nicht-verschwindender "
    "Gradient allein garantiert kein erfolgreiches Training - siehe Korrektur unten."
)

st.subheader("🎯 Die strukturelle Ursache: Gradientennorm am ersten Zeitschritt")
grad_rows = _gradient_norm_sweep()
st.plotly_chart(viz.build_gradient_norm_figure(grad_rows), key="grad_norm_chart",
                use_container_width=True)
st.caption(
    "Trotzdem gilt strukturell: bei T=1200 ist der RNN-Gradient (‖δ₁‖≈1·10⁻⁸⁵) in "
    "Gleitkommazahlen praktisch null, der LSTM-Gradient (‖dc₁‖≈3·10⁻²³) ist immer noch eine "
    "normale, von null verschiedene Zahl - 62 Zehnerpotenzen Unterschied. Der Gradient "
    "verschwindet strukturell NICHT beim LSTM - das reicht nur allein nicht für "
    "zuverlässiges Kaltstart-Training (siehe unten)."
)

st.markdown("---")
st.subheader("🔬 Plan-Korrektur: Curriculum Learning löst das Rätsel")
st.markdown(
    "Wenn beide Architekturen beim Kaltstart ähnlich scheitern, obwohl der Gradient beim LSTM "
    "nicht verschwindet - woran liegt das? Beide Netze fallen bei großem T oft in eine "
    "**entartete Lösung** (immer dieselbe Konstante vorhersagen). **Curriculum Learning** "
    "(Bengio et al. 2009) trainiert dasselbe Netz erst auf kurzen, dann auf wachsenden "
    f"Sequenzen ({', '.join(str(t) for t in C.CURRICULUM_STAGES)}) - das umgeht die entartete "
    "Lösung für BEIDE Architekturen."
)
if st.button("🧪 Curriculum-Training testen (T=150, ~10 Zufalls-Initialisierungen je Modell, "
            "kann ~1,5 Minuten dauern)"):
    with st.spinner("Trainiere RNN und LSTM mit wachsendem T (Curriculum)..."):
        curriculum_rnn = _curriculum_rate("rnn")
        curriculum_lstm = _curriculum_rate("lstm")
    cold_row = [r for r in sweep if r["T"] == C.CURRICULUM_T][0]
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(
            viz.build_curriculum_comparison_figure(cold_row["rate_rnn"], curriculum_rnn, "RNN"),
            key="curriculum_rnn_chart", use_container_width=True)
    with col_b:
        st.plotly_chart(
            viz.build_curriculum_comparison_figure(cold_row["rate_lstm"], curriculum_lstm, "LSTM"),
            key="curriculum_lstm_chart", use_container_width=True)
    st.caption(
        f"Bei T={C.CURRICULUM_T}: Curriculum hebt die Erfolgsquote für BEIDE Architekturen "
        f"deutlich an ({curriculum_rnn*100:.0f}% / {curriculum_lstm*100:.0f}%) - kein "
        "exklusiver LSTM-Vorteil in der Erfolgsquote, sobald beide fair mit Curriculum "
        "trainiert werden."
    )

st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    "| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |\n"
    "|---|---|---|\n"
    "| Nicht-verschwindender Gradient reicht für Trainierbarkeit | Reicht NICHT allein - "
    "die Optimierungslandschaft (entartete Lösungen) ist ein eigenes Problem | Curriculum "
    "Learning (gezeigt oben) |\n"
    "| Ein Bit reicht als Erinnerung | Komplexere Aufgaben (Temporal Order) wären noch "
    "schwerer | — |\n"
    "| Zugriff auf Information nur über Zustand, der Schritt für Schritt weitergegeben wird | "
    "Direkter Zugriff auf JEDEN Zeitschritt (keine Weitergabe nötig) ist eine andere Lösung | "
    "Attention/Transformer (nächstes Stück) |\n"
)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Rückwärtspass (BPTT durch alle vier Tore):** bei Fehlklassifikation sind die Kernschritte
$\delta c_t = \delta c_{t+1}\cdot f_{t+1} + \delta h_t \cdot o_t \cdot (1-\tanh^2(c_t))$ und
$\delta c_{t-1} = \delta c_t \cdot f_t$ - **keine** Multiplikation mit $\tanh'(z)\cdot W$ wie
beim RNN.

**Korrektheits-Kette** (Vergessens-/Eingabe-/Ausgabetor fest auf 1): $c_t = c_{t-1} + g_t$,
$h_t = \tanh(c_t)$ - ein RNN mit Restverbindung, kein gewöhnliches Elman-RNN.
"""
    )
    s_lstm, s_manual, identical = _reduction_check()
    c1, c2 = st.columns(2)
    c1.metric("LSTM-Ausgabe (Tore=1)", f"{s_lstm:.6f}")
    c2.metric("Manuell (Restverbindung)", f"{s_manual:.6f}")
    st.metric("Identisch?", "Ja" if identical else "Nein (Fehler!)")
    st.markdown("**Gradienten-Check** (BPTT durch alle vier Tore, gegen finite Differenzen):")
    st.metric("Maximaler relativer Fehler", f"{_gradient_check():.2e}")

    st.markdown("**GRU-Nebenvergleich** (Cho et al. 2014, kein eigenes Stück):")
    gru_rows = _gru_sweep()
    st.dataframe(
        {"T": [r["T"] for r in gru_rows], "GRU-Erfolgsquote": [f"{r['rate_gru']*100:.0f}%" for r in gru_rows]},
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Bei dieser kleinen Skala zeigt auch GRU keinen Vorteil - im Gegenteil, es ist hier "
        "sogar unzuverlässiger als RNN/LSTM (gemessen, nicht erwartet)."
    )
    st.markdown(
        "**Literatur:** Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen "
        "Netzen.* Diplomarbeit, TU München. — Hochreiter, S. & Schmidhuber, J. (1997). "
        "*Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780. — Cho, K. et al. "
        "(2014). *Learning Phrase Representations using RNN Encoder-Decoder.* EMNLP 2014. — "
        "Bengio, Y. et al. (2009). *Curriculum Learning.* ICML 2009."
    )
    st.caption(
        "Implementiert in `lstm_model.py` (RNN, LSTM, GRU, Curriculum), `lstm_scenario.py` "
        "(Sequenzen), `lstm_evaluation.py` (Sweeps, Gradienten-Check, Reduktions-Check), "
        "`lstm_visualization.py` (Plots)."
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Neuronale Netze: vom Perceptron zum Transformer](https://sebastianhanisch.net/konzepte-neuronale-netze.html)."
)
