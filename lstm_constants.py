"""Regler-Grenzen, feste Annahmen, gemessene Werte und Presets."""

DEFAULT_SEED = 0
SIGNAL_POS = 0

T_MIN, T_MAX, T_DEFAULT = 2, 1200, 80
HIDDEN_MIN, HIDDEN_MAX, HIDDEN_DEFAULT = 2, 16, 8
N_TRAIN_MIN, N_TRAIN_MAX, N_TRAIN_DEFAULT = 20, 120, 60
N_TEST_MIN, N_TEST_MAX, N_TEST_DEFAULT = 20, 100, 40
NOISE_MIN, NOISE_MAX, NOISE_DEFAULT = 0.0, 0.6, 0.3
ETA_MIN, ETA_MAX, ETA_DEFAULT = 0.02, 0.3, 0.1
EPOCHS_MIN, EPOCHS_MAX, EPOCHS_DEFAULT = 10, 100, 30

# Kaltstart-T-Sweep fuer den Hauptabschnitt (Erfolgsquote RNN vs. LSTM, SGD,
# kein Curriculum) - bewusst klein gehalten (n_inits=6), da bereits ~2 Minuten
# Rechenzeit; siehe feedback_expensive_sweep_needs_visible_spinner.md
T_SWEEP_VALUES = (10, 40, 80, 150)
T_SWEEP_INITS = 6

# Gradientennorm-Sweep (billig, keine Spinner noetig) - bewusst bis T=1200,
# um die Groessenordnungs-Kluft (RNN unterlaeuft float64, LSTM nicht) zu zeigen
GRADIENT_SWEEP_VALUES = (10, 40, 80, 150, 300, 600, 1200)

# Curriculum-Vergleich (on-demand per Knopf, nicht eager - zusaetzliche Kosten
# oben auf den bereits teuren Kaltstart-Sweep)
CURRICULUM_T = 150
CURRICULUM_STAGES = (10, 20, 40, 80, 150)
CURRICULUM_INITS = 10
CURRICULUM_EPOCHS_PER_STAGE = 15

# GRU-Nebenvergleich (kein eigenes Stueck, nur eine gemessene Randnotiz)
GRU_SWEEP_VALUES = (10, 40, 80)
GRU_SWEEP_INITS = 10

PRESETS = {
    "kurz": dict(
        label="Kurze Sequenz — beide gelingen",
        T=10, hidden=8, n_train=60, n_test=40, noise=0.3, eta=0.1, epochs=30, seed=0,
        help="Bei kleinem T lernen RNN und LSTM beide zuverlässig - noch kein Unterschied "
             "sichtbar.",
    ),
    "lang_kaltstart": dict(
        label="T=150, Kaltstart",
        T=150, hidden=8, n_train=60, n_test=40, noise=0.3, eta=0.1, epochs=30, seed=0,
        help="Direkt bei T=150 trainiert: überraschend scheitern hier BEIDE Architekturen "
             "meist - der nicht-verschwindende Gradient allein reicht nicht.",
    ),
}
