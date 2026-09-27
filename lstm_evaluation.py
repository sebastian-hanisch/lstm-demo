"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, T-Sweep (RNN vs.
LSTM, GRU-Nebenvergleich), Gradientennorm-Sweep (Zellzustand), Gradienten-
Check, Korrektheits-Kette (Tore fest auf 1)."""
from dataclasses import dataclass

import numpy as np

import lstm_constants as C
import lstm_model as m
import lstm_scenario as sc


@dataclass(frozen=True)
class Settings:
    T: int
    hidden: int
    n_train: int
    n_test: int
    noise: float
    eta: float
    epochs: int
    seed: int


def _train_and_eval(model_cls, settings: Settings) -> float:
    train_ds = sc.make_dataset(settings.n_train, settings.T, settings.seed, noise_std=settings.noise)
    test_ds = sc.make_dataset(settings.n_test, settings.T, settings.seed + 1000, noise_std=settings.noise)
    model = model_cls(settings.hidden, seed=settings.seed)
    optimizer = m.SGD(settings.eta)
    m.train(model, optimizer, train_ds.X, train_ds.y, settings.epochs)
    return m.accuracy(model, test_ds.X, test_ds.y)


def analyse(settings: Settings) -> dict:
    train_ds = sc.make_dataset(settings.n_train, settings.T, settings.seed, noise_std=settings.noise)
    test_ds = sc.make_dataset(settings.n_test, settings.T, settings.seed + 1000, noise_std=settings.noise)
    results = {}
    for name, cls in (("rnn", m.RNN), ("lstm", m.LSTM)):
        model = cls(settings.hidden, seed=settings.seed)
        optimizer = m.SGD(settings.eta)
        result = m.train(model, optimizer, train_ds.X, train_ds.y, settings.epochs)
        results[name] = {"model": model, "result": result,
                          "test_accuracy": m.accuracy(model, test_ds.X, test_ds.y)}
    return {"train_ds": train_ds, "test_ds": test_ds, **results}


def t_sweep(
    values=C.T_SWEEP_VALUES, n_inits=C.T_SWEEP_INITS, n_train=C.N_TRAIN_DEFAULT,
    n_test=C.N_TEST_DEFAULT, noise=C.NOISE_DEFAULT, hidden=C.HIDDEN_DEFAULT,
    eta=C.ETA_DEFAULT, epochs=C.EPOCHS_DEFAULT,
) -> list:
    """Kaltstart-Erfolgsquote (>=90% Testgenauigkeit, SGD, KEIN Curriculum)
    ueber n_inits Zufalls-Inits, RNN vs. LSTM, fuer jedes T.

    Echter, ueberraschender Befund (Plan-Korrektur, siehe README): entgegen
    der urspruenglichen Erwartung scheitert das LSTM hier NICHT seltener als
    das RNN - beide kollabieren bei mittlerem/grossem T oft auf eine entartete
    Konstant-Vorhersage (siehe reduction_check_gates_fixed-Nachbarfunktionen
    und den Curriculum-Vergleich unten fuer die Aufloesung)."""
    rows = []
    for T in values:
        row = {"T": T}
        for name, cls in (("rnn", m.RNN), ("lstm", m.LSTM)):
            successes = 0
            for seed in range(n_inits):
                settings = Settings(T=T, hidden=hidden, n_train=n_train, n_test=n_test,
                                   noise=noise, eta=eta, epochs=epochs, seed=seed)
                acc = _train_and_eval(cls, settings)
                if acc >= 0.9:
                    successes += 1
            row[f"rate_{name}"] = successes / n_inits
        rows.append(row)
    return rows


def gru_cold_sweep(
    values=C.GRU_SWEEP_VALUES, n_inits=C.GRU_SWEEP_INITS, n_train=C.N_TRAIN_DEFAULT,
    n_test=C.N_TEST_DEFAULT, noise=C.NOISE_DEFAULT, hidden=C.HIDDEN_DEFAULT,
    eta=C.ETA_DEFAULT, epochs=C.EPOCHS_DEFAULT,
) -> list:
    """GRU-Nebenvergleich (Cho et al. 2014, kein eigenes Stueck): dieselbe
    Kaltstart-Erfolgsquote wie t_sweep, nur fuer GRU. Bei dieser kleinen
    Skala/kurzen Trainingsbudget zeigt sich KEIN klarer Vorteil - im Gegenteil,
    GRU ist hier sogar unzuverlässiger als RNN/LSTM (gemessen, nicht erwartet)."""
    rows = []
    for T in values:
        successes = 0
        for seed in range(n_inits):
            settings = Settings(T=T, hidden=hidden, n_train=n_train, n_test=n_test,
                               noise=noise, eta=eta, epochs=epochs, seed=seed)
            acc = _train_and_eval(m.GRU, settings)
            if acc >= 0.9:
                successes += 1
        rows.append({"T": T, "rate_gru": successes / n_inits})
    return rows


def curriculum_success_rate(
    model_cls, T: int = C.CURRICULUM_T, stages=C.CURRICULUM_STAGES,
    n_inits: int = C.CURRICULUM_INITS, n_train: int = C.N_TRAIN_DEFAULT,
    n_test: int = C.N_TEST_DEFAULT, noise: float = C.NOISE_DEFAULT,
    hidden: int = C.HIDDEN_DEFAULT, eta: float = C.ETA_DEFAULT,
    epochs_per_stage: int = C.CURRICULUM_EPOCHS_PER_STAGE,
) -> float:
    """Curriculum Learning (Bengio et al. 2009): dasselbe Modell nacheinander
    auf wachsenden T trainiert, statt direkt bei T=150 zu starten. Loest das
    Kaltstart-Problem aus t_sweep() fuer BEIDE Architekturen (kein exklusiver
    LSTM-Vorteil in der Erfolgsquote, sobald beide fair mit Curriculum
    trainiert werden - siehe README)."""
    successes = 0
    for seed in range(n_inits):
        model = model_cls(hidden, seed=seed)
        optimizer = m.Adam(eta)

        def make_stage_dataset(n, T_stage, _seed=seed):
            ds = sc.make_dataset(n, T_stage, _seed, noise_std=noise)
            return ds.X, ds.y

        m.train_curriculum(model, optimizer, make_stage_dataset, stages, n_train, epochs_per_stage)
        test_ds = sc.make_dataset(n_test, T, seed + 1000, noise_std=noise)
        if m.accuracy(model, test_ds.X, test_ds.y) >= 0.9:
            successes += 1
    return successes / n_inits


def gradient_norms_lstm(T: int, hidden: int = C.HIDDEN_DEFAULT, seed: int = 5) -> dict:
    """Norm des Zellzustand-Gradienten (dc) am ersten vs. letzten Zeitschritt -
    die 'Gradienten-Autobahn' des LSTM."""
    lstm = m.LSTM(hidden, seed=0)
    rng = np.random.default_rng(seed)
    seq, _ = sc.make_sequence(rng, T)
    s, cache = lstm.forward(seq)
    y_forced = -np.sign(s) if s != 0 else 1.0
    grads, dc_norms = lstm.backward(y_forced, s, cache, seq)
    return {"T": T, "norm_first": dc_norms[0], "norm_last": dc_norms[-1]}


def gradient_norms_rnn(T: int, hidden: int = C.HIDDEN_DEFAULT, seed: int = 5) -> dict:
    rnn = m.RNN(hidden, seed=0)
    rng = np.random.default_rng(seed)
    seq, _ = sc.make_sequence(rng, T)
    s, hs = rnn.forward(seq)
    y_forced = -np.sign(s) if s != 0 else 1.0
    grads, deltas = rnn.backward(y_forced, s, hs, seq)
    return {"T": T, "norm_first": float(np.linalg.norm(deltas[0])),
            "norm_last": float(np.linalg.norm(deltas[-1]))}


def gradient_norm_sweep(values=C.T_SWEEP_VALUES, hidden: int = C.HIDDEN_DEFAULT) -> list:
    return [{"T": T, "rnn": gradient_norms_rnn(T, hidden), "lstm": gradient_norms_lstm(T, hidden)}
            for T in values]


def gradient_check(T: int = 6, hidden: int = 4, seed: int = 2, eps: float = 1e-5) -> float:
    """BPTT-Gradient (LSTM, alle 4 Tore) gegen finite Differenzen. Etwas
    lockerer als beim RNN (1e-9): die tiefere Ableitungskette durch vier
    Tore haeuft minimal mehr Gleitkomma-Rundung an, gemessen ~1e-7."""
    lstm = m.LSTM(hidden, seed=seed)
    rng = np.random.default_rng(1)
    seq, _ = sc.make_sequence(rng, T)
    s0, cache = lstm.forward(seq)
    y = -np.sign(s0) if s0 != 0 else 1.0
    grads, _ = lstm.backward(y, s0, cache, seq)

    def loss() -> float:
        s, _ = lstm.forward(seq)
        return max(0.0, -y * s)

    max_rel_err = 0.0
    for key, grad in grads.items():
        param = lstm.params[key]
        flat_param = param.reshape(-1)
        flat_grad = grad.reshape(-1)
        for idx in range(flat_grad.size):
            orig = flat_param[idx]
            flat_param[idx] = orig + eps
            l_plus = loss()
            flat_param[idx] = orig - eps
            l_minus = loss()
            flat_param[idx] = orig
            numeric = (l_plus - l_minus) / (2 * eps)
            analytic = float(flat_grad[idx])
            denom = max(abs(numeric), abs(analytic), 1e-12)
            max_rel_err = max(max_rel_err, abs(numeric - analytic) / denom)
    return max_rel_err


def reduction_check_gates_fixed(hidden: int = 4, seed: int = 0, T: int = 5) -> dict:
    """Vergessenstor UND Eingabe-/Ausgabetor fest auf 1: c_t=c_{t-1}+g_t,
    h_t=tanh(c_t) - ein RNN mit Restverbindung (Kontrollzustand), kein
    gewoehnliches Elman-RNN. Exakter Zahlenabgleich gegen eine unabhaengig
    nachgebaute Berechnung."""
    lstm = m.LSTM(hidden, seed=seed)
    H = hidden
    # Tore fest auf 1: W=0, U=0, b=grosse Zahl (sigmoid(20)~1)
    for gate in ("f", "i", "o"):
        lstm.params[f"W{gate}"][:] = 0.0
        lstm.params[f"U{gate}"][:] = 0.0
        lstm.params[f"b{gate}"][:] = 20.0
    rng = np.random.default_rng(3)
    seq = rng.normal(0, 1, size=T)
    s_lstm, cache = lstm.forward(seq)

    # Unabhaengig nachgebaut: c_t = c_{t-1} + g_t, h_t = tanh(c_t)
    h = np.zeros(H)
    c = np.zeros(H)
    for t in range(T):
        x = seq[t]
        g = np.tanh(lstm.params["Wg"][:, 0] * x + lstm.params["Ug"] @ h + lstm.params["bg"])
        c = c + g
        h = np.tanh(c)
    s_manual = float(lstm.params["Why"] @ h + lstm.params["by"])
    return {"s_lstm": s_lstm, "s_manual": s_manual, "identical": bool(np.isclose(s_lstm, s_manual))}
