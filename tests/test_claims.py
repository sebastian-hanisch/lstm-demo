"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen
neu berechnet. Die teuren Sweeps werden NUR EINMAL pro Testlauf berechnet
(Modul-Fixtures) und mit Toleranzband statt exakter Gleichheit geprüft - Lehre
aus rnn-demo (feedback_ci_platform_robust_tests.md): eine einzelne
Trainings-Trajektorie ist über viele Epochen chaotisch empfindlich gegenüber
Gleitkomma-Rundung (BLAS/Aktivierung unterscheidet sich zwischen Windows und
der Linux-CI), daher NIE exakte Einzel-Prozentzahlen pinnen."""
import pytest

import lstm_constants as C
import lstm_evaluation as ev
import lstm_model as m


@pytest.fixture(scope="module")
def t_sweep_rows():
    return ev.t_sweep()


@pytest.fixture(scope="module")
def gradient_norm_rows():
    return ev.gradient_norm_sweep(values=C.GRADIENT_SWEEP_VALUES)


@pytest.fixture(scope="module")
def gru_rows():
    return ev.gru_cold_sweep()


@pytest.fixture(scope="module")
def curriculum_rates():
    return {"rnn": ev.curriculum_success_rate(m.RNN), "lstm": ev.curriculum_success_rate(m.LSTM)}


def test_claim_t_sweep_short_T_always_succeeds(t_sweep_rows):
    row10 = [r for r in t_sweep_rows if r["T"] == 10][0]
    assert row10["rate_rnn"] == 1.0
    assert row10["rate_lstm"] == 1.0


def test_claim_t_sweep_lstm_does_not_outperform_rnn_cold_start(t_sweep_rows):
    """Der zentrale, ueberraschende Befund (Plan-Korrektur): entgegen der
    naiven Erwartung ist das LSTM beim Kaltstart-Training NICHT zuverlaessiger
    als das RNN - ab T=80 sogar deutlich unzuverlaessiger. Toleranzband
    (+-2 von 6 Initialisierungen), da einzelne Seeds chaotisch kippen koennen."""
    tolerance = 2 / 6
    expected = {10: (1.0, 1.0), 40: (2 / 3, 1 / 6), 80: (1 / 6, 0.0), 150: (1 / 3, 0.0)}
    for row in t_sweep_rows:
        exp_rnn, exp_lstm = expected[row["T"]]
        assert abs(row["rate_rnn"] - exp_rnn) <= tolerance + 1e-9
        assert abs(row["rate_lstm"] - exp_lstm) <= tolerance + 1e-9
    # der Kernbefund muss robust stehen: LSTM ist bei grossem T nicht besser als RNN
    by_T = {r["T"]: r for r in t_sweep_rows}
    assert by_T[80]["rate_lstm"] <= by_T[80]["rate_rnn"] + tolerance
    assert by_T[150]["rate_lstm"] <= by_T[150]["rate_rnn"] + tolerance


def test_claim_gradient_norm_rnn_underflows_lstm_does_not(gradient_norm_rows):
    by_T = {r["T"]: r for r in gradient_norm_rows}
    assert by_T[1200]["rnn"]["norm_first"] < 1e-50  # praktisch numerisch verschwunden
    assert by_T[1200]["lstm"]["norm_first"] > 1e-25  # noch eine normale, darstellbare Zahl
    # der letzte Zeitschritt bleibt bei beiden in einer aehnlichen Groessenordnung
    for T in C.GRADIENT_SWEEP_VALUES:
        assert 0.5 < by_T[T]["rnn"]["norm_last"] < 1.5
        assert 0.3 < by_T[T]["lstm"]["norm_last"] < 0.8


def test_claim_gradient_check_below_1e_minus_6():
    assert ev.gradient_check() < 1e-6


def test_claim_reduction_check_gates_fixed_is_exact():
    out = ev.reduction_check_gates_fixed()
    assert out["identical"]


def test_claim_gru_is_not_more_reliable_than_rnn_lstm_at_this_scale(gru_rows):
    """Echter, nicht erwarteter Nebenbefund: GRU ist bei dieser kleinen
    Skala/kurzen Trainingsbudget sogar UNZUVERLAESSIGER als RNN/LSTM."""
    by_T = {r["T"]: r["rate_gru"] for r in gru_rows}
    assert by_T[10] <= 0.5  # deutlich unter RNN/LSTMs 100% bei T=10
    assert by_T[40] < 0.3
    assert by_T[80] < 0.3


def test_claim_curriculum_rescues_both_architectures(curriculum_rates):
    """Die Aufloesung des Raetsels: Curriculum Learning hebt die Erfolgsquote
    fuer BEIDE Architekturen bei T=150 stark an (verglichen mit dem
    Kaltstart-Sweep) - kein exklusiver LSTM-Vorteil."""
    assert curriculum_rates["rnn"] >= 0.4
    assert curriculum_rates["lstm"] >= 0.4
