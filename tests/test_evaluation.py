"""Korrektheits-/Verhaltenstests mit BILLIGEN Parametern (nicht die teuren,
offiziellen Sweep-Werte - die stehen mit Toleranzband in test_claims.py und
werden dort nur EINMAL pro Sweep berechnet, um die Gesamtlaufzeit zu
begrenzen)."""
import lstm_evaluation as ev
import lstm_model as m


def test_t_sweep_short_T_is_easy_for_both():
    rows = ev.t_sweep(values=(10,), n_inits=4)
    assert rows[0]["rate_rnn"] == 1.0
    assert rows[0]["rate_lstm"] == 1.0


def test_gradient_norm_sweep_rnn_vanishes_faster_than_lstm():
    rows = ev.gradient_norm_sweep(values=(10, 300))
    rnn_by_T = {r["T"]: r["rnn"]["norm_first"] for r in rows}
    lstm_by_T = {r["T"]: r["lstm"]["norm_first"] for r in rows}
    assert rnn_by_T[300] < 1e-15
    assert lstm_by_T[300] > 1e-8
    assert rnn_by_T[10] > rnn_by_T[300]
    assert lstm_by_T[10] > lstm_by_T[300]


def test_gru_cold_sweep_runs_and_returns_valid_rates():
    rows = ev.gru_cold_sweep(values=(10,), n_inits=4)
    assert 0.0 <= rows[0]["rate_gru"] <= 1.0


def test_curriculum_success_rate_runs_and_returns_valid_rate():
    rate = ev.curriculum_success_rate(m.RNN, T=40, stages=(10, 20, 40), n_inits=3,
                                      epochs_per_stage=8)
    assert 0.0 <= rate <= 1.0
