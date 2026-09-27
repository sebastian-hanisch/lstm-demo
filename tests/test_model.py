import numpy as np

import lstm_evaluation as ev
import lstm_model as m
import lstm_scenario as sc


def test_lstm_forward_returns_scalar_score():
    lstm = m.LSTM(4, seed=0)
    seq = np.zeros(10)
    s, cache = lstm.forward(seq)
    assert isinstance(s, float)


def test_gru_forward_returns_scalar_score():
    gru = m.GRU(4, seed=0)
    seq = np.zeros(10)
    s, cache = gru.forward(seq)
    assert isinstance(s, float)


def test_lstm_backward_returns_none_extra_when_correctly_classified():
    lstm = m.LSTM(4, seed=0)
    lstm.params["Why"][:] = 0.0
    lstm.params["by"] = np.asarray(5.0)
    seq = np.zeros(5)
    s, cache = lstm.forward(seq)
    grads, extra = lstm.backward(1.0, s, cache, seq)
    assert extra is None
    assert np.allclose(grads["Uf"], 0.0)


def test_lstm_gradient_check_below_1e_minus_6():
    """Etwas lockerer als beim RNN (1e-9): die tiefere Ableitungskette durch
    vier Tore haeuft minimal mehr Gleitkomma-Rundung an, gemessen ~1e-7."""
    assert ev.gradient_check() < 1e-6


def test_lstm_cell_gradient_decays_far_slower_than_rnn_hidden_gradient():
    row_rnn = ev.gradient_norms_rnn(T=150)
    row_lstm = ev.gradient_norms_lstm(T=150)
    assert row_rnn["norm_first"] < 1e-8  # praktisch verschwunden
    assert row_lstm["norm_first"] > 1e-6  # noch klar von null verschieden
    assert row_lstm["norm_first"] > row_rnn["norm_first"] * 1000


def test_reduction_check_gates_fixed_is_exact():
    out = ev.reduction_check_gates_fixed()
    assert out["identical"]


def test_train_curriculum_runs_and_produces_a_valid_probability():
    """Nur ein Rauchtest hier: eine einzelne Trainings-Trajektorie (Kaltstart
    oder Curriculum) ist chaotisch seed-abhaengig (siehe
    feedback_ci_platform_robust_tests.md - dieselbe Lehre wie in rnn-demo) -
    die robuste, statistische Aussage steht in test_claim_curriculum_rescues_both
    in test_claims.py (gemittelt ueber mehrere Seeds)."""
    T = 60
    lstm_curr = m.LSTM(6, seed=1)
    opt = m.Adam(0.1)

    def make_stage(n, T_stage):
        ds = sc.make_dataset(n, T_stage, seed=1)
        return ds.X, ds.y

    m.train_curriculum(lstm_curr, opt, make_stage, stages=(10, 30, 60), n_train=30,
                       epochs_per_stage=10)
    X_test, y_test = sc.make_dataset(20, T, seed=1001).X, sc.make_dataset(20, T, seed=1001).y
    acc = m.accuracy(lstm_curr, X_test, y_test)
    assert 0.0 <= acc <= 1.0
