"""Unabhängige Orakel für LSTM/GRU/RNN-BPTT:

* Parametergradienten aller drei Architekturen gegen den Complex-Step-Gradienten einer
  eigenen Vorwärtsrechnung (exakt bis auf Maschinengenauigkeit),
* die Zellzustand-Gradientennormen ||dL/dc_t|| des LSTM (README-Tabelle, bis hinunter zu
  3e-23 bei T=1200) gegen Complex-Step-Ableitungen nach einem auf c_t addierten Offset,
* `analyse` (LSTM) gegen einen eigenen Trainer in verschmolzener Matrixform.
"""
import numpy as np

import lstm_evaluation as ev
import lstm_model as M
import lstm_scenario as sc


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _lstm_score(p, seq, c_off=None):
    H = p["bf"].shape[0]
    h = np.zeros(H, dtype=complex)
    c = np.zeros(H, dtype=complex)
    for t, x in enumerate(seq):
        pre = {g: p["W" + g][:, 0] * x + p["U" + g] @ h + p["b" + g] for g in "fiog"}
        c = _sig(pre["f"]) * c + _sig(pre["i"]) * np.tanh(pre["g"])
        if c_off is not None:
            c = c + c_off[t]
        h = _sig(pre["o"]) * np.tanh(c)
    return p["Why"] @ h + p["by"]


def _gru_score(p, seq):
    h = np.zeros(p["bz"].shape[0], dtype=complex)
    for x in seq:
        z = _sig(p["Wz"][:, 0] * x + p["Uz"] @ h + p["bz"])
        r = _sig(p["Wr"][:, 0] * x + p["Ur"] @ h + p["br"])
        n = np.tanh(p["Wh"][:, 0] * x + p["Uh"] @ (r * h) + p["bh"])
        h = (1 - z) * h + z * n
    return p["Why"] @ h + p["by"]


def _rnn_score(p, seq):
    h = np.zeros(p["bh"].shape[0], dtype=complex)
    for x in seq:
        h = np.tanh(p["Wxh"][:, 0] * x + p["Whh"] @ h + p["bh"])
    return p["Why"] @ h + p["by"]


def _cs_grads(score_fn, p, seq, y):
    out = {}
    for key, v in p.items():
        v = np.asarray(v, dtype=float)
        g = np.zeros(v.shape)
        for idx in np.ndindex(*v.shape):
            q = {k: np.asarray(vv, dtype=complex).copy() for k, vv in p.items()}
            q[key][idx] += 1e-30j
            g[idx] = (-y * score_fn(q, seq)).imag / 1e-30
        out[key] = g
    return out


def _cs_dc_norm(p, seq, y, t):
    T, H = len(seq), p["bf"].shape[0]
    pc = {k: np.asarray(v, dtype=complex) for k, v in p.items()}
    sq = 0.0
    for j in range(H):
        off = np.zeros((T, H), dtype=complex)
        off[t, j] = 1e-30j
        sq += ((-y * _lstm_score(pc, seq, off)).imag / 1e-30) ** 2
    return float(np.sqrt(sq))


def test_backward_of_lstm_gru_rnn_matches_complex_step():
    rng = np.random.default_rng(21)
    for cls, score_fn in ((M.LSTM, _lstm_score), (M.GRU, _gru_score), (M.RNN, _rnn_score)):
        for trial in range(25):
            H, T = int(rng.integers(1, 6)), int(rng.integers(1, 10))
            net = cls(H, seed=trial)
            for k in net.params:
                net.params[k] = rng.normal(0, 0.8, size=net.params[k].shape)
            seq = rng.normal(0, 1, size=T)
            s, cache = net.forward(seq)
            assert abs(s - score_fn(net.params, seq).real) < 1e-12
            y = -1.0 if s >= 0 else 1.0
            grads, extra = net.backward(y, s, cache, seq)
            ref = _cs_grads(score_fn, net.params, seq, y)
            for k in grads:
                np.testing.assert_allclose(grads[k], ref[k], atol=1e-9)
            assert net.backward(-y, s, cache, seq)[1] is None


def test_lstm_cell_gradient_norms_match_complex_step_including_tiny_values():
    for T in (10, 80, 600, 1200):
        d = ev.gradient_norms_lstm(T)
        net = M.LSTM(8, seed=0)
        seq, _ = sc.make_sequence(np.random.default_rng(5), T)
        s = float(_lstm_score(net.params, seq).real)
        y = -np.sign(s) if s != 0 else 1.0
        np.testing.assert_allclose(d["norm_first"], _cs_dc_norm(net.params, seq, y, 0), rtol=1e-8)
        np.testing.assert_allclose(d["norm_last"], _cs_dc_norm(net.params, seq, y, T - 1), rtol=1e-8)


def _init_lstm(H, seed):
    r = np.random.default_rng(seed)
    lx, lh, lhy = np.sqrt(6.0 / (1 + H)), np.sqrt(6.0 / (2 * H)), np.sqrt(6.0 / (H + 1))
    p = {}
    for g in "fiog":
        p["W" + g] = r.uniform(-lx, lx, size=(H, 1))
        p["U" + g] = r.uniform(-lh, lh, size=(H, H))
        p["b" + g] = np.zeros(H)
    p["bf"] = np.ones(H)
    p["Why"] = r.uniform(-lhy, lhy, size=H)
    p["by"] = np.zeros(())
    return p


def _own_epoch(p, X, Y, eta):
    H = p["bf"].shape[0]
    Wx = np.stack([p["W" + g][:, 0] for g in "fiog"])
    U = np.stack([p["U" + g] for g in "fiog"])
    B = np.stack([p["b" + g] for g in "fiog"])
    errs = 0
    for seq, y in zip(X, Y):
        T = len(seq)
        hs, cs, gates = np.zeros((T + 1, H)), np.zeros((T + 1, H)), np.zeros((T, 4, H))
        for t in range(T):
            pre = Wx * seq[t] + np.einsum("gij,j->gi", U, hs[t]) + B
            gates[t] = (_sig(pre[0]), _sig(pre[1]), _sig(pre[2]), np.tanh(pre[3]))
            f, i, o, g = gates[t]
            cs[t + 1] = f * cs[t] + i * g
            hs[t + 1] = o * np.tanh(cs[t + 1])
        if y * (p["Why"] @ hs[T] + p["by"]) > 0:
            continue
        errs += 1
        dWx, dU, dB = np.zeros_like(Wx), np.zeros_like(U), np.zeros_like(B)
        dh, dc = -y * p["Why"], np.zeros(H)
        for t in range(T, 0, -1):
            f, i, o, g = gates[t - 1]
            tc = np.tanh(cs[t])
            dc = dc + dh * o * (1 - tc ** 2)
            dpre = np.stack([dc * cs[t - 1] * f * (1 - f), dc * g * i * (1 - i),
                             dh * tc * o * (1 - o), dc * i * (1 - g ** 2)])
            dWx += dpre * seq[t - 1]
            dU += np.einsum("gi,j->gij", dpre, hs[t - 1])
            dB += dpre
            dh = np.einsum("gij,gi->j", U, dpre)
            dc = dc * f
        p["Why"] = p["Why"] - eta * (-y * hs[T])
        p["by"] = p["by"] - eta * (-y)
        Wx, U, B = Wx - eta * dWx, U - eta * dU, B - eta * dB
        for k, g in enumerate("fiog"):
            p["W" + g], p["U" + g], p["b" + g] = Wx[k][:, None], U[k], B[k]
    return errs


def test_analyse_lstm_matches_independent_trainer():
    rng = np.random.default_rng(5)
    for _ in range(12):
        st = ev.Settings(T=int(rng.choice([2, 3, 5, 8, 10])), hidden=int(rng.integers(2, 8)),
                         n_train=int(rng.integers(20, 35)), n_test=20,
                         noise=float(rng.choice([0.0, 0.3, 0.6])),
                         eta=float(rng.choice([0.02, 0.1, 0.3])), epochs=int(rng.integers(3, 8)),
                         seed=int(rng.integers(0, 1000)))
        out = ev.analyse(st)
        tr = sc.make_dataset(st.n_train, st.T, st.seed, noise_std=st.noise)
        te = sc.make_dataset(st.n_test, st.T, st.seed + 1000, noise_std=st.noise)
        p = _init_lstm(st.hidden, st.seed)
        errs = [_own_epoch(p, tr.X, tr.y, st.eta) for _ in range(st.epochs)]
        assert errs == out["lstm"]["result"].errors_per_epoch
        acc = np.mean([(1.0 if _lstm_score(p, s_).real >= 0 else -1.0) == y_
                       for s_, y_ in zip(te.X, te.y)])
        assert abs(acc - out["lstm"]["test_accuracy"]) < 1e-12


def test_adam_matches_textbook_formula():
    """Adam (Curriculum-Vergleich) gegen Kingma & Ba 2015, Alg. 1 mit Bias-Korrektur."""
    rng = np.random.default_rng(8)
    for _ in range(30):
        eta = float(rng.uniform(0.001, 0.3))
        p0 = {"a": rng.normal(size=(3, 3)), "b": rng.normal(size=4)}
        seq = [{k: rng.normal(size=v.shape) * 3 for k, v in p0.items()}
               for _ in range(int(rng.integers(1, 25)))]
        params = {k: v.copy() for k, v in p0.items()}
        opt = M.Adam(eta)
        ref = {k: v.copy() for k, v in p0.items()}
        mk = {k: np.zeros_like(v) for k, v in p0.items()}
        vk = {k: np.zeros_like(v) for k, v in p0.items()}
        for t, g in enumerate(seq, 1):
            opt.step(params, {k: v.copy() for k, v in g.items()})
            for k in ref:
                mk[k] = 0.9 * mk[k] + 0.1 * g[k]
                vk[k] = 0.999 * vk[k] + 0.001 * g[k] ** 2
                ref[k] = ref[k] - eta * (mk[k] / (1 - 0.9 ** t)) / (
                    np.sqrt(vk[k] / (1 - 0.999 ** t)) + 1e-8)
        for k in ref:
            np.testing.assert_allclose(params[k], ref[k], atol=1e-12)
