"""RNN-Referenz (eigenstaendige Kopie aus rnn-demo), LSTM (Hochreiter &
Schmidhuber 1997) und GRU (Cho et al. 2014, Nebenvergleich) mit
Backpropagation Through Time von Hand, Perceptron-Criterion-Verlust
(dieselbe Familie wie Stueck 1/2/4) und SGD/Adam.

Kernidee des LSTM: der Zellzustand c_t hat einen eigenen Rueckwaertspfad
(dc_{t-1} = dc_t * f_t), der NICHT bei jedem Schritt mit einer
Aktivierungsableitung multipliziert wird wie beim RNN-Zustand - solange das
Vergessenstor f_t nahe 1 bleibt, kann der Gradient ueber viele Zeitschritte
nahezu unveraendert zurueckfliessen ("konstanter Fehlerfluss")."""
from dataclasses import dataclass, field

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------- RNN-Referenz (Stueck 4) ----------

class RNN:
    def __init__(self, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = n_hidden
        limit_xh = np.sqrt(6.0 / (1 + H))
        limit_hh = np.sqrt(6.0 / (H + H))
        limit_hy = np.sqrt(6.0 / (H + 1))
        self.H = H
        self.params = {
            "Wxh": rng.uniform(-limit_xh, limit_xh, size=(H, 1)),
            "Whh": rng.uniform(-limit_hh, limit_hh, size=(H, H)),
            "bh": np.zeros(H),
            "Why": rng.uniform(-limit_hy, limit_hy, size=H),
            "by": np.zeros(()),
        }

    def forward(self, seq: np.ndarray):
        T = len(seq)
        h = np.zeros(self.H)
        hs = [h]
        p = self.params
        for t in range(T):
            z = p["Wxh"][:, 0] * seq[t] + p["Whh"] @ h + p["bh"]
            h = np.tanh(z)
            hs.append(h)
        s = float(p["Why"] @ hs[-1] + p["by"])
        return s, hs

    def backward(self, y: float, s: float, hs: list, seq: np.ndarray):
        T = len(seq)
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        if y * s > 0:
            return grads, None
        ds = -y
        grads["Why"] = ds * hs[-1]
        grads["by"] = np.asarray(ds)
        dh_next = ds * p["Why"]
        deltas = []
        for t in range(T, 0, -1):
            delta = dh_next * (1 - hs[t] ** 2)
            deltas.append(delta)
            grads["Wxh"][:, 0] += delta * seq[t - 1]
            grads["Whh"] += np.outer(delta, hs[t - 1])
            grads["bh"] += delta
            dh_next = p["Whh"].T @ delta
        deltas.reverse()
        return grads, deltas

    def predict(self, seq: np.ndarray) -> float:
        s, _ = self.forward(seq)
        return 1.0 if s >= 0 else -1.0


# ---------- LSTM ----------

class LSTM:
    GATES = ("f", "i", "o", "g")

    def __init__(self, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = n_hidden
        lx = np.sqrt(6.0 / (1 + H))
        lh = np.sqrt(6.0 / (H + H))
        lhy = np.sqrt(6.0 / (H + 1))
        self.H = H
        self.params = {}
        for gate in self.GATES:
            self.params[f"W{gate}"] = rng.uniform(-lx, lx, size=(H, 1))
            self.params[f"U{gate}"] = rng.uniform(-lh, lh, size=(H, H))
            self.params[f"b{gate}"] = np.zeros(H)
        self.params["bf"] = np.ones(H)  # Vergessenstor-Bias auf 1 (gaengige Praxis)
        self.params["Why"] = rng.uniform(-lhy, lhy, size=H)
        self.params["by"] = np.zeros(())

    def forward(self, seq: np.ndarray):
        T = len(seq)
        p = self.params
        h = np.zeros(self.H)
        c = np.zeros(self.H)
        cache = {"h": [h], "c": [c], "f": [], "i": [], "o": [], "g": []}
        for t in range(T):
            x = seq[t]
            f = sigmoid(p["Wf"][:, 0] * x + p["Uf"] @ h + p["bf"])
            i = sigmoid(p["Wi"][:, 0] * x + p["Ui"] @ h + p["bi"])
            o = sigmoid(p["Wo"][:, 0] * x + p["Uo"] @ h + p["bo"])
            g = np.tanh(p["Wg"][:, 0] * x + p["Ug"] @ h + p["bg"])
            c = f * c + i * g
            h = o * np.tanh(c)
            cache["f"].append(f)
            cache["i"].append(i)
            cache["o"].append(o)
            cache["g"].append(g)
            cache["c"].append(c)
            cache["h"].append(h)
        s = float(p["Why"] @ cache["h"][-1] + p["by"])
        return s, cache

    def backward(self, y: float, s: float, cache: dict, seq: np.ndarray):
        """Gibt (grads, dc_norms) zurueck; dc_norms[0] = Norm des
        Zellzustand-Gradienten am ERSTEN Zeitschritt, dc_norms[-1] am
        LETZTEN - die direkte Messung der 'Gradienten-Autobahn'."""
        T = len(seq)
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        if y * s > 0:
            return grads, None
        ds = -y
        grads["Why"] = ds * cache["h"][-1]
        grads["by"] = np.asarray(ds)
        dh_next = ds * p["Why"]
        dc_next = np.zeros(self.H)
        dc_norms = []
        for t in range(T, 0, -1):
            x = seq[t - 1]
            h_prev = cache["h"][t - 1]
            c_prev = cache["c"][t - 1]
            f, i, o, g, c = (cache["f"][t - 1], cache["i"][t - 1], cache["o"][t - 1],
                             cache["g"][t - 1], cache["c"][t])
            tanh_c = np.tanh(c)

            dh = dh_next
            do = dh * tanh_c
            dc = dc_next + dh * o * (1 - tanh_c ** 2)
            dc_norms.append(float(np.linalg.norm(dc)))

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc_prev = dc * f

            dz_f = df * f * (1 - f)
            dz_i = di * i * (1 - i)
            dz_o = do * o * (1 - o)
            dz_g = dg * (1 - g ** 2)

            for gate, dz in zip(self.GATES, (dz_f, dz_i, dz_o, dz_g)):
                grads[f"W{gate}"][:, 0] += dz * x
                grads[f"U{gate}"] += np.outer(dz, h_prev)
                grads[f"b{gate}"] += dz

            dh_next = (p["Uf"].T @ dz_f + p["Ui"].T @ dz_i + p["Uo"].T @ dz_o + p["Ug"].T @ dz_g)
            dc_next = dc_prev
        dc_norms.reverse()
        return grads, dc_norms

    def predict(self, seq: np.ndarray) -> float:
        s, _ = self.forward(seq)
        return 1.0 if s >= 0 else -1.0


# ---------- GRU (Cho et al. 2014, Nebenvergleich - kein eigenes Stueck) ----------

class GRU:
    def __init__(self, n_hidden: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        H = n_hidden
        lx = np.sqrt(6.0 / (1 + H))
        lh = np.sqrt(6.0 / (H + H))
        lhy = np.sqrt(6.0 / (H + 1))
        self.H = H
        self.params = {}
        for gate in ("z", "r", "h"):
            self.params[f"W{gate}"] = rng.uniform(-lx, lx, size=(H, 1))
            self.params[f"U{gate}"] = rng.uniform(-lh, lh, size=(H, H))
            self.params[f"b{gate}"] = np.zeros(H)
        self.params["Why"] = rng.uniform(-lhy, lhy, size=H)
        self.params["by"] = np.zeros(())

    def forward(self, seq: np.ndarray):
        T = len(seq)
        p = self.params
        h = np.zeros(self.H)
        cache = {"h": [h], "z": [], "r": [], "n": []}
        for t in range(T):
            x = seq[t]
            z = sigmoid(p["Wz"][:, 0] * x + p["Uz"] @ h + p["bz"])
            r = sigmoid(p["Wr"][:, 0] * x + p["Ur"] @ h + p["br"])
            n = np.tanh(p["Wh"][:, 0] * x + p["Uh"] @ (r * h) + p["bh"])
            h = (1 - z) * h + z * n
            cache["z"].append(z)
            cache["r"].append(r)
            cache["n"].append(n)
            cache["h"].append(h)
        s = float(p["Why"] @ cache["h"][-1] + p["by"])
        return s, cache

    def backward(self, y: float, s: float, cache: dict, seq: np.ndarray):
        T = len(seq)
        p = self.params
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        if y * s > 0:
            return grads, None
        ds = -y
        grads["Why"] = ds * cache["h"][-1]
        grads["by"] = np.asarray(ds)
        dh_next = ds * p["Why"]
        for t in range(T, 0, -1):
            x = seq[t - 1]
            h_prev = cache["h"][t - 1]
            z, r, n = cache["z"][t - 1], cache["r"][t - 1], cache["n"][t - 1]

            dz = dh_next * (n - h_prev)
            dn = dh_next * z
            dh_prev_direct = dh_next * (1 - z)

            dn_pre = dn * (1 - n ** 2)
            grads["Wh"][:, 0] += dn_pre * x
            grads["Uh"] += np.outer(dn_pre, r * h_prev)
            grads["bh"] += dn_pre

            d_r_hprev = p["Uh"].T @ dn_pre
            dr = d_r_hprev * h_prev
            dh_prev_via_r = d_r_hprev * r

            dz_pre = dz * z * (1 - z)
            dr_pre = dr * r * (1 - r)
            grads["Wz"][:, 0] += dz_pre * x
            grads["Uz"] += np.outer(dz_pre, h_prev)
            grads["bz"] += dz_pre
            grads["Wr"][:, 0] += dr_pre * x
            grads["Ur"] += np.outer(dr_pre, h_prev)
            grads["br"] += dr_pre

            dh_next = (dh_prev_direct + dh_prev_via_r + p["Uz"].T @ dz_pre + p["Ur"].T @ dr_pre)
        return grads, True  # kein Gradienten-Autobahn-Tracking noetig (Nebenvergleich)

    def predict(self, seq: np.ndarray) -> float:
        s, _ = self.forward(seq)
        return 1.0 if s >= 0 else -1.0


# ---------- Optimierer (SGD als Standard - Lehre aus Stueck 4: Adam verdeckt
# Gradienten-Effekte) ----------

class SGD:
    def __init__(self, eta: float = 0.1):
        self.eta = eta

    def step(self, params: dict, grads: dict) -> None:
        for key, g in grads.items():
            params[key] -= self.eta * g


class Adam:
    def __init__(self, eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        for key, g in grads.items():
            m = self.beta1 * self.m.get(key, np.zeros_like(g)) + (1 - self.beta1) * g
            v = self.beta2 * self.v.get(key, np.zeros_like(g)) + (1 - self.beta2) * g ** 2
            self.m[key] = m
            self.v[key] = v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params[key] -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class TrainResult:
    errors_per_epoch: list = field(default_factory=list)


def train_epoch(model, optimizer, X: np.ndarray, y: np.ndarray) -> int:
    n_errors = 0
    for seq, label in zip(X, y):
        s, cache = model.forward(seq)
        grads, extra = model.backward(label, s, cache, seq)
        if extra is not None:
            n_errors += 1
            optimizer.step(model.params, grads)
    return n_errors


def train(model, optimizer, X: np.ndarray, y: np.ndarray, epochs: int) -> TrainResult:
    errors_per_epoch = []
    for _ in range(epochs):
        errors_per_epoch.append(train_epoch(model, optimizer, X, y))
    return TrainResult(errors_per_epoch)


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    correct = sum(model.predict(seq) == label for seq, label in zip(X, y))
    return correct / len(y)


def train_curriculum(model, optimizer, make_dataset_fn, stages, n_train: int,
                     epochs_per_stage: int):
    """Trainiert dasselbe Modell nacheinander auf wachsenden Sequenzlaengen
    (Bengio et al. 2009, Curriculum Learning). Jede Stufe muss nur eine KLEINE
    Erweiterung einer bereits funktionierenden Loesung lernen, statt bei der
    Ziel-Sequenzlaenge bei Null anzufangen - das umgeht das Kaltstart-Problem
    (siehe README: beide Architekturen scheitern sonst oft schon bei
    mittlerem T an einer entarteten Konstant-Vorhersage-Loesung)."""
    for T_stage in stages:
        X, y = make_dataset_fn(n_train, T_stage)
        train(model, optimizer, X, y, epochs_per_stage)
    return model
