"""Signal-in-Rauschen-Sequenz (Vehikel C, eigenstaendige Kopie aus rnn-demo -
kein Cross-Repo-Import, jedes Demo-Repo bleibt fuer sich deploybar): an einer
festen, fruehen Position steht das Signal (+-1 plus Rauschen), der Rest ist
reines Rauschen. Aufgabe am Ende der Sequenz: das Vorzeichen des Signals
reproduzieren (Hochreiter & Schmidhuber 1997, Adding Problem/Temporal Order)."""
from dataclasses import dataclass

import numpy as np

import lstm_constants as C


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    T: int


def make_sequence(rng: np.random.Generator, T: int, signal_pos: int = C.SIGNAL_POS,
                  noise_std: float = C.NOISE_DEFAULT):
    label = float(rng.choice([-1.0, 1.0]))
    seq = rng.normal(0.0, noise_std, size=T)
    seq[signal_pos] += label
    return seq, label


def make_dataset(n: int, T: int, seed: int, signal_pos: int = C.SIGNAL_POS,
                 noise_std: float = C.NOISE_DEFAULT) -> Dataset:
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for _ in range(n):
        seq, label = make_sequence(rng, T, signal_pos, noise_std)
        Xs.append(seq)
        ys.append(label)
    return Dataset(np.array(Xs), np.array(ys), T)
