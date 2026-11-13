import json
import math
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

C0, C1 = -9.0, 6.0
LO, HI = -10.0, 10.0

X_RAW = np.array([[-9.0, -9.0],
                  [-9.0,  6.0],
                  [ 6.0, -9.0],
                  [ 6.0,  6.0]])
T_RAW = np.array([-9.0, 6.0, 6.0, -9.0])


def normalize(x):
    """Нормировка входов по границам диапазона [-10; 10] -> [0; 1] (как в ЛР №1 и №2)."""
    return (np.asarray(x, dtype=float) - LO) / (HI - LO)


def to_binary(y_real):
    """Нормализация целей для BCE: c0 -> 0, c1 -> 1 (ЛР №2)."""
    return (np.asarray(y_real, dtype=float) - C0) / (C1 - C0)


def from_binary(y_hat):
    """Выход сети ŷ из (0; 1) -> исходная шкала [c0; c1]."""
    return C0 + np.asarray(y_hat, dtype=float) * (C1 - C0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid_from_output(y):
    """Производная сигмоиды через её выход y = sigmoid(x)."""
    return y * (1.0 - y)


# НОВОЕ (ЛР №3): ReLU и её производная (в точке 0 производная принимается равной 0)
def relu(x):
    return np.maximum(x, 0.0)


def drelu(x):
    return (x > 0.0).astype(float)


LEAKY_K = 0.01


def leaky_relu(x, k=LEAKY_K):
    """LeakyReLU - используется только в дополнительном исследовании (--extra) для сравнения."""
    return np.where(x > 0.0, x, k * x)


def dleaky_relu(x, k=LEAKY_K):
    return np.where(x > 0.0, 1.0, k)


ACTIVATIONS = {
    "sigmoid": (sigmoid, lambda z, h: dsigmoid_from_output(h)),
    "relu": (relu, lambda z, h: drelu(z)),
    "leaky_relu": (leaky_relu, lambda z, h: dleaky_relu(z)),
}

X = normalize(X_RAW)
T = to_binary(T_RAW)

LR = 0.5
INIT = 0.5                    # веса и пороги из U[-INIT; INIT]
MAX_EPOCHS = 50000
EE_MSE = 0.001                # порог MSE из ЛР №1


def equivalent_ee_bce(ee_mse=EE_MSE, n=4):
    """Порог Ee для BCE, эквивалентный по точности порогу MSE = 0.001 (ЛР №2): Ee = 0.0852."""
    eps = (HI - LO) * math.sqrt(ee_mse / n) / (C1 - C0)
    return n * (-math.log(1.0 - eps))


EE = equivalent_ee_bce()

CONFIGS = {
    "A": {"name": "А: Сигмоида", "activation": "sigmoid"},
    "B": {"name": "В: ReLU", "activation": "relu"},
}


def bce_value(o, t):
    o = np.clip(o, 1e-12, 1.0 - 1e-12)          # защита от log(0)
    return float(-np.sum(t * np.log(o) + (1.0 - t) * np.log(1.0 - o)))


class MLP221:
    def __init__(self, activation="sigmoid", lr=LR, seed=None, init=INIT):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-init, init, (2, 2))
        self.b1 = rng.uniform(-init, init, (2,))
        self.W2 = rng.uniform(-init, init, (2, 1))
        self.b2 = rng.uniform(-init, init, (1,))
        self.lr = lr
        self.activation = activation             # НОВОЕ: "sigmoid" (ЛР №2) или "relu"
        self.act, self.dact = ACTIVATIONS[activation]

    def forward(self, x):
        self.h_in = x @ self.W1 + self.b1
        self.h_out = self.act(self.h_in)
        self.o_in = self.h_out @ self.W2 + self.b2
        self.o_out = sigmoid(self.o_in)          # выходной слой - всегда сигмоида
        return self.o_out

    def gradients(self, x, t):
        """Дельты слоёв для одного примера (BCE + сигмоида на выходе)."""
        o = self.forward(x)
        delta_out = t - o                                                   # BCE + сигмоида (ЛР №2)
        delta_hidden = (delta_out @ self.W2.T) * self.dact(self.h_in, self.h_out)   # НОВОЕ: производная ReLU
        return o, delta_out, delta_hidden

    def train_step(self, x, t):
        o, delta_out, delta_hidden = self.gradients(x, t)
        self.W2 += self.lr * np.outer(self.h_out, delta_out)
        self.b2 += self.lr * delta_out
        self.W1 += self.lr * np.outer(x, delta_hidden)
        self.b1 += self.lr * delta_hidden
        return bce_value(o, t)

    def predict(self, x):
        return float(self.forward(np.asarray(x, dtype=float)).item())

    def hidden_sums(self, x):
        return np.asarray(x, dtype=float) @ self.W1 + self.b1

    def active_points(self):
        """Сколько обучающих примеров активируют каждый скрытый нейрон (z > 0)."""
        return (self.hidden_sums(X) > 0.0).sum(axis=0)

    def params(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])


def train_loop(model, X, T, Ee, max_epochs, rng):
    """Онлайн-обучение: Es накапливается по примерам внутри эпохи (как в ЛР №1 и №2)."""
    n = len(X)
    Es = float("inf")
    history = []
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(n)
        Es = 0.0
        for i in order:
            Es += model.train_step(X[i], T[i:i + 1])
        history.append(Es)
        if Es <= Ee:
            return epoch, Es, history
    return max_epochs, Es, history


def classify(y_real):
    return C0 if abs(y_real - C0) <= abs(y_real - C1) else C1


def evaluate(model):
    """Выходы ŷ и y в шкале [c0; c1], accuracy и средняя абсолютная ошибка (MAE) в шкале [c0; c1]."""
    outputs_hat = [model.predict(x) for x in X]
    outputs_real = [float(from_binary(v)) for v in outputs_hat]
    accuracy = float(np.mean([classify(y) == t for y, t in zip(outputs_real, T_RAW)]))
    mae = float(np.mean(np.abs(np.array(outputs_real) - T_RAW)))
    return outputs_hat, outputs_real, accuracy, mae


def gradient_check(activation, seed=7, h=1e-6):
    """Сравнение аналитического градиента (backprop) с численным (центральная разность) на всех 4 примерах."""
    model = MLP221(activation=activation, seed=seed)
    worst = 0.0
    for x, t in zip(X, T):
        _, d_out, d_hid = model.gradients(x, t[None])
        analytic = -np.concatenate([np.outer(x, d_hid).ravel(), d_hid,
                                    np.outer(model.h_out, d_out).ravel(), d_out])
        base = model.params()
        numeric = np.zeros_like(base)
        for k in range(len(base)):
            for sign in (1, -1):
                p = base.copy()
                p[k] += sign * h
                m = MLP221(activation=activation, seed=seed)
                m.W1, m.b1 = p[:4].reshape(2, 2), p[4:6]
                m.W2, m.b2 = p[6:8].reshape(2, 1), p[8:9]
                numeric[k] += sign * bce_value(m.forward(x), t[None]) / (2 * h)
        worst = max(worst, float(np.max(np.abs(analytic - numeric))))
    return worst


def run_experiment(key, seed, verbose=False, lr=LR, init=INIT):
    cfg = CONFIGS[key]
    model = MLP221(activation=cfg["activation"], lr=lr, seed=seed, init=init)
    active_init = model.active_points()
    rng = np.random.default_rng(seed)
    epochs, Es, history = train_loop(model, X, T, EE, MAX_EPOCHS, rng)
    outputs_hat, outputs_real, accuracy, mae = evaluate(model)
    active_end = model.active_points()
    converged = Es <= EE

    if verbose:
        print(f"=== {cfg['name']} (seed = {seed}) ===")
        print(f"Критерий остановки Ee = {EE:.4f}, максимум эпох = {MAX_EPOCHS}")
        print(f"Эпох затрачено: {epochs}  |  Сходимость достигнута: {'да' if converged else 'нет'}")
        print(f"Итоговая суммарная ошибка Es = {Es:.6f}")
        print(f"{'A':>6}{'B':>6}{'A xor B (цель)':>16}{'ŷ':>10}{'выход сети':>14}{'класс':>8}")
        for (a, b), t_raw, y_hat, y_real in zip(X_RAW, T_RAW, outputs_hat, outputs_real):
            print(f"{a:6.0f}{b:6.0f}{t_raw:16.0f}{y_hat:10.4f}{y_real:14.3f}{classify(y_real):8.0f}")
        print(f"Accuracy = {accuracy * 100:.1f}%")
        print(f"Средняя абсолютная ошибка в шкале [c0; c1] = {mae:.4f}")
        print(f"Активных примеров у скрытых нейронов (z > 0): в начале {[int(v) for v in active_init]}, "
              f"в конце {[int(v) for v in active_end]}")
        print()
    return {"key": key, "name": cfg["name"], "model": model, "seed": seed, "epochs": epochs, "Es": Es,
            "converged": converged, "accuracy": accuracy, "mae": mae, "history": np.array(history),
            "outputs_hat": outputs_hat, "outputs_real": outputs_real,
            "active_init": active_init, "active_end": active_end}


def run_series(key, seeds):
    """Серия запусков с разной случайной инициализацией весов (и порядка примеров)."""
    return [run_experiment(key, s) for s in seeds]


def pick_representative(runs):
    """Представительный запуск: среди сошедшихся - с медианным числом эпох."""
    ok = sorted((r for r in runs if r["converged"]), key=lambda r: r["epochs"])
    return ok[len(ok) // 2] if ok else runs[0]


def dead_status(r):
    """Состояние скрытых нейронов: мёртвым считается нейрон, у которого z <= 0 на всех 4 примерах."""
    if r["model"].activation != "relu":
        return "-"
    dead_init = int(np.sum(r["active_init"] == 0))
    dead_end = int(np.sum(r["active_end"] == 0))
    if dead_end == 0:
        return "нет"
    parts = []
    if dead_init:
        parts.append(f"{dead_init} с начала")
    if dead_end > dead_init:
        parts.append(f"{dead_end - dead_init} умер при обучении")
    return f"{dead_end} ({', '.join(parts)})"


def series_stats(runs):
    ep = np.array([r["epochs"] for r in runs if r["converged"]])
    mae = [r["mae"] for r in runs if r["converged"]]
    return {"n": len(runs), "converged": len(ep),
            "min": int(ep.min()) if len(ep) else None, "max": int(ep.max()) if len(ep) else None,
            "mean": float(ep.mean()) if len(ep) else None, "median": float(np.median(ep)) if len(ep) else None,
            "std": float(ep.std()) if len(ep) else None,
            "mae": float(np.mean(mae)) if mae else None,
            "acc_mean": float(np.mean([r["accuracy"] for r in runs]))}


def print_series(runs_a, runs_b):
    for runs in (runs_a, runs_b):
        print(f"=== Серия запусков: конфигурация {runs[0]['name']} (Ee = {EE:.4f}) ===")
        print(f"{'seed':>5}{'эпох':>8}{'сошлась':>9}{'Es':>11}{'accuracy':>10}{'MAE':>9}   мёртвые нейроны")
        for r in runs:
            print(f"{r['seed']:>5}{r['epochs']:>8}{'да' if r['converged'] else 'нет':>9}{r['Es']:>11.6f}"
                  f"{r['accuracy'] * 100:>9.0f}%{r['mae']:>9.4f}   {dead_status(r)}")
        s = series_stats(runs)
        if s["converged"]:
            print(f"Сошлось {s['converged']} из {s['n']}; эпох: min={s['min']}, max={s['max']}, "
                  f"среднее={s['mean']:.0f}, медиана={s['median']:.0f}, std={s['std']:.0f}; "
                  f"MAE (сошедшиеся) = {s['mae']:.4f}")
        else:
            print(f"Сошлось 0 из {s['n']}")
        print()


def print_comparison(runs_a, runs_b):
    sa, sb = series_stats(runs_a), series_stats(runs_b)

    def fmt(v, f="{:.0f}"):
        return "-" if v is None else f.format(v)

    rows = [("Сошлось", f"{sa['converged']} из {sa['n']}", f"{sb['converged']} из {sb['n']}"),
            ("Эпох: мин - макс", f"{fmt(sa['min'])} - {fmt(sa['max'])}", f"{fmt(sb['min'])} - {fmt(sb['max'])}"),
            ("Эпох: среднее / медиана", f"{fmt(sa['mean'])} / {fmt(sa['median'])}",
             f"{fmt(sb['mean'])} / {fmt(sb['median'])}"),
            ("Эпох: std", fmt(sa["std"]), fmt(sb["std"])),
            ("Accuracy: среднее по запускам", f"{sa['acc_mean'] * 100:.1f}%", f"{sb['acc_mean'] * 100:.1f}%"),
            ("MAE (сошедшиеся)", fmt(sa["mae"], "{:.4f}"), fmt(sb["mae"], "{:.4f}"))]
    print("=== Таблица сравнения (статистика по серии) ===")
    print(f"{'показатель':<32}{runs_a[0]['name']:>16}{runs_b[0]['name']:>16}")
    for name, va, vb in rows:
        print(f"{name:<32}{va:>16}{vb:>16}")
    print()


# ----------------------------------------------------------------------------------------
# Дополнительное исследование (--extra): векторизованное обучение многих сетей одновременно.
# Каждая сеть обучается тем же онлайн-алгоритмом, с теми же начальными весами и порядком
# примеров, что и MLP221 при том же seed (совпадение проверяется в check_batch).
# ----------------------------------------------------------------------------------------
def train_many(activation, seeds, lr=LR, init=INIT, max_epochs=MAX_EPOCHS):
    act, dact = ACTIVATIONS[activation]
    n = len(seeds)
    W1, b1 = np.zeros((n, 2, 2)), np.zeros((n, 2))
    W2, b2 = np.zeros((n, 2)), np.zeros(n)
    rngs = []
    for k, s in enumerate(seeds):
        r = np.random.default_rng(s)
        W1[k] = r.uniform(-init, init, (2, 2))
        b1[k] = r.uniform(-init, init, (2,))
        W2[k] = r.uniform(-init, init, (2, 1))[:, 0]
        b2[k] = r.uniform(-init, init, (1,))[0]
        rngs.append(np.random.default_rng(s))
    dead_init = ((np.einsum("pi,nij->npj", X, W1) + b1[:, None, :]) <= 0).all(axis=1).sum(axis=1)

    epochs = np.full(n, max_epochs)
    Es_final = np.zeros(n)
    active = np.ones(n, dtype=bool)
    for epoch in range(1, max_epochs + 1):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        order = np.array([rngs[k].permutation(4) for k in idx])
        Es = np.zeros(len(idx))
        for step in range(4):
            x, t = X[order[:, step]], T[order[:, step]]
            z = np.einsum("ni,nij->nj", x, W1[idx]) + b1[idx]
            h = act(z)
            o = sigmoid(np.einsum("nj,nj->n", h, W2[idx]) + b2[idx])
            d_out = t - o
            oc = np.clip(o, 1e-12, 1.0 - 1e-12)
            Es += -(t * np.log(oc) + (1.0 - t) * np.log(1.0 - oc))
            d_hid = d_out[:, None] * W2[idx] * dact(z, h)
            W2[idx] += lr * h * d_out[:, None]
            b2[idx] += lr * d_out
            W1[idx] += lr * x[:, :, None] * d_hid[:, None, :]
            b1[idx] += lr * d_hid
        Es_final[idx] = Es
        done = Es <= EE
        epochs[idx[done]] = epoch
        active[idx[done]] = False

    z_end = np.einsum("pi,nij->npj", X, W1) + b1[:, None, :]
    dead_end = (z_end <= 0).all(axis=1).sum(axis=1)
    o_end = sigmoid(np.einsum("npj,nj->np", act(z_end), W2) + b2[:, None])
    acc = ((o_end > 0.5) == (T > 0.5)).mean(axis=1)
    return {"epochs": epochs, "Es": Es_final, "converged": Es_final <= EE,
            "dead_init": dead_init, "dead_end": dead_end, "accuracy": acc}


def check_batch(seeds=range(1, 6)):
    """Векторизованная реализация даёт те же эпохи и Es, что и MLP221."""
    worst = 0
    for key in ("A", "B"):
        res = train_many(CONFIGS[key]["activation"], list(seeds))
        for k, s in enumerate(seeds):
            r = run_experiment(key, s)
            worst = max(worst, abs(r["epochs"] - int(res["epochs"][k])), abs(r["Es"] - res["Es"][k]))
    return worst


def summarize(res):
    ok = res["converged"]
    ep = res["epochs"][ok]
    fail = ~ok
    return {"n": int(len(ok)), "converged": int(ok.sum()),
            "median": float(np.median(ep)) if len(ep) else None,
            "mean": float(ep.mean()) if len(ep) else None,
            "min": int(ep.min()) if len(ep) else None, "max": int(ep.max()) if len(ep) else None,
            "std": float(ep.std()) if len(ep) else None,
            "fail_dead_init": int(np.sum(fail & (res["dead_init"] > 0))),
            "fail_died": int(np.sum(fail & (res["dead_init"] == 0) & (res["dead_end"] > 0))),
            "fail_no_dead": int(np.sum(fail & (res["dead_end"] == 0))),
            "dead_init_any": int(np.sum(res["dead_init"] > 0)),
            "fail_acc": [float(a) for a in res["accuracy"][fail]]}


EXTRA_SEEDS = list(range(100, 150))
EXTRA_LRS = [0.5, 0.2, 0.1, 0.05]
EXTRA_INITS = [0.5, 1.0]
EXTRA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "extra_stats.json")


def extra_study():
    out = {"check": check_batch(), "grid": []}
    for activation in ("sigmoid", "relu", "leaky_relu"):
        for lr in EXTRA_LRS:
            for init in (EXTRA_INITS if activation != "leaky_relu" else [INIT]):
                s = summarize(train_many(activation, EXTRA_SEEDS, lr=lr, init=init))
                s.update({"activation": activation, "lr": lr, "init": init})
                out["grid"].append(s)
                print(f"  {activation:<11} η = {lr:<5} U[-{init}; {init}]: сошлось {s['converged']}/{s['n']}",
                      flush=True)
    os.makedirs(os.path.dirname(EXTRA_FILE), exist_ok=True)
    with open(EXTRA_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def load_extra():
    if os.path.exists(EXTRA_FILE):
        with open(EXTRA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


def print_extra(stats):
    names = {"sigmoid": "Сигмоида", "relu": "ReLU", "leaky_relu": f"LeakyReLU k={LEAKY_K}"}
    print(f"=== Дополнительное исследование: по 50 запусков (seed {EXTRA_SEEDS[0]}-{EXTRA_SEEDS[-1]}) ===")
    print(f"Проверка: векторизованное обучение совпадает с MLP221 (макс. расхождение {stats['check']:.1e})")
    print(f"{'активация':<20}{'η':>6}{'веса':>8}{'сошлось':>9}{'мин':>7}{'медиана':>9}{'среднее':>9}"
          f"{'макс':>7}{'std':>7}   неудачи: мёртв. с начала / умер / без мёртвых")
    for s in stats["grid"]:
        f = lambda v: "-" if v is None else f"{v:.0f}"
        dead = (f"{s['fail_dead_init']} / {s['fail_died']} / {s['fail_no_dead']}"
                if s["activation"] == "relu" else "-")
        print(f"{names[s['activation']]:<20}{s['lr']:>6}{'±' + str(s['init']):>8}{s['converged']:>6}/{s['n']:<3}"
              f"{f(s['min']):>7}{f(s['median']):>9}{f(s['mean']):>9}{f(s['max']):>7}{f(s['std']):>7}   {dead}")
    print()


# ----------------------------------------------------------------------------------------
# Режим функционирования
# ----------------------------------------------------------------------------------------
def functioning(run, a, b):
    """Прямой проход обученной сети для пары (A, B): ŷ, значение в шкале [c0; c1], класс."""
    y_hat = run["model"].predict(normalize(np.array([a, b])))
    y_real = float(from_binary(y_hat))
    return y_hat, y_real, classify(y_real)


def describe(run_a, run_b, a, b):
    lines = []
    for run in (run_a, run_b):
        yh, yr, cl = functioning(run, a, b)
        lines.append(f"  {run['name']:<12}: ŷ = {yh:.4f}  y = {yr:.3f}  -> класс {cl:.0f} "
                     f"({'c0' if cl == C0 else 'c1'})")
    return lines


def demo_predictions(run_a, run_b, examples):
    print("=== Режим функционирования (проверочные примеры) ===")
    print(f"{'A':>6}{'B':>6}{'А: ŷ':>9}{'А: y':>9}{'А: класс':>10}{'В: ŷ':>9}{'В: y':>9}{'В: класс':>10}")
    for a, b in examples:
        yh_a, yr_a, cl_a = functioning(run_a, a, b)
        yh_b, yr_b, cl_b = functioning(run_b, a, b)
        print(f"{a:6.1f}{b:6.1f}{yh_a:9.4f}{yr_a:9.3f}{cl_a:10.0f}{yh_b:9.4f}{yr_b:9.3f}{cl_b:10.0f}")
    print()


def interactive_loop(run_a, run_b):
    print("=== Интерактивный режим ===")
    print(f"Введите пару чисел A, B из диапазона [{LO:.0f}; {HI:.0f}] через пробел")
    print("(например: 3 -5). Для выхода введите 'q'.")
    while True:
        try:
            raw = input("A B > ").strip()
        except EOFError:
            break
        if not sys.stdin.isatty():
            print(raw)
        if raw.lower() in ("q", "quit", "exit"):
            break
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            print("Нужно ввести ровно два числа.")
            continue
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            print("Не удалось распознать числа.")
            continue
        if not (LO <= a <= HI and LO <= b <= HI):
            print(f"Значения должны быть в диапазоне [{LO:.0f}; {HI:.0f}].")
            continue
        for line in describe(run_a, run_b, a, b):
            print(line)


# ----------------------------------------------------------------------------------------
# Визуализация
# ----------------------------------------------------------------------------------------
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COLOR_A = "#1f77b4"    # синий - конфигурация А (сигмоида)
COLOR_B = "#d62728"    # красный - конфигурация В (ReLU)


def fig_convergence(rep_a, rep_b, fail_b, path):
    """Es(эпоха) для А и В на одних осях; серым - неудачный запуск ReLU (для сравнения)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for rep, color in ((rep_a, COLOR_A), (rep_b, COLOR_B)):
        ep = np.arange(1, len(rep["history"]) + 1)
        ax.plot(ep, rep["history"], color=color, lw=2.2,
                label=f"{rep['name']} (seed={rep['seed']}, {rep['epochs']} эпох)")
    if fail_b is not None:
        ep = np.arange(1, len(fail_b["history"]) + 1)
        ax.plot(ep, fail_b["history"], color="gray", lw=1.5, ls="--",
                label=f"{fail_b['name']}, неудачный запуск (seed={fail_b['seed']}, мёртвых нейронов: "
                      f"{int(np.sum(fail_b['active_end'] == 0))})")
    ax.axhline(EE, color="black", ls=":", lw=1.3, label=f"порог Ee = {EE:.4f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Номер эпохи (лог. шкала)")
    ax.set_ylabel("Суммарная ошибка Es = Σ BCE (лог. шкала)")
    ax.set_title("Сходимость: Es(эпоха) для сигмоиды и ReLU на скрытом слое")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def fig_epochs(runs_a, runs_b, path):
    """Число эпох по запускам (seed). Запуски, не достигшие критерия, - серая штриховка."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    seeds = [r["seed"] for r in runs_a]
    x = np.arange(len(seeds))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for runs, dx, color in ((runs_a, -w / 2, COLOR_A), (runs_b, w / 2, COLOR_B)):
        for xi, r in zip(x, runs):
            ok = r["converged"]
            ax.bar(xi + dx, r["epochs"], w, color=color if ok else "lightgray",
                   edgecolor=color if ok else "black", hatch=None if ok else "//")
            if ok:
                ax.text(xi + dx, r["epochs"] * 1.08, f"{r['epochs']}", ha="center", va="bottom", fontsize=8,
                        rotation=90)
            else:
                ax.text(xi + dx, 12000, "не\nсошлась", ha="center", va="center", fontsize=7,
                        bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_yscale("log")
    ax.set_ylim(10, MAX_EPOCHS * 20)
    ax.axhline(MAX_EPOCHS, color="gray", ls=":", lw=1)
    ax.text(len(seeds) - 0.5, MAX_EPOCHS * 1.15, f"лимит {MAX_EPOCHS} эпох", ha="right", va="bottom",
            fontsize=8, color="gray")
    ax.set_xlabel("Запуск (начальная инициализация весов)")
    ax.set_ylabel("Число эпох до Es ≤ Ee (лог. шкала)")
    ax.set_title(f"Разброс числа эпох по {len(seeds)} запускам")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(handles=[Patch(color=COLOR_A, label=runs_a[0]["name"]), Patch(color=COLOR_B, label=runs_b[0]["name"]),
                       Patch(facecolor="lightgray", edgecolor="black", hatch="//",
                             label="критерий не достигнут (50000 эпох)")],
              loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def grid_points(n=301):
    g = np.linspace(LO, HI, n)
    AA, BB = np.meshgrid(g, g)
    return AA, BB, normalize(np.stack([AA.ravel(), BB.ravel()], axis=1))


def draw_surface(ax, fig, run):
    """Карта выхода сети на плоскости (A, B) ∈ [-10; 10]^2 в шкале [c0; c1] + 4 обучающие точки."""
    from matplotlib.lines import Line2D
    model = run["model"]
    AA, BB, pts = grid_points()
    y_real = from_binary(model.forward(pts).ravel()).reshape(AA.shape)
    z_hidden = model.hidden_sums(pts)

    im = ax.imshow(y_real, origin="lower", extent=[LO, HI, LO, HI], cmap="viridis", vmin=C0, vmax=C1,
                   aspect="equal")
    mid = (C0 + C1) / 2.0
    ax.contour(AA, BB, y_real, levels=[mid], colors="black", linewidths=2.5)
    for j, ls in enumerate(("--", ":")):
        ax.contour(AA, BB, z_hidden[:, j].reshape(AA.shape), levels=[0.0], colors="red", linewidths=1.8,
                   linestyles=ls)
    for cls, face, marker in ((C0, "white", "o"), (C1, "black", "s")):
        m = T_RAW == cls
        ax.scatter(X_RAW[m, 0], X_RAW[m, 1], c=face, marker=marker, s=140, edgecolors="red", linewidths=1.8,
                   zorder=5)
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_title(f"{run['name']}: выход сети в шкале [c0; c1]\n(seed={run['seed']}, {run['epochs']} эпох)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"y в шкале [c0; c1]   (ŷ = (y − {C0:g}) / {C1 - C0:g})")
    kink = "излом ReLU" if model.activation == "relu" else "скрытый нейрон"
    return [Line2D([], [], color="black", lw=2.5, label=f"граница классов (y = {mid:g}, ŷ = 0.5)"),
            Line2D([], [], color="red", lw=1.8, ls="--", label="h1: z = 0"),
            Line2D([], [], color="red", lw=1.8, ls=":", label="h2: z = 0"),
            Line2D([], [], marker="o", color="w", markerfacecolor="white", markeredgecolor="red", markersize=9,
                   label=f"обучающие точки класса c0 = {C0:g}"),
            Line2D([], [], marker="s", color="w", markerfacecolor="black", markeredgecolor="red", markersize=9,
                   label=f"обучающие точки класса c1 = {C1:g}")]


def fig_surfaces(rep_a, rep_b, path):
    import matplotlib.pyplot as plt
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 6.6), layout="constrained")
    handles = draw_surface(ax_a, fig, rep_a)
    draw_surface(ax_b, fig, rep_b)
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Разделяющая поверхность: сигмоида (А) и ReLU (В) на скрытом слое", fontsize=14)
    fig.savefig(path, dpi=150)
    return fig


def fig_logit(rep_a, rep_b, path):
    """Изолинии логита выходного нейрона z_out(A, B) и его сечения вдоль двух прямых через обучающие точки."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout="constrained")
    AA, BB, pts = grid_points()
    s = np.linspace(LO, HI, 801)
    lines = [("B = A (через точки класса c0)", s, "gold"),
             ("A + B = −3 (через точки класса c1)", -3.0 - s, "orange")]
    for ax, rep in zip(axes[0], (rep_a, rep_b)):
        model = rep["model"]
        model.forward(pts)
        z_out = model.o_in.reshape(AA.shape)
        lim = np.max(np.abs(z_out))
        cf = ax.contourf(AA, BB, z_out, levels=30, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.contour(AA, BB, z_out, levels=20, colors="black", linewidths=0.6)
        ax.contour(AA, BB, z_out, levels=[0.0], colors="black", linewidths=2.5)
        z_hidden = model.hidden_sums(pts)
        for j, ls in enumerate(("--", ":")):
            ax.contour(AA, BB, z_hidden[:, j].reshape(AA.shape), levels=[0.0], colors="lime", linewidths=2,
                       linestyles=ls)
        for _, b_line, color in lines:
            ax.plot(s, b_line, color=color, lw=1.5)
        for cls, face, marker in ((C0, "white", "o"), (C1, "black", "s")):
            m = T_RAW == cls
            ax.scatter(X_RAW[m, 0], X_RAW[m, 1], c=face, marker=marker, s=110, edgecolors="black", zorder=5)
        ax.set_xlim(LO, HI)
        ax.set_ylim(LO, HI)
        ax.set_aspect("equal")
        ax.set_xlabel("A")
        ax.set_ylabel("B")
        ax.set_title(f"{rep['name']}: изолинии логита z_out (жирная: z_out = 0)\n"
                     f"зелёные: z_h = 0 скрытых нейронов; жёлтая и оранжевая: сечения")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="z_out (ŷ = σ(z_out))")

    for ax, (title, b_line, color) in zip(axes[1], lines):
        xy = normalize(np.stack([s, b_line], axis=1))
        for rep, c in ((rep_a, COLOR_A), (rep_b, COLOR_B)):
            model = rep["model"]
            model.forward(xy)
            z = model.o_in.ravel()
            ax.plot(s, z, color=c, lw=2.2, label=rep["name"])
            if model.activation == "relu":
                zh = model.hidden_sums(xy)
                for j in range(2):
                    k = np.flatnonzero(np.diff(np.sign(zh[:, j])) != 0)
                    ax.plot(s[k], z[k], "v", color="black", ms=9,
                            label="излом ReLU (z_h = 0)" if j == 0 and len(k) else None)
        for a_pt, b_pt in X_RAW:
            if np.isclose(np.interp(a_pt, s, b_line), b_pt):
                ax.axvline(a_pt, color="gray", ls=":", lw=1.2)
        ax.axhline(0.0, color="black", lw=1)
        ax.set_xlabel("A (вертикальные пунктиры - обучающие точки на прямой)")
        ax.set_ylabel("логит выходного нейрона z_out")
        ax.set_title(f"Сечение вдоль прямой {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Функция сети до выходной сигмоиды: гладкая (сигмоида) и кусочно-линейная (ReLU)", fontsize=14)
    fig.savefig(path, dpi=150)
    return fig


def fig_extra(stats, path):
    """Доля сходящихся запусков при разных η (U[-0.5; 0.5]) и причины неудач ReLU."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), layout="constrained")
    styles = {"sigmoid": (COLOR_A, "Сигмоида"), "relu": (COLOR_B, "ReLU"),
              "leaky_relu": ("#2ca02c", f"LeakyReLU (k = {LEAKY_K})")}
    x = np.arange(len(EXTRA_LRS))
    w = 0.26
    for i, (act, (color, label)) in enumerate(styles.items()):
        rows = [s for s in stats["grid"] if s["activation"] == act and s["init"] == INIT]
        vals = [next(s for s in rows if s["lr"] == lr)["converged"] for lr in EXTRA_LRS]
        bars = ax1.bar(x + (i - 1) * w, vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"η = {lr}" for lr in EXTRA_LRS])
    ax1.set_ylim(0, 62)
    ax1.set_ylabel(f"Сошлось из {len(EXTRA_SEEDS)} запусков")
    ax1.set_title("Устойчивость к инициализации при разных η (веса из U[−0.5; 0.5])")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper center", ncol=3, fontsize=9)

    rows = [s for s in stats["grid"] if s["activation"] == "relu" and s["init"] == INIT]
    rows = [next(s for s in rows if s["lr"] == lr) for lr in EXTRA_LRS]
    parts = [("converged", "#2ca02c", "сошлась"), ("fail_dead_init", "#7f7f7f", "мёртвый нейрон с начала"),
             ("fail_died", "#ff7f0e", "нейрон умер при обучении"), ("fail_no_dead", "#9467bd",
                                                                   "без мёртвых (лок. минимум)")]
    bottom = np.zeros(len(rows))
    for key, color, label in parts:
        vals = np.array([s[key] for s in rows])
        ax2.bar(x, vals, 0.55, bottom=bottom, color=color, label=label)
        for xi, v, b in zip(x, vals, bottom):
            if v:
                ax2.text(xi, b + v / 2, str(v), ha="center", va="center", fontsize=9, color="white")
        bottom += vals
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"η = {lr}" for lr in EXTRA_LRS])
    ax2.set_ylabel(f"Число запусков из {len(EXTRA_SEEDS)}")
    ax2.set_title("ReLU: исходы обучения и причины неудач")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    fig.savefig(path, dpi=150)
    return fig


def make_figures(runs_a, runs_b, rep_a, rep_b, extra):
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = {"convergence": os.path.join(FIG_DIR, "fig1_convergence.png"),
             "epochs": os.path.join(FIG_DIR, "fig2_epochs.png"),
             "surface": os.path.join(FIG_DIR, "fig3_surfaces.png"),
             "logit": os.path.join(FIG_DIR, "fig4_logit.png")}
    fail_b = next((r for r in runs_b if not r["converged"]), None)
    fig_convergence(rep_a, rep_b, fail_b, paths["convergence"])
    fig_epochs(runs_a, runs_b, paths["epochs"])
    fig_surfaces(rep_a, rep_b, paths["surface"])
    fig_logit(rep_a, rep_b, paths["logit"])
    if extra is not None:
        paths["extra"] = os.path.join(FIG_DIR, "fig5_relu_stability.png")
        fig_extra(extra, paths["extra"])
    return paths


SEEDS = list(range(1, 11))
DEMO_EXAMPLES = [(-9, -9), (-9, 6), (6, -9), (6, 6), (0, 0), (3, -7), (-5, 8)]


if __name__ == "__main__":
    show = "--no-show" not in sys.argv

    if not show:
        import matplotlib
        matplotlib.use("Agg")

    print(f"Цели (BCE): c0 = {C0:g} -> 0, c1 = {C1:g} -> 1;  порог Ee = {EE:.4f};  η = {LR};  "
          f"веса из U[-{INIT}; {INIT}]")
    print(f"Проверка backprop (аналитический градиент vs численный): "
          f"сигмоида {gradient_check('sigmoid'):.1e}, ReLU {gradient_check('relu'):.1e}")
    print()

    # Серия из 10 запусков с разной инициализацией весов (seed = 1..10)
    runs_a = run_series("A", SEEDS)
    runs_b = run_series("B", SEEDS)
    rep_a, rep_b = pick_representative(runs_a), pick_representative(runs_b)

    # Представительные запуски (медиана по эпохам среди сошедшихся)
    for rep in (rep_a, rep_b):
        run_experiment(rep["key"], rep["seed"], verbose=True)
    fail_b = next((r for r in runs_b if not r["converged"]), None)
    if fail_b is not None:
        print("Пример неудачного запуска ReLU:")
        run_experiment("B", fail_b["seed"], verbose=True)

    print_series(runs_a, runs_b)
    print(f"Представительные запуски (медиана по эпохам среди сошедшихся): "
          f"А - seed {rep_a['seed']}, В - seed {rep_b['seed']}")
    print()
    print_comparison(runs_a, runs_b)

    extra = extra_study() if "--extra" in sys.argv else load_extra()
    if extra is not None:
        print_extra(extra)

    paths = make_figures(runs_a, runs_b, rep_a, rep_b, extra)
    print("Графики сохранены:")
    for p in paths.values():
        print("  ", p)
    print()

    demo_predictions(rep_a, rep_b, DEMO_EXAMPLES)

    if "--no-input" not in sys.argv:
        interactive_loop(rep_a, rep_b)

    if show:
        import matplotlib.pyplot as plt
        plt.show()
