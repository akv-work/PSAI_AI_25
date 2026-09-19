"""
Лабораторная работа №2, вариант 3 (c0 = -9, c1 = -8).

Сравнение MSE (ЛР №1) и Binary Cross-Entropy на MLP 2-2-1 для XOR.

Отличия от lab1.py (ЛР №1):
- две функции потерь: MSE (конфиг. А) и BCE (конфиг. Б);
- для BCE градиент выходного слоя упрощается до (ŷ − t);
- цели c0/c1 отображаются в {0, 1} перед BCE и обратно после обучения;
- серия из 5 запусков с разными seed, метрики accuracy/MAE, визуализация.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# --- Вариант 3: тот же, что в ЛР №1 ---
C0, C1 = -9.0, -8.0
X = np.array([[-9, -9], [-9, -8], [-8, -9], [-8, -8]], dtype=float)
Y = np.array([-9, -8, -8, -9], dtype=float)

# Нормализация целей: c0 → 0, c1 → 1  (нужна BCE и сигмоиде; в ЛР №1 то же отображение)
T = (Y - C0) / (C1 - C0)

# Гиперпараметры ЛР №1 (для MSE). Для BCE скорость меньше:
# при ŷ=0.5 градиент BCE в 4 раза больше, чем у MSE+сигмоида
# ((ŷ−t) против (ŷ−t)·ŷ·(1−ŷ)), поэтому η_BCE = 2.0 / 4 = 0.5
# уравнивает стартовый шаг и не даёт BCE «перепрыгивать» минимум.
MAX_EPOCHS = 10_000
LR_MSE = 2.0
LR_BCE = 0.5
INIT_SCALE = 0.5

# Пороги остановки. MSE и BCE — разные величины, поэтому Ee раздельные.
# Ee_MSE = 0.01 — как в ЛР №1 (сумма квадратов на шкале 0/1).
# Ee_BCE подобран так, чтобы соответствовать той же точности классификации:
# при |ŷ − t| ≈ 0.05 (это RMS для Es_MSE = 0.01 на 4 примерах)
# BCE ≈ −log(0.95) ≈ 0.051 на пример → сумма ≈ 0.20.
EE_MSE = 0.01
EE_BCE = 0.20

SEEDS = [0, 1, 2, 3, 4]
REP_SEED = 0  # представительный запуск для графиков
EPS = 1e-7
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def dsig(y):
    return y * (1.0 - y)


def norm_in(x):
    """Входы в [0, 1] — как в ЛР №1: (x − c0) / (c1 − c0)."""
    return (np.asarray(x, dtype=float) - C0) / (C1 - C0)


def to_real(y_hat):
    """Обратный пересчёт ŷ ∈ (0, 1) → шкала [c0, c1] (в ЛР №1 это scale)."""
    return C0 + float(y_hat) * (C1 - C0)


def to_class(v):
    return C1 if abs(v - C1) < abs(v - C0) else C0


def mse_sum(preds, targets):
    return float(np.sum((preds - targets) ** 2))


def bce_sum(preds, targets):
    p = np.clip(preds, EPS, 1.0 - EPS)
    return float(-np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))


class MLP:
    """Многослойный персептрон 2-2-1, сигмоида на всех слоях, online-backprop — как в ЛР №1."""

    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(2, 2) * INIT_SCALE
        self.b1 = np.zeros(2)
        self.W2 = rng.randn(2, 1) * INIT_SCALE
        self.b2 = np.zeros(1)

    def forward(self, x):
        x = norm_in(x).ravel()
        self.h = sigmoid(x @ self.W1 + self.b1)
        self.y_hat = float(sigmoid(self.h @ self.W2 + self.b2).item())
        return self.y_hat

    def step(self, x, t, loss, lr):
        y = self.forward(x)
        x = norm_in(x).ravel()

        # Конфиг. А (MSE + сигмоида, как в ЛР №1): δ = (ŷ − t) · ŷ · (1 − ŷ)
        # Конфиг. Б (BCE + сигмоида, новое):     δ = ŷ − t
        #   L = −[t ln ŷ + (1−t) ln(1−ŷ)]
        #   ∂L/∂ŷ = (ŷ − t) / (ŷ(1−ŷ)),  ∂ŷ/∂z = ŷ(1−ŷ)  ⇒  ∂L/∂z = ŷ − t
        if loss == "mse":
            delta = (y - t) * y * (1.0 - y)
        elif loss == "bce":
            delta = y - t
        else:
            raise ValueError(loss)

        self.W2 -= lr * self.h.reshape(-1, 1) * delta
        self.b2 -= lr * delta
        dh = (self.W2.ravel() * delta) * dsig(self.h)
        self.W1 -= lr * np.outer(x, dh)
        self.b1 -= lr * dh

    def predict_all(self):
        return np.array([self.forward(x) for x in X])


def train(seed, loss):
    net = MLP(seed)
    ee = EE_MSE if loss == "mse" else EE_BCE
    lr = LR_MSE if loss == "mse" else LR_BCE
    history = []
    err = None

    for epoch in range(1, MAX_EPOCHS + 1):
        for x, t in zip(X, T):
            net.step(x, t, loss, lr)
        preds = net.predict_all()
        err = mse_sum(preds, T) if loss == "mse" else bce_sum(preds, T)
        history.append(err)
        if err <= ee:
            converged = True
            return {
                "net": net,
                "epochs": epoch,
                "err": err,
                "history": history,
                "converged": converged,
            }

    return {
        "net": net,
        "epochs": MAX_EPOCHS,
        "err": err,
        "history": history,
        "converged": False,
    }


def metrics(net):
    preds = net.predict_all()
    real = np.array([to_real(p) for p in preds])
    acc = float(np.mean([to_class(r) == y for r, y in zip(real, Y)]))
    mae = float(np.mean(np.abs(real - Y)))
    return preds, real, acc, mae


def evaluate(seed, loss):
    result = train(seed, loss)
    preds, real, acc, mae = metrics(result["net"])
    result.update({"preds": preds, "real": real, "acc": acc, "mae": mae, "seed": seed, "loss": loss})
    return result


def print_table(name, runs):
    print(f"\n=== {name} ===")
    print(f"{'seed':>6} {'эпохи':>8} {'сход.':>6} {'Es':>12} {'acc':>8} {'MAE':>10}")
    n_ok = 0
    epochs_ok = []
    for r in runs:
        mark = "да" if r["converged"] else "нет"
        n_ok += int(r["converged"])
        if r["converged"]:
            epochs_ok.append(r["epochs"])
        print(
            f"{r['seed']:6d} {r['epochs']:8d} {mark:>6} {r['err']:12.6f} "
            f"{r['acc'] * 100:7.0f}% {r['mae']:10.4f}"
        )
    print(f"Сходимость: {n_ok}/5 запусков")
    if epochs_ok:
        print(
            f"Эпохи (успешные): min={min(epochs_ok)}, max={max(epochs_ok)}, "
            f"mean={np.mean(epochs_ok):.1f}, std={np.std(epochs_ok):.1f}"
        )


def plot_convergence(mse_hist, bce_hist, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(mse_hist) + 1), mse_hist, color="#1f77b4", lw=2, label="А: MSE")
    ax.plot(range(1, len(bce_hist) + 1), bce_hist, color="#d62728", lw=2, ls="--", label="Б: BCE")
    ax.axhline(EE_MSE, color="#1f77b4", ls=":", lw=1.5, label=f"Ee MSE = {EE_MSE}")
    ax.axhline(EE_BCE, color="#d62728", ls=":", lw=1.5, label=f"Ee BCE = {EE_BCE}")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Суммарная ошибка Es")
    ax.set_title("Сходимость MLP 2-2-1: MSE vs BCE (seed = 0)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_epochs(mse_runs, bce_runs, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.arange(len(SEEDS))
    w = 0.35
    mse_e = [r["epochs"] for r in mse_runs]
    bce_e = [r["epochs"] for r in bce_runs]
    mse_ok = [r["converged"] for r in mse_runs]
    bce_ok = [r["converged"] for r in bce_runs]

    bars_a = ax.bar(idx - w / 2, mse_e, w, color="#1f77b4", label="А: MSE")
    bars_b = ax.bar(idx + w / 2, bce_e, w, color="#d62728", label="Б: BCE")
    for bar, ok in zip(bars_a, mse_ok):
        if not ok:
            bar.set_hatch("///")
            bar.set_edgecolor("black")
    for bar, ok in zip(bars_b, bce_ok):
        if not ok:
            bar.set_hatch("///")
            bar.set_edgecolor("black")

    fail_proxy = plt.Rectangle((0, 0), 1, 1, facecolor="0.7", hatch="///", edgecolor="black")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Число эпох до остановки")
    ax.set_title("Устойчивость сходимости по 5 запускам")
    ax.legend(handles=[bars_a, bars_b, fail_proxy], labels=["А: MSE", "Б: BCE", "критерий не достигнут"])
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def decision_grid(net, lim=(-10.0, 10.0), n=201):
    a = np.linspace(lim[0], lim[1], n)
    b = np.linspace(lim[0], lim[1], n)
    aa, bb = np.meshgrid(a, b)
    zz = np.empty_like(aa)
    for i in range(n):
        for j in range(n):
            zz[i, j] = to_real(net.forward([aa[i, j], bb[i, j]]))
    return aa, bb, zz


def _draw_surface(fig, ax, net, title, lim):
    aa, bb, zz = decision_grid(net, lim=lim)
    im = ax.contourf(aa, bb, zz, levels=20, cmap="RdYlBu", vmin=C0, vmax=C1)
    fig.colorbar(im, ax=ax, label="ŷ в шкале [c0, c1]")
    seen = set()
    for x, y in zip(X, Y):
        color = "#1f77b4" if y == C0 else "#d62728"
        marker = "o" if y == C0 else "s"
        label = f"класс {int(y)}" if y not in seen else None
        seen.add(y)
        ax.scatter(x[0], x[1], c=color, marker=marker, s=80, edgecolors="k", zorder=3, label=label)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_title(title)
    ax.set_aspect("equal")


def plot_surfaces_and_mae(mse_run, bce_run, path, zoom_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    _draw_surface(fig, axes[0, 0], mse_run["net"], "А: MSE, диапазон [−10; 10]", (-10, 10))
    _draw_surface(fig, axes[0, 1], bce_run["net"], "Б: BCE, диапазон [−10; 10]", (-10, 10))

    ax = axes[1, 0]
    ax.bar(["А: MSE", "Б: BCE"], [mse_run["mae"], bce_run["mae"]], color=["#1f77b4", "#d62728"])
    ax.set_ylabel("MAE в шкале [c0, c1]")
    ax.set_title("Точность восстановления исходной шкалы")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(["А: MSE", "Б: BCE"], [mse_run["acc"] * 100, bce_run["acc"] * 100], color=["#1f77b4", "#d62728"])
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(0, 110)
    ax.set_title("Точность классификации на 4 примерах")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Разделяющая поверхность и качество восстановления шкалы (seed = 0)", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Цветовую шкалу строим в [c0; c1], а не в (0; 1): так обе конфигурации сравнимы.
    # Дополнительно — зум около обучающих точек, иначе XOR на [−10; 10] сжимается в угол.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    zoom = (-10.2, -6.8)
    _draw_surface(fig, axes[0], mse_run["net"], "А: MSE, зум около обучающей выборки", zoom)
    _draw_surface(fig, axes[1], bce_run["net"], "Б: BCE, зум около обучающей выборки", zoom)
    fig.suptitle("Форма границы XOR (пересечение полуплоскостей скрытого слоя)")
    fig.tight_layout()
    fig.savefig(zoom_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def demo_inference(net, title):
    extra = np.array(
        [
            [-10.0, -10.0],
            [0.0, 0.0],
            [-8.5, -9.0],
            [10.0, 10.0],
            [-9.0, 0.0],
        ]
    )
    print(f"\n--- Режим функционирования ({title}) ---")
    print(f"{'A':>8} {'B':>8} {'ŷ (0;1)':>12} {'y_real':>10} {'класс':>8}")
    for x in np.vstack([X, extra]):
        y_hat = net.forward(x)
        y_real = to_real(y_hat)
        print(f"{x[0]:8.1f} {x[1]:8.1f} {y_hat:12.4f} {y_real:10.4f} {to_class(y_real):8.0f}")


def interactive_loop(net):
    print("\n--- Интерактивный ввод (выход: q) ---")
    while True:
        s = input("A B [-10..10]: ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            a, b = map(float, s.split())
            if not (-10 <= a <= 10 and -10 <= b <= 10):
                raise ValueError
            y_hat = net.forward([a, b])
            y_real = to_real(y_hat)
            print(
                f"ŷ={y_hat:.4f}, y_real={y_real:.4f}, ближе к c={to_class(y_real):g}"
            )
        except ValueError:
            print("Введите два числа из [-10, 10]")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("ЛР №2, вариант 3: c0 = -9, c1 = -8")
    print(f"Архитектура 2-2-1, max эпох={MAX_EPOCHS}")
    print(f"η_MSE={LR_MSE} (ЛР №1), η_BCE={LR_BCE} (компенсация масштаба градиента)")
    print(f"Ee_MSE={EE_MSE} (как в ЛР №1), Ee_BCE={EE_BCE} (эквивалент по |ŷ−t|≈0.05)")

    mse_runs = [evaluate(seed, "mse") for seed in SEEDS]
    bce_runs = [evaluate(seed, "bce") for seed in SEEDS]

    print_table("Конфигурация А — MSE (как ЛР №1)", mse_runs)
    print_table("Конфигурация Б — BCE", bce_runs)

    mse_rep = next(r for r in mse_runs if r["seed"] == REP_SEED)
    bce_rep = next(r for r in bce_runs if r["seed"] == REP_SEED)

    print("\nПредставительный запуск (seed = 0), выходы на обучающей выборке:")
    for name, run in (("MSE", mse_rep), ("BCE", bce_rep)):
        print(f"  {name}: эпохи={run['epochs']}, Es={run['err']:.6f}, acc={run['acc']*100:.0f}%, MAE={run['mae']:.4f}")
        for x, y, p, r in zip(X, Y, run["preds"], run["real"]):
            print(f"    ({x[0]:g},{x[1]:g}) ŷ={p:.4f} y_real={r:.4f} класс={to_class(r):g}  ожид. {y:g}")

    plot_convergence(
        mse_rep["history"],
        bce_rep["history"],
        os.path.join(FIG_DIR, "convergence.png"),
    )
    plot_epochs(mse_runs, bce_runs, os.path.join(FIG_DIR, "epochs.png"))
    plot_surfaces_and_mae(
        mse_rep,
        bce_rep,
        os.path.join(FIG_DIR, "surfaces_mae.png"),
        os.path.join(FIG_DIR, "surfaces_zoom.png"),
    )
    print(f"\nГрафики сохранены в {FIG_DIR}")

    demo_inference(mse_rep["net"], "конфиг. А, MSE")
    demo_inference(bce_rep["net"], "конфиг. Б, BCE")

    if sys.stdin.isatty():
        interactive_loop(bce_rep["net"])


if __name__ == "__main__":
    main()
