import numpy as np
import matplotlib.pyplot as plt

C0, C1 = -10.0, 9.0
SPAN = C1 - C0

X_raw = np.array([[-10, -10], [-10, 9], [9, -10], [9, 9]], dtype=float)
T_raw = np.array([-10, 9, 9, -10], dtype=float)

def norm(x):   return (x - C0) / SPAN
def denorm(y): return C0 + y * SPAN

X = norm(X_raw)
T = norm(T_raw)

SEEDS      = [0, 1, 2, 3, 4]
MAX_EPOCHS = 10000
Ee_MSE     = 0.001
Ee_BCE     = 0.05
LR         = 1.0
INIT       = 0.5

def sigmoid(s):
    s = np.clip(s, -60, 60)
    return 1.0 / (1.0 + np.exp(-s))

class MLP:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-INIT, INIT, (2, 2))
        self.b1 = rng.uniform(-INIT, INIT, 2)
        self.W2 = rng.uniform(-INIT, INIT, (2, 1))
        self.b2 = rng.uniform(-INIT, INIT, 1)

    def forward(self, x):
        h = sigmoid(x @ self.W1 + self.b1)
        y = sigmoid(h @ self.W2 + self.b2)
        return h, y

    def train(self, loss, Ee, max_epochs=MAX_EPOCHS):
        hist = []
        for epoch in range(1, max_epochs + 1):
            for x, t in zip(X, T):
                h, y = self.forward(x)
                if loss == "bce":
                    d_out = y - t
                else:
                    d_out = (y - t) * y * (1.0 - y)
                d_h = (self.W2[:, 0] * d_out[0]) * h * (1.0 - h)
                self.W2 -= LR * np.outer(h, d_out)
                self.b2 -= LR * d_out
                self.W1 -= LR * np.outer(x, d_h)
                self.b1 -= LR * d_h

            Es = 0.0
            for x, t in zip(X, T):
                _, y = self.forward(x)
                if loss == "bce":
                    yc = np.clip(y[0], 1e-12, 1 - 1e-12)
                    Es += -(t * np.log(yc) + (1 - t) * np.log(1 - yc))
                else:
                    Es += 0.5 * (t - y[0]) ** 2
            hist.append(Es)
            if Es <= Ee:
                return epoch, Es, hist, True
        return max_epochs, Es, hist, False

    def predict_norm(self):
        return np.array([self.forward(x)[1][0] for x in X])

def evaluate(net):
    yn = net.predict_norm()
    yr = denorm(yn)
    acc = float(np.mean((yn >= 0.5).astype(int) == T.astype(int)))
    mae = float(np.mean(np.abs(yr - T_raw)))
    return acc, mae, yn, yr

def run_config(loss, Ee):
    out = []
    for s in SEEDS:
        net = MLP(s)
        ep, Es, hist, conv = net.train(loss, Ee)
        acc, mae, yn, yr = evaluate(net)
        out.append(dict(seed=s, net=net, epochs=ep, Es=Es, hist=hist,
                        conv=conv, acc=acc, mae=mae))
    return out

def pick_rep(runs):
    for r in runs:
        if r["conv"]: return r
    return runs[0]

def print_table(name, runs, Ee):
    print(f"\n===== {name} (Ee={Ee}) =====")
    print(f"{'Seed':>4} | {'Эпохи':>6} | {'Ошибка':>10} | "
          f"{'Acc':>5} | {'MAE':>7} | Сходилась")
    print("-" * 55)
    for r in runs:
        print(f"{r['seed']:>4} | {r['epochs']:>6} | "
              f"{r['Es']:>10.6f} | {r['acc']:>5.3f} | "
              f"{r['mae']:>7.4f} | {'да' if r['conv'] else 'НЕТ'}")
    ok = [r for r in runs if r["conv"]]
    print(f"Сошлось: {len(ok)}/{len(runs)}")
    if ok:
        e = np.array([r["epochs"] for r in ok])
        print(f"Разброс эпох: {e.min()}–{e.max()}")

def plot_convergence(repA, repB):
    plt.figure(figsize=(9, 5))
    plt.plot(repA["hist"], label=f"MSE (Ee={Ee_MSE})", color="tab:blue")
    plt.plot(repB["hist"], label=f"BCE (Ee={Ee_BCE})", color="tab:red")
    plt.axhline(Ee_MSE, ls="--", color="tab:blue", alpha=0.5)
    plt.axhline(Ee_BCE, ls="--", color="tab:red", alpha=0.5)
    plt.xlabel("Эпоха"); plt.ylabel("Суммарная ошибка Es")
    # Changed title here
    plt.title(f"Сходимость (c0={C0:g}, c1={C1:g})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("convergence.png", dpi=150); plt.show()

def plot_epochs_bar(runsA, runsB):
    x = np.arange(len(SEEDS)); w = 0.38
    eA = [r["epochs"] if r["conv"] else MAX_EPOCHS for r in runsA]
    eB = [r["epochs"] if r["conv"] else MAX_EPOCHS for r in runsB]
    fig, ax = plt.subplots(figsize=(9, 5))
    bA = ax.bar(x - w/2, eA, w, label="MSE", color="tab:blue")
    bB = ax.bar(x + w/2, eB, w, label="BCE", color="tab:red")
    for bars, runs in [(bA, runsA), (bB, runsB)]:
        for bar, r in zip(bars, runs):
            if not r["conv"]:
                bar.set_hatch("//"); bar.set_edgecolor("black")
    ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_xlabel("Seed"); ax.set_ylabel("Эпох")
    ax.set_title("Количество эпох по 5 запускам (штриховка — не сошлась)")
    ax.legend(); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig("epochs.png", dpi=150); plt.show()

def plot_surface(net, ax, title):
    g = 120
    a = np.linspace(C0, C1, g)
    AA, BB = np.meshgrid(a, a)
    Z = np.zeros_like(AA)
    for i in range(g):
        for j in range(g):
            x = norm(np.array([AA[i, j], BB[i, j]]))
            Z[i, j] = denorm(net.forward(x)[1][0])
    cf = ax.contourf(AA, BB, Z, levels=20, cmap="viridis")
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=T_raw, cmap="coolwarm",
               edgecolor="k", s=120, vmin=C0, vmax=C1)
    ax.set_xlabel("A"); ax.set_ylabel("B"); ax.set_title(title)
    ax.set_xlim(C0, C1); ax.set_ylim(C0, C1)
    return cf

def plot_surfaces(netA, netB):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    c1 = plot_surface(netA, axs[0], f"MSE (в шкале [{C0:g};{C1:g}])")
    c2 = plot_surface(netB, axs[1], f"BCE (в шкале [{C0:g};{C1:g}])")
    for cf, ax in [(c1, axs[0]), (c2, axs[1])]:
        cb = fig.colorbar(cf, ax=ax); cb.set_label("Выход сети")
    plt.tight_layout(); plt.savefig("surfaces.png", dpi=150); plt.show()

def plot_mae(runsA, runsB):
    mA = np.mean([r["mae"] for r in runsA])
    mB = np.mean([r["mae"] for r in runsB])
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["MSE", "BCE"], [mA, mB], color=["tab:blue", "tab:red"])
    ax.set_ylabel(f"MAE в шкале [{C0:g};{C1:g}]")
    ax.set_title("Средняя абсолютная ошибка в шкале [c0;c1]")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, [mA, mB]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom")
    plt.tight_layout(); plt.savefig("mae.png", dpi=150); plt.show()

def demo(net):
    print("\n===== Проверочные примеры =====")
    for A, B in [(C0, C0), (C0, C1), (C1, C0), (C1, C1),
                 (0, 0), (-5, 5), (5, -5), (2, 2), (-8, 3)]:
        x = norm(np.array([A, B]))
        yn = float(net.forward(x)[1][0])
        yr = denorm(yn)
        cls = C1 if yn >= 0.5 else C0
        print(f"A={A:>5.1f}, B={B:>5.1f}: "
              f"y_norm={yn:.4f}, y_real={yr:.4f}, класс={cls:g}")

    print("\nВведите A и B через пробел (q для выхода):")
    while True:
        s = input("> ").strip()
        if s.lower() == "q": break
        try:
            A, B = map(float, s.split())
        except Exception:
            print("Ошибка ввода"); continue
        x = norm(np.array([A, B]))
        yn = float(net.forward(x)[1][0])
        yr = denorm(yn)
        cls = C1 if yn >= 0.5 else C0
        print(f"  Нормализованный выход: {yn:.4f}")
        print(f"  Выход в исходной шкале: {yr:.4f}")
        print(f"  Ближайший класс: {cls:g}")

def main():
    # Changed print here
    print(f"c0={C0:g},  c1={C1:g}\n")
    for x, y in zip(X_raw, T_raw):
        print(f"  A={x[0]:>4.0f}, B={x[1]:>4.0f} -> {y:>4.0f}")

    runsA = run_config("mse", Ee_MSE)
    runsB = run_config("bce", Ee_BCE)
    print_table("MSE (Конфигурация А)", runsA, Ee_MSE)
    print_table("BCE (Конфигурация Б)", runsB, Ee_BCE)

    repA, repB = pick_rep(runsA), pick_rep(runsB)
    print(f"\nПредставители: MSE seed={repA['seed']}, BCE seed={repB['seed']}")

    plot_convergence(repA, repB)
    plot_epochs_bar(runsA, runsB)
    plot_surfaces(repA["net"], repB["net"])
    plot_mae(runsA, runsB)
    demo(runsB[2]["net"])

if __name__ == "__main__":
    main()