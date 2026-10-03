import numpy as np
import matplotlib.pyplot as plt

C0, C1 = 3, -8
DATA = [(3, 3, C0), (3, -8, C1), (-8, 3, C1), (-8, -8, C0)]
LABELS = np.array([0.0 if y == C0 else 1.0 for _, _, y in DATA])   # c0 -> 0, c1 -> 1
Y_REAL = np.array([y for _, _, y in DATA], dtype=float)

LR = 0.5
MAX_EPOCHS = 100_000
EE_MSE = 0.001
EE_BCE = 0.1
SEEDS = [0, 1, 2, 3, 4]
T_LO, T_HI = 0.1, 0.9  # цели для MSE: сигмоида не может выдать ровно 0 и 1

X = np.array([[a / 10.0, b / 10.0] for a, b, _ in DATA])   # входы [-10;10] -> [-1;1]
sigmoid = lambda s: 1.0 / (1.0 + np.exp(-s))

# ---------- шкалы ----------
# Б: метка 0/1 -> шкала [c0; c1]: y = c0 + yhat * (c1 - c0)
def to_real_B(yh): return C0 + yh * (C1 - C0)
# А: цель 0.1/0.9 -> шкала [c0; c1]
def to_real_A(yh): return C0 + (yh - T_LO) / (T_HI - T_LO) * (C1 - C0)

T_A = T_LO + LABELS * (T_HI - T_LO)     # цели конфигурации А
T_B = LABELS                            # цели конфигурации Б

# ---------- сеть 2-2-1 ----------
class MLP:
    def __init__(self, rng):
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2)); self.b1 = rng.uniform(-0.5, 0.5, 2)
        self.W2 = rng.uniform(-0.5, 0.5, 2);      self.b2 = rng.uniform(-0.5, 0.5)

    def predict(self, x):
        self.h = sigmoid(x @ self.W1 + self.b1)
        self.y = sigmoid(self.h @ self.W2 + self.b2)
        return self.y

    def step(self, x, t, lr, loss):
        y = self.predict(x)
        # MSE + сигмоида: dE/ds = (y - t) * y * (1 - y)
        # BCE + сигмоида: dE/ds = (y - t)   (множитель y(1-y) сокращается)
        d_out = (y - t) * y * (1 - y) if loss == "mse" else (y - t)
        d_hid = d_out * self.W2 * self.h * (1 - self.h)
        self.W2 -= lr * d_out * self.h; self.b2 -= lr * d_out
        self.W1 -= lr * np.outer(x, d_hid); self.b1 -= lr * d_hid

def Es_mse(net, T): return 0.5 * sum((T[i] - net.predict(X[i])) ** 2 for i in range(4))
def Es_bce(net, T):
    eps = 1e-12; s = 0.0
    for i in range(4):
        y = np.clip(net.predict(X[i]), eps, 1 - eps)
        s += -(T[i] * np.log(y) + (1 - T[i]) * np.log(1 - y))
    return s

CONFIGS = {
    "А (MSE)": dict(loss="mse", T=T_A, Es=Es_mse, Ee=EE_MSE, real=to_real_A),
    "Б (BCE)": dict(loss="bce", T=T_B, Es=Es_bce, Ee=EE_BCE, real=to_real_B),
}

def train(cfg, seed):
    rng = np.random.default_rng(seed)
    net = MLP(rng); hist = []
    for ep in range(1, MAX_EPOCHS + 1):
        for i in rng.permutation(4):               # онлайн-режим
            net.step(X[i], cfg["T"][i], LR, cfg["loss"])
        Es = float(cfg["Es"](net, cfg["T"])); hist.append(Es)
        if Es <= cfg["Ee"]:
            return net, ep, Es, True, hist
    return net, MAX_EPOCHS, Es, False, hist

def metrics(net, cfg):
    outs = np.array([cfg["real"](net.predict(X[i])) for i in range(4)])
    mae = float(np.mean(np.abs(outs - Y_REAL)))
    acc = np.mean([(abs(o - C0) < abs(o - C1)) == (y == C0) for o, y in zip(outs, Y_REAL)])
    return mae, float(acc)

def classify(y_real): return "c0" if abs(y_real - C0) < abs(y_real - C1) else "c1"

# ---------- эксперименты ----------
results = {k: [] for k in CONFIGS}
for name, cfg in CONFIGS.items():
    for s in SEEDS:
        net, ep, Es, ok, hist = train(cfg, s)
        mae, acc = metrics(net, cfg)
        results[name].append(dict(seed=s, net=net, ep=ep, Es=Es, ok=ok, hist=hist, mae=mae, acc=acc))

print(f"Ee: MSE = {EE_MSE}, BCE = {EE_BCE}; lr = {LR}; max эпох = {MAX_EPOCHS}\n")
for name, runs in results.items():
    print(f"=== Конфигурация {name} ===")
    print(f"{'seed':>4} {'эпох':>7} {'Es':>10} {'сошлась':>8} {'accuracy':>9} {'MAE':>8}")
    for r in runs:
        print(f"{r['seed']:>4} {r['ep']:>7} {r['Es']:>10.5f} {('да' if r['ok'] else 'НЕТ'):>8} {r['acc']*100:>8.0f}% {r['mae']:>8.3f}")
    ok_ep = [r["ep"] for r in runs if r["ok"]]
    print(f"Сошлось: {len(ok_ep)} из {len(runs)}", end="")
    if ok_ep: print(f"; эпох: min={min(ok_ep)}, max={max(ok_ep)}, среднее={np.mean(ok_ep):.0f}, std={np.std(ok_ep):.0f}")
    else: print()
    print()

def representative(runs):
    ok = [r for r in runs if r["ok"]]
    pool = ok if ok else runs
    return sorted(pool, key=lambda r: r["ep"])[len(pool) // 2]       # медианный по эпохам
rep = {k: representative(v) for k, v in results.items()}
print("Представительные запуски:", {k: f"seed={r['seed']}, эпох={r['ep']}" for k, r in rep.items()})

# ---------- визуализация ----------
colors = {"А (MSE)": "tab:blue", "Б (BCE)": "tab:red"}

fig, ax = plt.subplots(figsize=(8, 5))
for k, r in rep.items():
    ax.plot(range(1, len(r["hist"]) + 1), r["hist"], color=colors[k], label=f"{k}, seed={r['seed']}")
    ax.axhline(CONFIGS[k]["Ee"], color=colors[k], ls=":", alpha=.7, label=f"Ee {k} = {CONFIGS[k]['Ee']}")
ax.set_yscale("log"); ax.set_xscale("log")
ax.set_xlabel("Эпоха"); ax.set_ylabel("Суммарная ошибка Es (лог. шкала)")
ax.set_title("График сходимости: MSE vs BCE"); ax.legend(); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig("fig1_convergence.png", dpi=150)

fig, ax = plt.subplots(figsize=(8, 5)); w = 0.38
for j, (k, runs) in enumerate(results.items()):
    xs = np.arange(len(runs)) + (j - 0.5) * w
    for x, r in zip(xs, runs):
        ax.bar(x, r["ep"], w, color=colors[k], hatch=None if r["ok"] else "//", edgecolor="black",
               alpha=1 if r["ok"] else .45)
        ax.text(x, r["ep"], str(r["ep"]) + ("" if r["ok"] else "\nне сошлась"), ha="center", va="bottom", fontsize=7)
ax.set_xticks(np.arange(len(SEEDS))); ax.set_xticklabels([f"seed {s}" for s in SEEDS])
ax.set_yscale("log"); ax.set_ylabel("Число эпох (лог. шкала)"); ax.set_xlabel("Запуск")
ax.set_title("Устойчивость сходимости (штриховка — критерий не достигнут)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=colors[k], label=k) for k in results] +
          [Patch(facecolor="white", edgecolor="black", hatch="//", label="не сошлась")])
ax.grid(alpha=.3, axis="y"); fig.tight_layout(); fig.savefig("fig2_epochs.png", dpi=150)

fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
g = np.linspace(-10, 10, 200); GA, GB = np.meshgrid(g, g)
grid = np.stack([GA / 10, GB / 10], axis=-1)
lo, hi = min(C0, C1), max(C0, C1)
for ax, (k, r) in zip(axs[:2], rep.items()):
    net = r["net"]
    H = sigmoid(grid @ net.W1 + net.b1); Z = sigmoid(H @ net.W2 + net.b2)
    Zr = np.clip(CONFIGS[k]["real"](Z), lo, hi)                  # единая шкала [c0; c1] для обеих конфигураций
    cs = ax.contourf(GA, GB, Zr, levels=np.linspace(lo, hi, 23), cmap="RdYlBu", vmin=lo, vmax=hi)
    ax.contour(GA, GB, Zr, levels=[(C0 + C1) / 2], colors="k", linewidths=1.5)
    for a, b, y in DATA:
        ax.scatter(a, b, s=140, c="tab:green" if y == C0 else "tab:purple",
                   edgecolors="k", zorder=3, marker="o" if y == C0 else "s")
    ax.set_xlabel("A"); ax.set_ylabel("B"); ax.set_title(f"Конфигурация {k}, seed={r['seed']}")
    fig.colorbar(cs, ax=ax, label="выход в шкале [c0; c1]")
axs[0].legend(handles=[plt.Line2D([], [], marker="o", ls="", color="tab:green", mec="k", label=f"класс c0 ({C0})"),
                       plt.Line2D([], [], marker="s", ls="", color="tab:purple", mec="k", label=f"класс c1 ({C1})"),
                       plt.Line2D([], [], color="k", label="граница классов")], loc="upper right", fontsize=8)
mae_mean = {k: np.mean([r["mae"] for r in v if r["ok"]]) for k, v in results.items()}   # только сошедшиеся запуски
bars = axs[2].bar(list(mae_mean), list(mae_mean.values()), color=[colors[k] for k in mae_mean])
for b_, v in zip(bars, mae_mean.values()): axs[2].text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
axs[2].set_ylabel("Средняя абсолютная ошибка в шкале [c0; c1]"); axs[2].set_title("MAE в шкале [c0; c1]\n(среднее по сошедшимся запускам)")
fig.tight_layout(); fig.savefig("fig3_surface_mae.png", dpi=150)

# ---------- режим функционирования ----------
def show(net, cfg, a, b):
    yh = float(net.predict(np.array([a / 10, b / 10])))
    yn = (yh - T_LO) / (T_HI - T_LO) if cfg["loss"] == "mse" else yh   # к шкале (0;1) для сравнимости
    yr = cfg["real"](yh)
    print(f"A={a:>6}, B={b:>6} | ŷ(0;1)={yn:.4f} | в шкале [c0;c1]: {yr:8.4f} | ближе к {classify(yr)}")

net_B, cfg_B = rep["Б (BCE)"]["net"], CONFIGS["Б (BCE)"]
print("\n--- Режим функционирования (конфигурация Б, BCE) ---")
print("Обучающие примеры:")
for a, b, _ in DATA: show(net_B, cfg_B, a, b)
print("Дополнительные пары:")
for a, b in [(5, -2), (9, 9), (-6, 4), (0, 0)]: show(net_B, cfg_B, a, b)

print("\nДалее ввод своих значений. Пустая строка — выход.")
while True:
    try: s = input("Введите A и B из [-10; 10] через пробел: ").strip()
    except EOFError: break
    if not s: break
    try:
        a, b = map(float, s.split()); assert -10 <= a <= 10 and -10 <= b <= 10
    except Exception: print("Некорректный ввод."); continue
    show(net_B, cfg_B, a, b)

import os
if os.environ.get("MPLBACKEND", "").lower() != "agg": plt.show()