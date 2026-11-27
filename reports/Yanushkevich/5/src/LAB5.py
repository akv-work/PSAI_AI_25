import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LETTERS = ["Я", "Н", "У"]
ROWS, COLS = 7, 5
H_VAR = 5
LR = 0.2
M_MAX = 6
EE = 0.05
MAX_EPOCHS = 5000
W_INIT = 0.5
SEEDS = [1, 2, 3, 4, 5]
TAU = 0.6
NOISE_REPEATS = 50
TRAIN_IDS = range(1, 8)
TEST_IDS = range(8, 11)

BASE_DIR = Path(__file__).resolve().parent


def read_dataset(path):
    items = []
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        header = lines[i].split()
        cls, num = int(header[0]), int(header[1])
        grid = lines[i + 1:i + 1 + ROWS]
        if len(grid) != ROWS or any(len(g) != COLS or set(g) - {"#", "."} for g in grid):
            raise ValueError(f"Ошибка в сетке: класс {cls}, экземпляр {num}")
        items.append(dict(cls=cls, num=num, grid=grid, x=grid_to_vector(grid)))
        i += 1 + ROWS
    return items


def grid_to_vector(grid):
    return np.array([1.0 if ch == "#" else 0.0 for row in grid for ch in row])


def vector_to_grid(x):
    return ["".join("#" if x[r * COLS + c] > 0.5 else "." for c in range(COLS)) for r in range(ROWS)]


def one_hot(cls, n_classes=3):
    y = np.zeros(n_classes)
    y[cls - 1] = 1.0
    return y


def hamming(a, b):
    return int(np.sum(a != b))


def dataset_quality(items):
    n_fill = int(sum(it["x"].sum() for it in items))
    by_class = {c: [it["x"] for it in items if it["cls"] == c] for c in range(1, 4)}
    intra = {c: float(np.mean([hamming(a, b) for a, b in itertools.combinations(v, 2)])) for c, v in by_class.items()}
    inter = {}
    for c1, c2 in itertools.combinations(range(1, 4), 2):
        inter[(c1, c2)] = float(np.mean([hamming(a, b) for a in by_class[c1] for b in by_class[c2]]))
    all_intra = [hamming(a, b) for v in by_class.values() for a, b in itertools.combinations(v, 2)]
    all_inter = [hamming(a, b) for c1, c2 in itertools.combinations(range(1, 4), 2)
                 for a in by_class[c1] for b in by_class[c2]]
    return n_fill, intra, inter, float(np.mean(all_intra)), float(np.mean(all_inter))


def split(items):
    train = [it for it in items if it["num"] in TRAIN_IDS]
    test = [it for it in items if it["num"] in TEST_IDS]
    to_xy = lambda lst: (np.array([it["x"] for it in lst]), np.array([one_hot(it["cls"]) for it in lst]))
    return train, test, to_xy(train), to_xy(test)


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def softmax(s):
    z = s - np.max(s)
    e = np.exp(z)
    return e / e.sum()


class MLP:
    def __init__(self, n_in, n_hidden, n_out, lr, w_init, seed):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = rng.uniform(-w_init, w_init, size=(n_in, n_hidden))
        self.T = rng.uniform(-w_init, w_init, size=n_hidden)
        self.V = rng.uniform(-w_init, w_init, size=(n_hidden, n_out))
        self.Tv = rng.uniform(-w_init, w_init, size=n_out)

    def n_params(self):
        return self.W.size + self.T.size + self.V.size + self.Tv.size

    def forward(self, x):
        h = sigmoid(x @ self.W - self.T)
        p = softmax(h @ self.V - self.Tv)
        return h, p

    def backward(self, x, y, h, p):
        g_out = p - y
        g_hidden = (self.V @ g_out) * h * (1.0 - h)
        return dict(W=np.outer(x, g_hidden), T=-g_hidden, V=np.outer(h, g_out), Tv=-g_out)

    def update(self, grads):
        self.W -= self.lr * grads["W"]
        self.T -= self.lr * grads["T"]
        self.V -= self.lr * grads["V"]
        self.Tv -= self.lr * grads["Tv"]

    def train_step(self, x, y):
        h, p = self.forward(x)
        self.update(self.backward(x, y, h, p))

    def predict_proba(self, X):
        return np.array([self.forward(x)[1] for x in X])


def cross_entropy(p, y):
    return float(-np.sum(y * np.log(np.clip(p, 1e-12, 1.0))))


def errors(model, X, Y):
    P = model.predict_proba(X)
    es = float(np.mean(0.5 * np.sum((P - Y) ** 2, axis=1)))
    ce = float(np.mean([cross_entropy(p, y) for p, y in zip(P, Y)]))
    return es, ce


def train(model, X, Y, eps=EE, max_epochs=MAX_EPOCHS, seed=1):
    rng = np.random.default_rng(seed)
    order = np.arange(len(X))
    hist_es, hist_ce = [], []
    for epoch in range(1, max_epochs + 1):
        rng.shuffle(order)
        for i in order:
            model.train_step(X[i], Y[i])
        es, ce = errors(model, X, Y)
        hist_es.append(es)
        hist_ce.append(ce)
        if es <= eps:
            return epoch, es, ce, hist_es, hist_ce
    return max_epochs, es, ce, hist_es, hist_ce


def accuracy(model, X, Y):
    P = model.predict_proba(X)
    return float(np.mean(np.argmax(P, axis=1) == np.argmax(Y, axis=1)))


def confusion_matrix(model, X, Y, n_classes=3):
    M = np.zeros((n_classes, n_classes), dtype=int)
    P = model.predict_proba(X)
    for t, pr in zip(np.argmax(Y, axis=1), np.argmax(P, axis=1)):
        M[t, pr] += 1
    return M


def confidence_stats(model, X, Y):
    P = model.predict_proba(X)
    conf = P.max(axis=1)
    ok = np.argmax(P, axis=1) == np.argmax(Y, axis=1)
    right = float(conf[ok].mean()) if ok.any() else None
    wrong = float(conf[~ok].mean()) if (~ok).any() else None
    return right, wrong


def add_noise(x, m, rng):
    x_noisy = x.copy()
    if m > 0:
        idx = rng.choice(len(x), size=m, replace=False)
        x_noisy[idx] = 1.0 - x_noisy[idx]
    return x_noisy


def noise_test(model, X, Y, m_max=M_MAX, repeats=NOISE_REPEATS, seed=2026):
    rng = np.random.default_rng(seed)
    true = np.argmax(Y, axis=1)
    overall, per_class = [], []
    for m in range(m_max + 1):
        hits = np.zeros(len(X))
        for i, x in enumerate(X):
            for _ in range(repeats):
                p = model.forward(add_noise(x, m, rng))[1]
                hits[i] += int(np.argmax(p) == true[i])
        acc_i = hits / repeats
        overall.append(float(acc_i.mean()))
        per_class.append([float(acc_i[true == c].mean()) for c in range(3)])
    return np.array(overall), np.array(per_class)


def numerical_gradient(model, x, y, eps=1e-6):
    num = {}
    for name in ("W", "T", "V", "Tv"):
        arr = getattr(model, name)
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + eps
            lp = cross_entropy(model.forward(x)[1], y)
            arr[idx] = old - eps
            lm = cross_entropy(model.forward(x)[1], y)
            arr[idx] = old
            g[idx] = (lp - lm) / (2 * eps)
        num[name] = g
    return num


def gradient_check(model, x, y):
    h, p = model.forward(x)
    ana = model.backward(x, y, h, p)
    num = numerical_gradient(model, x, y)
    res = {}
    for k in ana:
        diff = np.max(np.abs(ana[k] - num[k]))
        rel = np.linalg.norm(ana[k] - num[k]) / max(np.linalg.norm(ana[k]) + np.linalg.norm(num[k]), 1e-12)
        res[k] = (float(diff), float(rel))
    return res


def decide(p, tau=TAU):
    k = int(np.argmax(p))
    return (LETTERS[k] if p[k] >= tau else None), float(p[k])


def rejection_report(model, X, true_cls, tau=TAU):
    P = model.predict_proba(X)
    rejected = P.max(axis=1) < tau
    would_be_right = rejected & (np.argmax(P, axis=1) == np.array(true_cls))
    return int(rejected.sum()), int(would_be_right.sum()), len(X)


def print_grid(grid, indent="    "):
    for row in grid:
        print(indent + " ".join(row))


def describe(model, grid, label, true_letter=None):
    x = grid_to_vector(grid)
    p = model.forward(x)[1]
    k = int(np.argmax(p))
    dec, conf = decide(p)
    print(f"\n  {label}")
    print_grid(grid)
    print(f"    p({LETTERS[0]})={p[0]:.4f}  p({LETTERS[1]})={p[1]:.4f}  p({LETTERS[2]})={p[2]:.4f}")
    print(f"    класс: {LETTERS[k]}   уверенность: {conf:.4f}   решение с порогом τ={TAU}: "
          f"{dec if dec else 'не распознано'}" + (f"   (истинная буква: {true_letter})" if true_letter else ""))


def interactive_mode(model):
    print("\nРежим ввода: 7 строк по 5 символов (# - закрашено, . - пусто); q - выход")
    while True:
        grid = []
        while len(grid) < ROWS:
            line = input(f"строка {len(grid) + 1}: ").strip()
            if line.lower() == "q":
                return
            if len(line) != COLS or set(line) - {"#", "."}:
                print("  нужно ровно 5 символов из # и .")
                continue
            grid.append(line)
        describe(model, grid, "введённая сетка")


def title(text, n_fill):
    return f"{text}\nN_fill = {n_fill}"


def main():
    items = read_dataset(BASE_DIR / "dataset.txt")
    print(f"Прочитано объектов: {len(items)} (классы: {', '.join(f'{i + 1} - {c}' for i, c in enumerate(LETTERS))})")

    n_fill, intra, inter, intra_all, inter_all = dataset_quality(items)
    print("\nКонтроль качества датасета")
    print(f"N_fill (всего закрашенных клеток) = {n_fill}")
    for c in range(1, 4):
        mark = "  (меньше 3 - экземпляры слишком похожи!)" if intra[c] < 3 else ""
        print(f"среднее расстояние Хэмминга внутри класса {LETTERS[c - 1]}: {intra[c]:.2f}{mark}")
    for (c1, c2), v in inter.items():
        print(f"среднее расстояние Хэмминга между {LETTERS[c1 - 1]} и {LETTERS[c2 - 1]}: {v:.2f}")
    print(f"в целом: внутри классов {intra_all:.2f}, между классами {inter_all:.2f}")

    train_items, test_items, (Xtr, Ytr), (Xte, Yte) = split(items)
    print(f"\nОбучающая выборка: {len(Xtr)} объектов (экз. 1-7), тестовая: {len(Xte)} объектов (экз. 8-10)")

    probe = MLP(35, H_VAR, 3, LR, W_INIT, seed=1)
    gc = gradient_check(probe, Xtr[0], Ytr[0])
    print("\nПроверка градиентов (аналитический vs численный, центральная разность, ε=1e-6)")
    for k, (d, r) in gc.items():
        print(f"  {k:>2s}: макс. абс. разница = {d:.2e}, относительная ошибка = {r:.2e}")

    print(f"\nБазовый эксперимент: 35-{H_VAR}-3, η={LR}, Ee={EE}, seed=1")
    base = MLP(35, H_VAR, 3, LR, W_INIT, seed=1)
    ep, es, ce, hist_es, hist_ce = train(base, Xtr, Ytr, seed=1)
    print(f"эпох: {ep}, Es = {es:.4f}, CE = {ce:.4f}, критерий {'достигнут' if es <= EE else 'НЕ достигнут'}")
    acc_tr, acc_te = accuracy(base, Xtr, Ytr), accuracy(base, Xte, Yte)
    print(f"accuracy: обучающая {acc_tr:.1%}, тестовая {acc_te:.1%}")
    for name, X, Y in (("обучающая", Xtr, Ytr), ("тестовая", Xte, Yte)):
        M = confusion_matrix(base, X, Y)
        print(f"матрица ошибок ({name}), строки - истинный класс, столбцы - предсказанный:")
        print("       " + "  ".join(f"{c:>3s}" for c in LETTERS))
        for i, row in enumerate(M):
            print(f"    {LETTERS[i]}  " + "  ".join(f"{v:>3d}" for v in row))
    right, wrong = confidence_stats(base, Xte, Yte)
    print(f"средняя уверенность на тесте: верные {right:.4f}" if right is not None else "верных на тесте нет", end="")
    print(f", неверные {wrong:.4f}" if wrong is not None else ", неверно классифицированных нет")
    P_te = base.predict_proba(Xte)
    for it, p in zip(test_items, P_te):
        k = int(np.argmax(p))
        flag = "" if k == it["cls"] - 1 else "  <- ошибка"
        print(f"  {LETTERS[it['cls'] - 1]}{it['num']}: p = [{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}] -> {LETTERS[k]}{flag}")

    print(f"\nИсследование скрытого слоя (5 запусков, seed {SEEDS[0]}..{SEEDS[-1]})")
    hs = [2, H_VAR, 2 * H_VAR]
    study = {}
    for H in hs:
        runs = []
        for s in SEEDS:
            m = MLP(35, H, 3, LR, W_INIT, seed=s)
            e, es_, ce_, he, hc = train(m, Xtr, Ytr, seed=s)
            runs.append(dict(epochs=e, conv=es_ <= EE, acc=accuracy(m, Xte, Yte), he=he, hc=hc, params=m.n_params()))
        study[H] = runs
    print(f"{'H':>4s}{'параметров':>12s}{'эпохи ср.':>11s}{'эпохи min-max':>16s}{'acc test ср.':>14s}"
          f"{'acc min-max':>15s}{'не сошлось':>12s}")
    for H in hs:
        r = study[H]
        ep_ = [x["epochs"] for x in r]
        ac_ = [x["acc"] for x in r]
        print(f"{H:>4d}{r[0]['params']:>12d}{np.mean(ep_):>11.1f}{f'{min(ep_)}-{max(ep_)}':>16s}"
              f"{np.mean(ac_):>14.1%}{f'{min(ac_):.1%}-{max(ac_):.1%}':>15s}{sum(not x['conv'] for x in r):>10d}/5")

    print(f"\nУстойчивость к шуму (инверсия m клеток, {NOISE_REPEATS} зашумлений на пример)")
    acc_m, acc_m_cls = noise_test(base, Xte, Yte)
    print(f"{'m':>3s}{'общая':>9s}" + "".join(f"{c:>9s}" for c in LETTERS))
    for m in range(M_MAX + 1):
        print(f"{m:>3d}{acc_m[m]:>9.1%}" + "".join(f"{v:>9.1%}" for v in acc_m_cls[m]))
    below = np.where(acc_m < 0.8)[0]
    print(f"accuracy впервые ниже 80% при m = {below[0]}" if len(below) else
          f"accuracy не опускается ниже 80% вплоть до m = {M_MAX}")

    print("\nКарты весов скрытых нейронов (базовая сеть)")
    for j in range(H_VAR):
        w = base.W[:, j]
        v = base.V[j]
        top = np.argsort(-np.abs(w))[:3]
        cells = ", ".join(f"({t // COLS + 1},{t % COLS + 1}) {w[t]:+.2f}" for t in top)
        signs = ", ".join(f"v({LETTERS[k]})={v[k]:+.2f}" for k in range(3))
        print(f"  нейрон {j + 1}: {signs}; работает на букву {LETTERS[int(np.argmax(v))]}; "
              f"наибольшие |w| в клетках (строка,столбец): {cells}")

    conf_te = P_te.max(axis=1)
    unsure = np.argsort(conf_te)[:3]
    print("\nТри самых неуверенных тестовых примера")
    for i in unsure:
        it = test_items[i]
        p = P_te[i]
        print(f"  {LETTERS[it['cls'] - 1]}{it['num']}: p = [{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}], "
              f"уверенность {conf_te[i]:.4f}, предсказано {LETTERS[int(np.argmax(p))]}")

    alien_path = BASE_DIR / "alien.txt"
    alien = read_dataset(alien_path) if alien_path.exists() else None

    print("\n«Чужая» буква")
    if alien:
        pa = base.forward(alien[0]["x"])[1]
        dec, conf = decide(pa)
        print(f"  p = [{pa[0]:.4f} {pa[1]:.4f} {pa[2]:.4f}] -> {LETTERS[int(np.argmax(pa))]} "
              f"с вероятностью {conf:.4f}; с порогом τ={TAU}: {dec if dec else 'не распознано'}")
    else:
        print("  файл alien.txt не найден - пункт пропущен")

    print(f"\nПорог отказа τ = {TAU}")
    rep = rejection_report(base, Xte, np.argmax(Yte, axis=1))
    print(f"  тестовая выборка: отвергнуто {rep[0]} из {rep[2]}, из них были бы верными {rep[1]}")
    if alien:
        rej = base.forward(alien[0]["x"])[1].max() < TAU
        print(f"  «чужая» буква: {'отвергнута' if rej else 'НЕ отвергнута (принята за букву набора)'}")

    colors = {2: "#dc2626", H_VAR: "#2563eb", 2 * H_VAR: "#16a34a"}
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for H in hs:
        r = study[H][0]
        ep_axis = np.arange(1, len(r["he"]) + 1)
        ax.plot(ep_axis, r["he"], color=colors[H], lw=1.6, label=f"Es, H={H}")
        ax.plot(ep_axis, r["hc"], color=colors[H], lw=1.2, ls="--", label=f"CE, H={H}")
    ax.axhline(EE, color="gray", ls=":", label=f"Ee = {EE}")
    ax.set_yscale("log")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Ошибка")
    ax.set_title(title("Сходимость Es и CE для разных H (seed = 1)", n_fill))
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "convergence.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pos, H in enumerate(hs):
        r = study[H]
        jitter = np.linspace(-0.12, 0.12, len(r))
        for jx, x in zip(jitter, r):
            ax.scatter(pos + jx, x["epochs"], s=60, color=colors[H] if x["conv"] else "white",
                       edgecolor=colors[H], linewidth=1.5, zorder=3)
        ax.hlines(np.mean([x["epochs"] for x in r]), pos - 0.25, pos + 0.25, color="black", lw=1.2)
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"H = {H}" for H in hs])
    ax.set_ylabel("Эпох до критерия остановки")
    ax.set_title(title("Разброс числа эпох по 5 запускам (черта - среднее)", n_fill))
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(BASE_DIR / "epochs_spread.png", dpi=150)
    plt.close(fig)

    M = confusion_matrix(base, Xte, Yte)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=max(3, M.max()))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(3))
    ax.set_xticklabels(LETTERS)
    ax.set_yticks(range(3))
    ax.set_yticklabels(LETTERS)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title("Матрица ошибок на тестовой выборке", n_fill), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "confusion_test.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ms = np.arange(M_MAX + 1)
    ax.plot(ms, acc_m * 100, color="black", lw=2.2, marker="o", label="общая")
    for c, col in zip(range(3), ("#2563eb", "#dc2626", "#16a34a")):
        ax.plot(ms, acc_m_cls[:, c] * 100, color=col, lw=1.4, marker="s", ms=4, ls="--", label=f"буква {LETTERS[c]}")
    ax.axhline(80, color="gray", ls=":", label="80 %")
    ax.set_xlabel("Число инвертированных клеток m")
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(-3, 103)
    ax.set_title(title("Устойчивость к шуму", n_fill))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASE_DIR / "noise.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, H_VAR, figsize=(2.3 * H_VAR, 3.6))
    vmax = np.abs(base.W).max()
    for j, ax in enumerate(np.atleast_1d(axes)):
        im = ax.imshow(base.W[:, j].reshape(ROWS, COLS), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"нейрон {j + 1}\n→ {LETTERS[int(np.argmax(base.V[j]))]}", fontsize=10)
        ax.set_xticks(range(COLS))
        ax.set_xticklabels(range(1, COLS + 1), fontsize=7)
        ax.set_yticks(range(ROWS))
        ax.set_yticklabels(range(1, ROWS + 1), fontsize=7)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle(title("Карты весов скрытых нейронов (красный - вес > 0, синий - < 0)", n_fill), fontsize=10)
    fig.savefig(BASE_DIR / "weight_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    for ax, i in zip(axes, unsure):
        it = test_items[i]
        p = P_te[i]
        ax.imshow(it["x"].reshape(ROWS, COLS), cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
        ax.grid(which="minor", color="#999999", lw=0.5)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(f"{LETTERS[it['cls'] - 1]}{it['num']} → {LETTERS[int(np.argmax(p))]}\n"
                     + "\n".join(f"p({LETTERS[k]}) = {p[k]:.4f}" for k in range(3)), fontsize=9)
    fig.suptitle(title("Три самых неуверенных тестовых примера", n_fill), fontsize=10)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "unsure_examples.png", dpi=150)
    plt.close(fig)

    print("\nГрафики сохранены: convergence.png, epochs_spread.png, confusion_test.png, noise.png, "
          "weight_maps.png, unsure_examples.png")

    print("\nРежим функционирования (базовая сеть)")
    by_key = {(it["cls"], it["num"]): it for it in items}
    for c in range(1, 4):
        describe(base, by_key[(c, 1)]["grid"], f"обучающий пример {LETTERS[c - 1]}1", LETTERS[c - 1])
    for c in range(1, 4):
        describe(base, by_key[(c, 8)]["grid"], f"тестовый пример {LETTERS[c - 1]}8", LETTERS[c - 1])
    rng = np.random.default_rng(7)
    for c, m in ((1, 3), (3, 6)):
        src = by_key[(c, 9)]
        noisy = add_noise(src["x"], m, rng)
        describe(base, vector_to_grid(noisy), f"искажённый {LETTERS[c - 1]}9 (изменено клеток: {m})", LETTERS[c - 1])
    if alien:
        describe(base, alien[0]["grid"], "«чужая» буква")

    interactive_mode(base)


if __name__ == "__main__":
    main()