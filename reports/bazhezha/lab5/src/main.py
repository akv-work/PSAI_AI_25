import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-mrzis-lab5")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LETTERS = ("Б", "А", "Ж")
CLASS = {"1": 0, "2": 1, "3": 2, "Б": 0, "А": 1, "Ж": 2}
H_BASE, ETA, MMAX = 3, 0.10, 6
EE, EPOCHS, TAU = 0.05, 5000, 0.6
SEEDS = (1, 2, 3, 4, 5)
H_LIST = (2, H_BASE, 2 * H_BASE)
ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"


class Log:
    def __init__(self):
        self.buf = []

    def __call__(self, msg=""):
        print(msg)
        self.buf.append(str(msg))

    def save(self, path):
        Path(path).write_text("\n".join(self.buf) + "\n", encoding="utf-8")


def is_grid_row(s):
    return len(s) == 5 and set(s) <= set(".#")


def read_dataset(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    items, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or (line.startswith("#") and not is_grid_row(line)):
            continue
        cls, inst = line.split()
        rows = []
        while len(rows) < 7:
            s = lines[i].strip()
            i += 1
            if is_grid_row(s):
                rows.append(s)
        vec = []
        for r, s in enumerate(rows):
            if len(s) != 5 or any(c not in ".#" for c in s):
                raise ValueError(f"{cls} {inst}, строка {r + 1}: {s}")
            vec.extend(0.0 if c == "." else 1.0 for c in s)
        items.append({"cls": cls, "inst": int(inst), "x": np.array(vec), "rows": rows})
    return items


def rows_to_vec(rows):
    if len(rows) != 7:
        raise ValueError("нужно 7 строк")
    vec, clean = [], []
    for s in rows:
        s = s.strip()
        if len(s) != 5 or any(c not in ".#" for c in s):
            raise ValueError(f"строка из 5 символов # и .: {s}")
        clean.append(s)
        vec.extend(0.0 if c == "." else 1.0 for c in s)
    return np.array(vec), clean


def vec_to_rows(x):
    g = np.asarray(x, float).reshape(7, 5)
    return ["".join("#" if v >= 0.5 else "." for v in row) for row in g]


def one_hot(y, k=3):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def split_xy(items):
    X = np.vstack([it["x"] for it in items])
    y = np.array([CLASS[it["cls"]] for it in items])
    inst = np.array([it["inst"] for it in items])
    return X, y, inst, inst <= 7, inst >= 8


def pairwise(A):
    d = [np.sum(A[i] != A[j]) for i in range(len(A)) for j in range(i + 1, len(A))]
    return float(np.mean(d)), float(np.min(d))


def between(A, B):
    return float(np.mean([np.sum(a != b) for a in A for b in B]))


def quality_control(X, y, log):
    nfill = int(X.sum())
    log(f"Nfill = {nfill}")
    within = []
    for c, name in enumerate(LETTERS):
        mean, mn = pairwise(X[y == c])
        within.append(mean)
        log(f"  внутри {name}: среднее {mean:.2f}, минимум {mn:.0f}")
    cross = []
    for i in range(3):
        for j in range(i + 1, 3):
            d = between(X[y == i], X[y == j])
            cross.append(d)
            log(f"  между {LETTERS[i]} и {LETTERS[j]}: {d:.2f}")
    log(f"внутриклассовое среднее = {np.mean(within):.2f}")
    log(f"межклассовое среднее = {np.mean(cross):.2f}")
    if min(within) < 3:
        log("ПРЕДУПРЕЖДЕНИЕ: внутриклассовое расстояние < 3, экземпляры слишком похожи")
    return nfill


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -60.0, 60.0)))


def softmax(z):
    z = np.asarray(z, float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def init_net(h, rng):
    return {
        "W1": rng.uniform(-0.5, 0.5, (35, h)),
        "T1": rng.uniform(-0.5, 0.5, h),
        "W2": rng.uniform(-0.5, 0.5, (h, 3)),
        "T2": rng.uniform(-0.5, 0.5, 3),
    }


def forward(net, x):
    x = np.asarray(x, float)
    h = sigmoid(x @ net["W1"] - net["T1"])
    p = softmax(h @ net["W2"] - net["T2"])
    net["cache"] = (x, h, p)
    return p


def backward(net, y):
    x, h, p = net["cache"]
    dz = p - np.asarray(y, float)
    dW2 = np.outer(h, dz)
    dT2 = -dz
    ds = (net["W2"] @ dz) * h * (1.0 - h)
    dW1 = np.outer(x, ds)
    dT1 = -ds
    return dW1, dT1, dW2, dT2


def apply_grad(net, grads):
    for key, g in zip(("W1", "T1", "W2", "T2"), grads):
        net[key] -= ETA * g


def cross_entropy(p, y):
    return float(-np.sum(np.asarray(y) * np.log(np.clip(p, 1e-12, 1.0))))


def mean_errors(net, X, Y):
    es = ce = 0.0
    for x, y in zip(X, Y):
        p = forward(net, x)
        es += 0.5 * np.sum((p - y) ** 2)
        ce += cross_entropy(p, y)
    n = len(X)
    return es / n, ce / n


def train(X, Y, h, seed, keep_hist=False):
    rng = np.random.default_rng(seed)
    net = init_net(h, rng)
    idx = np.arange(len(X))
    hist, es, ce, stopped = [], None, None, False
    ep_done = 0
    for ep in range(1, EPOCHS + 1):
        rng.shuffle(idx)
        for i in idx:
            forward(net, X[i])
            apply_grad(net, backward(net, Y[i]))
        es, ce = mean_errors(net, X, Y)
        ep_done = ep
        if keep_hist:
            hist.append((es, ce))
        if not np.isfinite(es) or es <= EE:
            stopped = bool(np.isfinite(es) and es <= EE)
            break
    return {"net": net, "epochs": ep_done, "es": es, "ce": ce,
            "stopped": stopped, "hist": hist}


def predict(net, X):
    P = np.vstack([forward(net, x) for x in X])
    return np.argmax(P, axis=1), P


def confusion_matrix(y_true, y_pred, k=3):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def fmt_cm(cm):
    head = "     " + " ".join(f"{n:>3}" for n in LETTERS)
    rows = [head]
    for i, name in enumerate(LETTERS):
        rows.append(f"{name:>3}  " + " ".join(f"{int(cm[i, j]):3d}" for j in range(3)))
    return "\n".join(rows)


def n_params(h):
    return 36 * h + 3 * (h + 1)


def numerical_gradient(net, x, y, eps=1e-5):
    grads = []
    for key in ("W1", "T1", "W2", "T2"):
        A = net[key]
        G = np.zeros_like(A)
        for idx in np.ndindex(A.shape):
            orig = A[idx]
            A[idx] = orig + eps
            lp = cross_entropy(forward(net, x), y)
            A[idx] = orig - eps
            lm = cross_entropy(forward(net, x), y)
            A[idx] = orig
            G[idx] = (lp - lm) / (2 * eps)
        grads.append(G)
    return grads


def grad_check(X, Y, log):
    rng = np.random.default_rng(0)
    net = init_net(H_BASE, rng)
    forward(net, X[0])
    analytic = backward(net, Y[0])
    numeric = numerical_gradient(net, X[0], Y[0])
    err = max(np.max(np.abs(a - n)) for a, n in zip(analytic, numeric))
    log(f"проверка градиента (макс. |аналит. − числ.|) = {err:.3e}")


def noise_curve(net, X, y, trials, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros(MMAX + 1)
    acc_c = np.zeros((3, MMAX + 1))
    for m in range(MMAX + 1):
        hit = np.zeros(3)
        tot = np.zeros(3)
        ok = 0
        for x, yi in zip(X, y):
            yi = int(yi)
            for _ in range(trials):
                xn = np.array(x, float, copy=True)
                if m:
                    idx = rng.choice(35, size=m, replace=False)
                    xn[idx] = 1.0 - xn[idx]
                pred = int(np.argmax(forward(net, xn)))
                tot[yi] += 1
                if pred == yi:
                    ok += 1
                    hit[yi] += 1
        acc[m] = ok / tot.sum()
        acc_c[:, m] = hit / tot
    return acc, acc_c


def rejection_stats(net, X, y):
    rej = rej_ok = 0
    for i, x in enumerate(X):
        p = forward(net, x)
        k = int(np.argmax(p))
        if p[k] < TAU:
            rej += 1
            if y is not None and k == int(y[i]):
                rej_ok += 1
    return rej, rej_ok, len(X)


def functioning_mode(net, rows, log=print, note=""):
    x, rows = rows_to_vec(rows)
    if note:
        log(note)
    log("Сетка:")
    for row in rows:
        log(row)
    p = forward(net, x)
    k = int(np.argmax(p))
    conf = float(p[k])
    log(f"p1={p[0]:.4f}  p2={p[1]:.4f}  p3={p[2]:.4f}")
    log(f"Класс: {LETTERS[k]}")
    log(f"Уверенность: {conf:.4f}")
    log("Решение: " + ("не распознано" if conf < TAU else LETTERS[k]))
    log("")
    return p


def title(nfill, text):
    return f"Бажежа, вариант 1, Nfill={nfill}\n{text}"


def plot_convergence(hists, nfill, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for H, hist in hists.items():
        es = np.clip([e for e, _ in hist], 1e-12, None)
        ce = np.clip([c for _, c in hist], 1e-12, None)
        t = np.arange(1, len(hist) + 1)
        ax.semilogy(t, es, color=f"C{H}", label=f"Es, H={H}")
        ax.semilogy(t, ce, color=f"C{H}", ls="--", label=f"CE, H={H}")
    ax.axhline(EE, color="0.4", ls=":", label="Ee = 0.05")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("ошибка")
    ax.set_title(title(nfill, "Сходимость Es и CE"))
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_spread(epoch_lists, nfill, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for i, ep in enumerate(epoch_lists):
        ep = np.asarray(ep, float)
        ax.plot([i, i], [ep.min(), ep.max()], color="0.45", lw=2, zorder=2)
        shift = np.linspace(-0.18, 0.18, len(ep))
        ax.scatter(i + shift, ep, s=42, zorder=3, label="запуски" if i == 0 else None)
        ax.scatter([i], [ep.mean()], marker="D", c="black", s=46, zorder=4,
                   label="среднее" if i == 0 else None)
    lo = min(float(np.min(ep)) for ep in epoch_lists)
    hi = max(float(np.max(ep)) for ep in epoch_lists)
    pad = max(0.8, 0.15 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks(range(3), [f"H={h}" for h in H_LIST])
    ax.set_ylabel("число эпох")
    ax.set_title(title(nfill, "Разброс числа эпох, 5 запусков"))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, nfill, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), LETTERS)
    ax.set_yticks(range(3), LETTERS)
    ax.set_xlabel("предсказанный класс")
    ax.set_ylabel("истинный класс")
    ax.set_title(title(nfill, "Матрица ошибок, тест"))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_noise(acc, acc_c, nfill, path):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    m = np.arange(MMAX + 1)
    ax.plot(m, acc, "o-", color="black", label="все классы")
    for c, name in enumerate(LETTERS):
        ax.plot(m, acc_c[c], "s--", label=name)
    ax.axhline(0.8, color="0.4", ls=":", label="80%")
    ax.set_xlabel("m, число инвертированных клеток")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.legend(loc="lower left")
    ax.set_xticks(m)
    ax.set_title(title(nfill, "Устойчивость к шуму"))
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_weights(net, nfill, path):
    W1 = net["W1"]
    vmax = float(np.max(np.abs(W1))) or 1.0
    H = W1.shape[1]
    fig, axes = plt.subplots(1, H, figsize=(3.1 * H, 4.4))
    im = None
    for j in range(H):
        ax = axes[j]
        im = ax.imshow(W1[:, j].reshape(7, 5), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"нейрон {j + 1}")
        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(7), range(1, 8))
        ax.set_xlabel("столбец")
        ax.set_ylabel("строка" if j == 0 else "")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03)
    fig.suptitle(title(nfill, "Карты весов скрытых нейронов"))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_uncertain(X, y, inst, P, order, nfill, path):
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 4.8))
    for ax, i in zip(axes, order[:3]):
        ax.imshow(X[i].reshape(7, 5), cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks(range(5), range(1, 6))
        ax.set_yticks(range(7), range(1, 8))
        p = P[i]
        ax.set_title(f"{LETTERS[int(y[i])]} №{int(inst[i])}")
        ax.set_xlabel(f"p1={p[0]:.2f}  p2={p[1]:.2f}  p3={p[2]:.2f}")
    fig.suptitle(title(nfill, "Три самых неуверенных тестовых примера"))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def neuron_report(net, Xtr, ytr, log):
    means = np.vstack([Xtr[ytr == c].mean(axis=0) for c in range(3)])
    scores = (2 * means - 1.0) @ net["W1"]
    for j in range(net["W1"].shape[1]):
        w = net["W1"][:, j]
        letter = LETTERS[int(np.argmax(scores[:, j]))]
        vote = LETTERS[int(np.argmax(net["W2"][j]))]
        top = np.argsort(np.abs(w))[::-1][:3]
        cells = ", ".join(
            f"({idx // 5 + 1},{idx % 5 + 1})={w[idx]:+.3f}" for idx in top
        )
        v = " ".join(f"v{k + 1}={net['W2'][j, k]:+.3f}" for k in range(3))
        log(f"нейрон {j + 1}: по знакам входа → {letter}; по выходу → {vote}")
        log(f"  наибольшие |w|: {cells}")
        log(f"  веса выхода {v}")


def find_item(items, cls, inst):
    for it in items:
        if it["cls"] == cls and it["inst"] == inst:
            return it
    raise KeyError((cls, inst))


def flip_cells(x, cells):
    z = np.array(x, float, copy=True)
    for r, c in cells:
        z[r * 5 + c] = 1.0 - z[r * 5 + c]
    return z


def fmt_range(vals, pct=False):
    a = np.asarray(vals, float)
    if pct:
        a = a * 100
        return f"{a.mean():.1f}% ({a.min():.1f}–{a.max():.1f})"
    return f"{a.mean():.1f} ({a.min():.0f}–{a.max():.0f})"


def mean_or_dash(xs, digits=4):
    return "—" if not xs else f"{float(np.mean(xs)):.{digits}f}"


def run_study(Xtr, Ytr, Xte, yte):
    runs = {}
    for H in H_LIST:
        for seed in SEEDS:
            runs[(H, seed)] = train(Xtr, Ytr, H, seed, keep_hist=(seed == SEEDS[0]))
            pred, _ = predict(runs[(H, seed)]["net"], Xte)
            runs[(H, seed)]["acc"] = float(np.mean(pred == yte))
    return runs


def report_base(run, Xtr, ytr, Xte, yte, inst_te, log):
    net = run["net"]
    pred_tr, _ = predict(net, Xtr)
    pred_te, Pte = predict(net, Xte)
    ok_tr = int(np.sum(pred_tr == ytr))
    ok_te = int(np.sum(pred_te == yte))
    cm_tr, cm_te = confusion_matrix(ytr, pred_tr), confusion_matrix(yte, pred_te)
    conf_ok, conf_bad = [], []
    for p, yi in zip(Pte, yte):
        (conf_ok if int(np.argmax(p)) == int(yi) else conf_bad).append(float(np.max(p)))
    log("--- базовый эксперимент: H=3, seed=1, η=0.10 ---")
    log(f"эпох = {run['epochs']}, сошлась = {'да' if run['stopped'] else 'нет'}")
    log(f"Es = {run['es']:.6f}, CE = {run['ce']:.6f}")
    log(f"accuracy обучение = {ok_tr / len(ytr):.3f} ({ok_tr}/{len(ytr)})")
    log(f"accuracy тест = {ok_te / len(yte):.3f} ({ok_te}/{len(yte)})")
    log("матрица ошибок, обучение (строки — истина, столбцы — ответ):")
    log(fmt_cm(cm_tr))
    log("матрица ошибок, тест:")
    log(fmt_cm(cm_te))
    log(f"средняя уверенность на верных тестовых = {mean_or_dash(conf_ok)}")
    log(f"средняя уверенность на неверных тестовых = {mean_or_dash(conf_bad)}")
    return cm_te, Pte


def report_table(runs, log):
    log("--- исследование H, 5 запусков (seeds 1..5) ---")
    log(f"{'H':>4} {'эпохи, среднее (min–max)':>28} {'тест accuracy':>22} {'не сошл.':>10} {'парам.':>8}")
    epoch_lists = []
    for H in H_LIST:
        pack = [runs[(H, s)] for s in SEEDS]
        epochs = [r["epochs"] for r in pack]
        accs = [r["acc"] for r in pack]
        fails = sum(not r["stopped"] for r in pack)
        epoch_lists.append(epochs)
        log(f"{H:4d} {fmt_range(epochs):>28} {fmt_range(accs, pct=True):>22} {fails:>6}/5 {n_params(H):8d}")
    log("несошедшийся запуск входит в разброс эпох как 5000")
    log("параметры = 36·H + 3·(H+1)")
    return epoch_lists


def report_noise(acc, acc_c, log):
    log("--- шум: 50 инверсий на каждый тестовый пример ---")
    for m in range(MMAX + 1):
        parts = "  ".join(f"{LETTERS[c]}={acc_c[c, m]:.3f}" for c in range(3))
        log(f"m={m}: accuracy={acc[m]:.3f}   {parts}")
    below = next((m for m, a in enumerate(acc) if a < 0.8), None)
    if below is None:
        log(f"accuracy не падает ниже 80% при m ≤ {MMAX}")
    else:
        log(f"наименьшее m, при котором accuracy < 80%: {below}")
    fragile = int(np.argmin(acc_c[:, -1]))
    log(f"при m={MMAX} ниже всех держится буква {LETTERS[fragile]} ({acc_c[fragile, -1]:.3f})")


def report_uncertain(Xte, yte, inst_te, Pte, log):
    order = np.argsort(Pte.max(axis=1))
    log("--- три самых неуверенных тестовых примера ---")
    for rank, i in enumerate(order[:3], 1):
        p = Pte[i]
        log(f"#{rank}  истина {LETTERS[int(yte[i])]} №{int(inst_te[i])}, "
            f"ответ {LETTERS[int(np.argmax(p))]}")
        for row in vec_to_rows(Xte[i]):
            log(row)
        log(f"p1={p[0]:.4f}  p2={p[1]:.4f}  p3={p[2]:.4f}")
        log("")
    return order


def report_foreign(net, items, log):
    log("--- чужой почерк, без дообучения ---")
    X = np.vstack([it["x"] for it in items])
    y = np.array([CLASS[it["cls"]] for it in items])
    pred, P = predict(net, X)
    ok = int(np.sum(pred == y))
    log(f"accuracy = {ok / len(y):.3f} ({ok}/{len(y)})")
    log("матрица ошибок:")
    log(fmt_cm(confusion_matrix(y, pred)))
    for it, p, k in zip(items, P, pred):
        mark = "верно" if k == CLASS[it["cls"]] else "ошибка"
        flag = ", не распознано" if p[k] < TAU else ""
        log(f"  {LETTERS[CLASS[it['cls']]]} №{it['inst']}: ответ {LETTERS[k]} "
            f"p=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) {mark}{flag}")
    return X, y, ok / len(y)


def report_alien(net, item, log):
    p = forward(net, item["x"])
    k = int(np.argmax(p))
    log("--- чужая буква К ---")
    for row in item["rows"]:
        log(row)
    log(f"p1={p[0]:.4f}  p2={p[1]:.4f}  p3={p[2]:.4f}")
    log(f"argmax: {LETTERS[k]}, уверенность {p[k]:.4f}")
    log("Решение: " + ("не распознано" if p[k] < TAU else LETTERS[k]))
    log("верного класса в наборе нет, поэтому «были бы верными» = 0")


def overlap_cells(X, y, a, b):
    both = (X[y == a].mean(0) >= 0.5) & (X[y == b].mean(0) >= 0.5)
    idx = np.flatnonzero(both)
    return ", ".join(f"({i // 5 + 1},{i % 5 + 1})" for i in idx) or "нет"


def main():
    log = Log()
    log("классы: c1=Б, c2=А, c3=Ж; p1, p2, p3 — их вероятности")
    log(f"H={H_BASE}, η={ETA:.2f}, mmax={MMAX}, Ee={EE}, τ={TAU}, эпох ≤ {EPOCHS}")
    log("Es — средняя ½‖p−y‖², CE — средняя кросс-энтропия")
    log("")

    items = read_dataset(ROOT / "dataset.txt")
    X, y, inst, tr, te = split_xy(items)
    if tr.sum() != 21 or te.sum() != 9 or len(X) != 30:
        raise SystemExit("в dataset.txt должно быть 30 объектов: 21 обучение и 9 тест")
    log("--- контроль датасета ---")
    nfill = quality_control(X, y, log)
    for i in range(3):
        for j in range(i + 1, 3):
            log(f"клетки, закрашенные хотя бы в половине экземпляров {LETTERS[i]} и {LETTERS[j]}: "
                f"{overlap_cells(X, y, i, j)}")
    log("")

    Xtr, Ytr, ytr = X[tr], one_hot(y[tr]), y[tr]
    Xte, yte, inst_te = X[te], y[te], inst[te]
    grad_check(Xtr, Ytr, log)
    log("")

    runs = run_study(Xtr, Ytr, Xte, yte)
    base = runs[(H_BASE, SEEDS[0])]
    net = base["net"]
    cm_te, Pte = report_base(base, Xtr, ytr, Xte, yte, inst_te, log)
    log("")
    epoch_lists = report_table(runs, log)
    log("")
    neuron_report(net, Xtr, ytr, log)
    log("")
    order = report_uncertain(Xte, yte, inst_te, Pte, log)

    acc, acc_c = noise_curve(net, Xte, yte, trials=50, seed=7)
    report_noise(acc, acc_c, log)
    log("")

    foreign = read_dataset(ROOT / "other_hand.txt")
    Xf, yf, acc_f = report_foreign(net, foreign, log)
    own = float(np.mean(np.argmax(Pte, axis=1) == yte))
    log(f"свой тест {own:.3f}, чужой почерк {acc_f:.3f}, разница {acc_f - own:+.3f}")
    log("")
    alien = read_dataset(ROOT / "alien.txt")[0]
    report_alien(net, alien, log)
    log("")
    log(f"порог отказа τ={TAU}")
    for title_s, Xx, yy in (
        ("тест", Xte, yte),
        ("чужой почерк", Xf, yf),
        ("чужая буква", alien["x"][None, :], None),
    ):
        rej, rej_ok, n = rejection_stats(net, Xx, yy)
        extra = " (верного класса нет)" if yy is None else f", из них были бы верными: {rej_ok}"
        log(f"  {title_s}: отвергнуто {rej} из {n}{extra}")
    log("")

    log("--- режим функционирования ---")
    for cls, num in (("1", 1), ("2", 3), ("3", 5)):
        it = find_item(items, cls, num)
        functioning_mode(net, it["rows"], log,
                         note=f"Обучение, {LETTERS[CLASS[cls]]} №{num}")
    for cls, num in (("1", 8), ("2", 9), ("3", 10)):
        it = find_item(items, cls, num)
        functioning_mode(net, it["rows"], log,
                         note=f"Тест, {LETTERS[CLASS[cls]]} №{num}")
    distortions = (
        ("1", 8, ((0, 4), (3, 2), (6, 0)), "Искажение теста Б №8"),
        ("2", 9, ((0, 2), (1, 1), (4, 0), (6, 4)), "Искажение теста А №9"),
    )
    for cls, num, cells, note in distortions:
        it = find_item(items, cls, num)
        rows = vec_to_rows(flip_cells(it["x"], cells))
        where = ", ".join(f"({r + 1},{c + 1})" for r, c in cells)
        functioning_mode(net, rows, log, note=f"{note}, изменено клеток: {len(cells)}, позиции {where}")
    functioning_mode(net, alien["rows"], log, note="Чужая буква К")
    for cls, num in (("2", 1), ("3", 2)):
        it = find_item(foreign, cls, num)
        functioning_mode(net, it["rows"], log,
                         note=f"Чужой почерк, {LETTERS[CLASS[cls]]} №{num}")

    PLOTS.mkdir(exist_ok=True)
    hists = {H: runs[(H, SEEDS[0])]["hist"] for H in H_LIST}
    plot_convergence(hists, nfill, PLOTS / "convergence.png")
    plot_spread(epoch_lists, nfill, PLOTS / "epochs_spread.png")
    plot_confusion(cm_te, nfill, PLOTS / "confusion_test.png")
    plot_noise(acc, acc_c, nfill, PLOTS / "noise_accuracy.png")
    plot_weights(net, nfill, PLOTS / "weight_maps.png")
    plot_uncertain(Xte, yte, inst_te, Pte, order, nfill, PLOTS / "uncertain.png")
    np.savez(ROOT / "model.npz", W1=net["W1"], T1=net["T1"], W2=net["W2"], T2=net["T2"])
    log("графики: plots/convergence.png, epochs_spread.png, confusion_test.png,")
    log("         noise_accuracy.png, weight_maps.png, uncertain.png")
    log.save(ROOT / "results.txt")


def interactive():
    path = ROOT / "model.npz"
    if not path.exists():
        items = read_dataset(ROOT / "dataset.txt")
        X, y, inst, tr, te = split_xy(items)
        net = train(X[tr], one_hot(y[tr]), H_BASE, SEEDS[0])["net"]
    else:
        z = np.load(path)
        net = {k: z[k] for k in ("W1", "T1", "W2", "T2")}
    print("Режим функционирования. Введите 7 строк по 5 символов # и .")
    print("Пустая строка или q — выход. Классы: p1=Б, p2=А, p3=Ж, τ=0.6")
    while True:
        rows = []
        try:
            for _ in range(7):
                line = input()
                if line.strip().lower() in {"q", "quit", ""}:
                    return
                rows.append(line)
        except EOFError:
            return
        try:
            functioning_mode(net, rows)
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "func":
        interactive()
    else:
        main()