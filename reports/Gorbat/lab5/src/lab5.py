import os
import sys

import matplotlib.pyplot as plt
import numpy as np

LETTERS = {1: "Г", 2: "О", 3: "Р"}
H_VAR, LR, MMAX, V = 5, 0.10, 10, 3
EE, MAX_EP, INIT, TAU = 0.05, 5000, 0.5, 0.6
SEEDS = [0, 1, 2, 3, 4]
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SURNAME = "Горбат"
TITLE = None

sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def softmax(z):
    z = np.asarray(z, float) - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def onehot(c, k=3):
    y = np.zeros(k)
    y[c - 1] = 1.0
    return y


def parse_file(path):
    X, C, I = [], [], []
    lines = [ln.rstrip("\n") for ln in open(path) if ln.strip() and not ln.startswith("# класс") and not ln.startswith("# чуж")]
    i = 0
    while i < len(lines):
        if lines[i].startswith("#"):
            i += 1
            continue
        parts = lines[i].split()
        if len(parts) == 2 and parts[0].lstrip("-").isdigit():
            c, inst = int(parts[0]), int(parts[1])
            grid = lines[i + 1 : i + 8]
            vec = np.array([1.0 if ch == "#" else 0.0 for row in grid for ch in row[:5]])
            X.append(vec)
            C.append(c)
            I.append(inst)
            i += 8
        else:
            i += 1
    return np.array(X), np.array(C), np.array(I)


def hamming(a, b):
    return float(np.sum(a != b))


def quality(X, C):
    nfill = int(X.sum())
    intra, inter = [], []
    n = len(X)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming(X[i], X[j])
            (intra if C[i] == C[j] else inter).append(d)
    return nfill, float(np.mean(intra)), float(np.mean(inter))


def split_by_instance(X, C, I):
    tr = I <= 7
    return X[tr], C[tr], X[~tr], C[~tr]


class MLP:
    def __init__(self, h, seed):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(35, h) * INIT
        self.b1 = np.zeros(h)
        self.W2 = rng.randn(h, 3) * INIT
        self.b2 = np.zeros(3)

    def fwd(self, x):
        x = np.asarray(x, float).ravel()
        self.h = sig(x @ self.W1 + self.b1)
        self.p = softmax(self.h @ self.W2 + self.b2)
        return self.p

    def step(self, x, c):
        p = self.fwd(x)
        y = onehot(c)
        # CE + softmax: δ = p − y
        d = p - y
        self.W2 -= LR * np.outer(self.h, d)
        self.b2 -= LR * d
        dh = (self.W2 @ d) * self.h * (1.0 - self.h)
        self.W1 -= LR * np.outer(x, dh)
        self.b1 -= LR * dh
        return -np.log(np.clip(p[c - 1], 1e-12, 1.0))

    def nparams(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


def ce_mean(net, X, C):
    s = 0.0
    for x, c in zip(X, C):
        p = net.fwd(x)
        s += -np.log(np.clip(p[c - 1], 1e-12, 1.0))
    return s / len(C)


def predict_all(net, X):
    P = np.array([net.fwd(x) for x in X])
    pred = P.argmax(axis=1) + 1
    conf = P.max(axis=1)
    return P, pred, conf


def accuracy(pred, C):
    return float(np.mean(pred == C))


def confusion(pred, C):
    m = np.zeros((3, 3), int)
    for t, p in zip(C, pred):
        m[t - 1, p - 1] += 1
    return m


def train(X, C, h, seed):
    net, rng, hist_es, hist_ce = MLP(h, seed), np.random.RandomState(seed + 9), [], []
    n = len(C)
    for ep in range(1, MAX_EP + 1):
        order = rng.permutation(n)
        ces = []
        for i in order:
            ces.append(net.step(X[i], int(C[i])))
        es = float(np.mean(ces))
        hist_es.append(es)
        hist_ce.append(ce_mean(net, X, C))
        if es <= EE:
            return net, ep, es, True, hist_es, hist_ce
    return net, MAX_EP, es, False, hist_es, hist_ce


def num_grad(net, x, c, eps=1e-5):
    p = net.fwd(x)
    y = onehot(c)
    anal = np.outer(net.h, p - y)
    num = np.zeros_like(net.W2)
    w = net.W2
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            old = w[i, j]
            w[i, j] = old + eps
            lp = -np.log(np.clip(net.fwd(x)[c - 1], 1e-12, 1))
            w[i, j] = old - eps
            lm = -np.log(np.clip(net.fwd(x)[c - 1], 1e-12, 1))
            w[i, j] = old
            num[i, j] = (lp - lm) / (2 * eps)
    net.fwd(x)
    return anal, num, float(np.max(np.abs(anal - num)))


def flip(x, m, rng):
    y = x.copy()
    if m <= 0:
        return y
    idx = rng.choice(35, m, replace=False)
    y[idx] = 1.0 - y[idx]
    return y


def noise_curve(net, X, C, mmax, nrep=50):
    acc_m = []
    acc_cls = {1: [], 2: [], 3: []}
    rng = np.random.RandomState(0)
    for m in range(mmax + 1):
        ok, n = 0, 0
        cok = {1: 0, 2: 0, 3: 0}
        cn = {1: 0, 2: 0, 3: 0}
        for x, c in zip(X, C):
            for _ in range(nrep):
                p = net.fwd(flip(x, m, rng)).argmax() + 1
                ok += int(p == c)
                n += 1
                cok[c] += int(p == c)
                cn[c] += 1
        acc_m.append(ok / n)
        for k in (1, 2, 3):
            acc_cls[k].append(cok[k] / cn[k])
    return np.array(acc_m), acc_cls


def grid_str(x):
    s = x.reshape(7, 5)
    return "\n".join("".join("#" if v > 0.5 else "." for v in row) for row in s)


def decide(p, tau=TAU):
    k = int(p.argmax()) + 1
    conf = float(p.max())
    letter = LETTERS[k]
    if conf < tau:
        return letter, conf, "не распознано"
    return letter, conf, letter


def caption(extra=""):
    t = f"{SURNAME}, вар.{V}, Nfill={Nfill}"
    return t if not extra else f"{t}. {extra}"


def plots(base, h_runs, Xte, Cte, Ite, acc_m, acc_cls, Pte, pred_te):
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for h, rows in h_runs.items():
        r = next((x for x in rows if x["ok"]), rows[0])
        ax.plot(r["hist_es"], label=f"H={h} Es")
        ax.plot(r["hist_ce"], ls="--", label=f"H={h} CE")
    ax.axhline(EE, color="k", ls=":", label="Ee")
    ax.set(xlabel="эпоха", ylabel="ошибка", yscale="log", title=caption("сходимость"))
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "conv.png"), dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(5)
    w = 0.25
    for i, (h, rows) in enumerate(h_runs.items()):
        bars = ax.bar(xs + (i - 1) * w, [r["ep"] for r in rows], w, label=f"H={h}")
        for b, r in zip(bars, rows):
            if not r["ok"]:
                b.set_hatch("///")
    ax.set(xticks=xs, xticklabels=SEEDS, xlabel="seed", ylabel="эпохи", title=caption("разброс эпох"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "epochs.png"), dpi=140)
    plt.close()

    cm = confusion(pred_te, Cte)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set(xticks=range(3), yticks=range(3),
           xticklabels=[LETTERS[k] for k in (1, 2, 3)],
           yticklabels=[LETTERS[k] for k in (1, 2, 3)],
           xlabel="прогноз", ylabel="истина", title=caption("матрица ошибок, тест"))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cm.png"), dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(MMAX + 1), acc_m, "k-o", label="все")
    for k in (1, 2, 3):
        ax.plot(range(MMAX + 1), acc_cls[k], label=LETTERS[k])
    ax.axhline(0.8, color="r", ls="--", label="80%")
    ax.set(xlabel="m испорченных клеток", ylabel="accuracy", title=caption("устойчивость к шуму"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "noise.png"), dpi=140)
    plt.close()

    H = base.W1.shape[1]
    fig, axes = plt.subplots(1, H, figsize=(2.2 * H, 3.2))
    vmax = np.max(np.abs(base.W1))
    for j, ax in enumerate(np.atleast_1d(axes)):
        w = base.W1[:, j].reshape(7, 5)
        ax.imshow(w, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"h{j+1}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(caption("карты весов скрытого слоя"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "weights.png"), dpi=140)
    plt.close()

    conf = Pte.max(axis=1)
    idx = np.argsort(conf)[:3]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2))
    for ax, i in zip(axes, idx):
        ax.imshow(Xte[i].reshape(7, 5), cmap="gray_r", vmin=0, vmax=1)
        pr = ", ".join(f"{LETTERS[k]}={Pte[i, k-1]:.2f}" for k in (1, 2, 3))
        ax.set_title(f"№{int(Ite[i])} {LETTERS[Cte[i]]}\n{pr}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(caption("самые неуверенные тест"))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "uncertain.png"), dpi=140)
    plt.close()
    return idx


def show(net, x, tag):
    p = net.fwd(x)
    letter, conf, dec = decide(p)
    print(f"\n[{tag}]")
    print(grid_str(x))
    print(f"p: Г={p[0]:.4f} О={p[1]:.4f} Р={p[2]:.4f}  класс={letter}  conf={conf:.4f}  решение={dec}")
    return p, dec


def refuse_stats(net, X, C, tag):
    rej = ok_rej = 0
    for x, c in zip(X, C):
        p = net.fwd(x)
        letter, conf, dec = decide(p)
        if dec == "не распознано":
            rej += 1
            if letter == LETTERS[c] if c in LETTERS else False:
                ok_rej += 1
            elif c in LETTERS and (p.argmax() + 1) == c:
                ok_rej += 1
        if c in LETTERS and dec == "не распознано" and (p.argmax() + 1) == c:
            pass
    # отвергнутые, которые без порога были бы верны
    would = 0
    nrej = 0
    for x, c in zip(X, C):
        p = net.fwd(x)
        if p.max() < TAU:
            nrej += 1
            if c in LETTERS and p.argmax() + 1 == c:
                would += 1
    print(f"{tag}: отвергнуто {nrej}/{len(C)}, из них верных без порога {would}")
    return nrej, would


def main():
    global Nfill, TITLE
    os.makedirs(FIG, exist_ok=True)
    X, C, I = parse_file(os.path.join(HERE, "dataset.txt"))
    Nfill, intra, inter = quality(X, C)
    TITLE = caption()
    print(f"объектов={len(X)} Nfill={Nfill} Hamming intra={intra:.2f} inter={inter:.2f}")
    if intra < 3:
        print("ВНИМАНИЕ: intra Hamming < 3, датасет слишком однородный")
    Xtr, Ctr, Xte, Cte = split_by_instance(X, C, I)
    Itr, Ite = I[I <= 7], I[I > 7]
    print(f"train={len(Ctr)} test={len(Cte)}")

    print("\n=== база H=5 seed=0 ===")
    net, ep, es, ok, hs, hc = train(Xtr, Ctr, H_VAR, 0)
    Ptr, pred_tr, conf_tr = predict_all(net, Xtr)
    Pte, pred_te, conf_te = predict_all(net, Xte)
    acc_tr, acc_te = accuracy(pred_tr, Ctr), accuracy(pred_te, Cte)
    print(f"эпохи={ep} Es={es:.4f} сход={ok} params={net.nparams()}")
    print(f"acc train={acc_tr:.3f} test={acc_te:.3f}")
    print("CM train\n", confusion(pred_tr, Ctr))
    print("CM test\n", confusion(pred_te, Cte))
    ok_m = conf_te[pred_te == Cte]
    bad_m = conf_te[pred_te != Cte]
    print(f"уверенность верных={ok_m.mean() if len(ok_m) else 0:.3f} "
          f"ошибок={bad_m.mean() if len(bad_m) else float('nan'):.3f}")

    anal, num, diff = num_grad(net, Xtr[0], int(Ctr[0]))
    print(f"аудит градиента W2: max|anal-num|={diff:.2e}")

    print("\n=== исследование H, 5 seed ===")
    h_runs = {}
    for h in (2, H_VAR, 2 * H_VAR):
        rows = []
        dummy = MLP(h, 0)
        for s in SEEDS:
            n, e, ess, ook, hse, hce = train(Xtr, Ctr, h, s)
            _, pr, _ = predict_all(n, Xte)
            rows.append(dict(seed=s, net=n, ep=e, es=ess, ok=ook, hist_es=hse, hist_ce=hce,
                             acc=accuracy(pr, Cte), npar=dummy.nparams()))
        h_runs[h] = rows
        eps = [r["ep"] for r in rows]
        accs = [r["acc"] for r in rows]
        nf = sum(not r["ok"] for r in rows)
        print(f"H={h:2d} params={dummy.nparams():3d} эпохи {np.mean(eps):.0f} [{min(eps)}–{max(eps)}]  "
              f"acc {np.mean(accs):.3f} [{min(accs):.3f}–{max(accs):.3f}] fail {nf}/5")

    acc_m, acc_cls = noise_curve(net, Xte, Cte, MMAX)
    m80 = next((m for m, a in enumerate(acc_m) if a < 0.8), None)
    print(f"\nшум: acc падает <80% при m={m80}")
    for k in (1, 2, 3):
        print(f"  {LETTERS[k]}: " + ", ".join(f"{a:.2f}" for a in acc_cls[k]))

    print("\nкарты весов: argmax |w| и знак связи с классами (W2)")
    for j in range(net.W1.shape[1]):
        w = net.W1[:, j].reshape(7, 5)
        r, c = divmod(int(np.abs(w).argmax()), 5)
        cls = LETTERS[int(np.abs(net.W2[j]).argmax()) + 1]
        print(f"  h{j+1}: max |w| клетка ({r+1},{c+1})={w[r, c]:+.2f}; W2={np.round(net.W2[j], 2)} ближе к {cls}")

    idx = plots(net, h_runs, Xte, Cte, Ite, acc_m, acc_cls, Pte, pred_te)
    print("неуверенные тест:", [(int(Ite[i]), LETTERS[Cte[i]], float(Pte[i].max())) for i in idx])

    Xo, Co, _ = parse_file(os.path.join(HERE, "other.txt"))
    Po, pred_o, _ = predict_all(net, Xo)
    print(f"\nчужой почерк acc={accuracy(pred_o, Co):.3f}  (свой тест {acc_te:.3f})")
    print("CM other\n", confusion(pred_o, Co))

    Xf, _, _ = parse_file(os.path.join(HERE, "foreign.txt"))
    pf = net.fwd(Xf[0])
    L, conf, dec = decide(pf)
    print(f"чужая К: p={np.round(pf, 4)} класс={L} conf={conf:.4f} решение={dec}")

    refuse_stats(net, Xte, Cte, "тест")
    refuse_stats(net, Xo, Co, "чужой почерк")
    p = net.fwd(Xf[0])
    print(f"чужая буква: отвергнута={'да' if p.max() < TAU else 'нет'}")

    print("\n--- режим функционирования ---")
    for x, c, inst in zip(Xtr, Ctr, Itr):
        if inst in (1, 3, 5):
            show(net, x, f"train {LETTERS[c]} №{inst}")
    for x, c, inst in zip(Xte, Cte, Ite):
        show(net, x, f"test {LETTERS[c]} №{inst}")
    rng = np.random.RandomState(1)
    show(net, flip(Xte[0], 2, rng), "искажение m=2")
    show(net, flip(Xte[3], 5, rng), "искажение m=5")
    show(net, Xf[0], "чужая К")
    show(net, Xo[0], "чужой почерк Г")
    show(net, Xo[2], "чужой почерк О")
    print(f"\nграфики: {FIG}")

    if sys.stdin.isatty():
        print("введите 7 строк по 5 символов #/.  (q — выход)")
        buf = []
        while True:
            s = input()
            if s.lower() in ("q", "quit", "exit"):
                break
            buf.append(s.strip())
            if len(buf) == 7:
                vec = np.array([1.0 if ch == "#" else 0.0 for row in buf for ch in row[:5]])
                show(net, vec, "ввод")
                buf = []


if __name__ == "__main__":
    main()
