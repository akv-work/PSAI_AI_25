import numpy as np
import matplotlib.pyplot as plt

def read_dataset(path):
    X, y, nums = [], [], []
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    i = 0
    while i < len(lines):
        cls, num = map(int, lines[i].split())
        grid = lines[i+1:i+8]
        vec = np.array([1 if c == '#' else 0 for row in grid for c in row], float)
        X.append(vec); y.append(cls - 1); nums.append(num)
        i += 8
    return np.array(X), np.array(y), np.array(nums)

def hamming(a, b):
    return int(np.sum(a != b))

def dataset_stats(X, y):
    Nfill = int(X.sum())
    intra, inter = [], []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            d = hamming(X[i], X[j])
            (intra if y[i] == y[j] else inter).append(d)
    print(f"N_fill = {Nfill}")
    print(f"intra-class mean Hamming = {np.mean(intra):.2f}")
    print(f"inter-class mean Hamming = {np.mean(inter):.2f}")
    return Nfill

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W1, b1, W2, b2):
    H = sigmoid(X @ W1 + b1)
    P = softmax(H @ W2 + b2)
    return H, P

def cross_entropy(P, y):
    n = len(y)
    return -np.mean(np.log(P[np.arange(n), y] + 1e-12))

def backward(X, y, W1, b1, W2, b2):
    H, P = forward(X, W1, b1, W2, b2)
    n = len(y)
    Y = np.zeros_like(P); Y[np.arange(n), y] = 1
    dZ2 = (P - Y) / n
    dW2 = H.T @ dZ2
    db2 = dZ2.sum(0)
    dH = dZ2 @ W2.T
    dZ1 = dH * H * (1 - H)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(0)
    return dW1, db1, dW2, db2

def init_weights(H, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.1, (35, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 0.1, (H, 3))
    b2 = np.zeros(3)
    return W1, b1, W2, b2

def train(X, y, H, eta=0.05, Ee=0.05, max_epochs=5000, seed=0):
    W1, b1, W2, b2 = init_weights(H, seed)
    rng = np.random.default_rng(seed)
    Es_hist = []
    ep = 0
    for ep in range(1, max_epochs + 1):
        idx = rng.permutation(len(X))
        for i in idx:
            xi = X[i:i+1]; yi = y[i:i+1]
            dW1, db1, dW2, db2 = backward(xi, yi, W1, b1, W2, b2)
            W1 -= eta * dW1; b1 -= eta * db1
            W2 -= eta * dW2; b2 -= eta * db2
        _, P = forward(X, W1, b1, W2, b2)
        Es = cross_entropy(P, y)
        Es_hist.append(Es)
        if Es <= Ee:
            break
    return W1, b1, W2, b2, Es_hist, ep

def accuracy(X, y, W1, b1, W2, b2):
    _, P = forward(X, W1, b1, W2, b2)
    return float(np.mean(P.argmax(1) == y))

def confusion_matrix(X, y, W1, b1, W2, b2):
    _, P = forward(X, W1, b1, W2, b2)
    pred = P.argmax(1)
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm

def numerical_gradient_check(X, y, W1, b1, W2, b2, eps=1e-5):
    def ce_of(W1_, W2_):
        _, P = forward(X, W1_, b1, W2_, b2)
        return cross_entropy(P, y)
    dW1, db1, dW2, db2 = backward(X, y, W1, b1, W2, b2)
    i, j = 3, 0
    W1p = W1.copy(); W1p[i, j] += eps
    W1m = W1.copy(); W1m[i, j] -= eps
    num = (ce_of(W1p, W2) - ce_of(W1m, W2)) / (2 * eps)
    ana = dW1[i, j]
    print(f"grad check W1[{i},{j}]: analytic={ana:.6e} numeric={num:.6e} rel_err={abs(ana-num)/max(abs(num),1e-12):.2e}")

def noise_test(W1, b1, W2, b2, Xte, yte, mmax=8, reps=50, seed=0):
    rng = np.random.default_rng(seed)
    accs = np.zeros((mmax + 1, 3))
    for m in range(mmax + 1):
        cls_correct = np.zeros(3); cls_total = np.zeros(3)
        for _ in range(reps):
            for i in range(len(Xte)):
                x = Xte[i].copy()
                flip = rng.choice(35, size=m, replace=False)
                x[flip] = 1 - x[flip]
                _, p = forward(x[None], W1, b1, W2, b2)
                cls_total[yte[i]] += 1
                if p.argmax() == yte[i]:
                    cls_correct[yte[i]] += 1
        accs[m] = cls_correct / np.maximum(cls_total, 1)
    return accs

def plot_convergence(curves, Nfill, surname, V):
    plt.figure()
    for H, (Es_hist, ce_hist) in curves.items():
        plt.semilogy(Es_hist, label=f"H={H}")
    plt.xlabel("epoch"); plt.ylabel("Es (log)")
    plt.title(f"{surname}, V={V}, N_fill={Nfill}")
    plt.legend(); plt.grid(True); plt.show()

def plot_confusion(cm, Nfill, surname, V):
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.colorbar()
    plt.title(f"{surname}, V={V}, N_fill={Nfill}")
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.show()

def plot_noise(acc_noise, Nfill, surname, V):
    plt.figure()
    plt.plot(range(len(acc_noise)), acc_noise.mean(1), 'o-')
    for c in range(3):
        plt.plot(range(len(acc_noise)), acc_noise[:, c], 's--', label=f"class {c+1}")
    plt.axhline(0.8, color='red', ls=':')
    plt.xlabel("m"); plt.ylabel("accuracy")
    plt.title(f"{surname}, V={V}, N_fill={Nfill}")
    plt.legend(); plt.grid(True); plt.show()

def plot_weights(W1, Nfill, surname, V):
    for j in range(W1.shape[1]):
        w = W1[:, j].reshape(7, 5)
        vmax = np.abs(W1).max()
        plt.figure()
        plt.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(f"{surname}, V={V}, N_fill={Nfill} | hidden neuron {j+1}")
        plt.show()
        idx = np.unravel_index(np.argsort(-np.abs(w).ravel())[:3], (7, 5))
        cells = [(int(idx[0][k]) + 1, int(idx[1][k]) + 1) for k in range(3)]
        signs = [float(np.sign(w[idx][k])) for k in range(3)]
        print(f"neuron {j+1}: top cells (row,col)={cells}, signs={signs}, class-sens v_kj={W2[j].round(3)}")

def show_least_confident(Xte, yte, Pte, k=3):
    conf = Pte.max(1)
    for i in np.argsort(conf)[:k]:
        print(f"test example {i+1}: probs={Pte[i].round(4)}, true={yte[i]+1}")
        for r in Xte[i].reshape(7, 5).astype(int):
            print(''.join('#' if v else '.' for v in r))

def reject_decisions(P, tau=0.6):
    return P.max(1) < tau

def predict_ui(W1, b1, W2, b2, letters, tau=0.6):
    print("enter 7 lines of 5 chars (# or .):")
    rows = [input().strip() for _ in range(7)]
    x = np.array([1 if c == '#' else 0 for r in rows for c in r], float)
    _, p = forward(x[None], W1, b1, W2, b2)
    p = p[0]
    for r in rows:
        print(r)
    print(f"p1={p[0]:.4f} p2={p[1]:.4f} p3={p[2]:.4f}")
    cls = int(p.argmax())
    print(f"class={letters[cls]} confidence={p[cls]:.4f}")
    decision = "not recognized" if p[cls] < tau else letters[cls]
    print(f"decision={decision}")
    return p

def main():
    surname = "IVANOV"
    V = 2
    H = 4
    eta = 0.05
    mmax = 8
    letters = ['A', 'B', 'C']

    X, y, nums = read_dataset('dataset.txt')
    Nfill = dataset_stats(X, y)

    tr = np.array([(n >= 1 and n <= 7) for n in nums])
    te = ~tr
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    W1, b1, W2, b2, Es_hist, epochs = train(Xtr, ytr, H, eta=eta, seed=42)
    _, P = forward(Xtr, W1, b1, W2, b2)
    Es_final = cross_entropy(P, ytr)
    print(f"H={H} eta={eta} epochs={epochs} Es_final={Es_final:.4f}")
    print(f"acc_train={accuracy(Xtr, ytr, W1, b1, W2, b2):.4f}")
    print(f"acc_test={accuracy(Xte, yte, W1, b1, W2, b2):.4f}")
    cm_te = confusion_matrix(Xte, yte, W1, b1, W2, b2)
    cm_tr = confusion_matrix(Xtr, ytr, W1, b1, W2, b2)
    print("confusion train:"); print(cm_tr)
    print("confusion test:");  print(cm_te)

    _, Pte = forward(Xte, W1, b1, W2, b2)
    pred = Pte.argmax(1); conf = Pte.max(1)
    ok = pred == yte
    if ok.any(): print(f"mean conf correct={conf[ok].mean():.4f}")
    if (~ok).any(): print(f"mean conf wrong={conf[~ok].mean():.4f}")

    numerical_gradient_check(Xtr, ytr, W1, b1, W2, b2)

    results = {}
    curves = {}
    for Hv in [2, H, 2 * H]:
        eps, accs = [], []
        first_hist = None
        for seed in range(5):
            W1h, b1h, W2h, b2h, h_hist, ep = train(Xtr, ytr, Hv, eta=eta, seed=seed)
            if first_hist is None: first_hist = h_hist
            eps.append(ep)
            accs.append(accuracy(Xte, yte, W1h, b1h, W2h, b2h))
        curves[Hv] = (first_hist, first_hist)
        results[Hv] = (np.mean(eps), min(eps), max(eps),
                       np.mean(accs), min(accs), max(accs),
                       sum(1 for e in eps if e >= 5000),
                       35 * Hv + Hv + Hv * 3 + 3)
        print(f"H={Hv}: epochs mean={np.mean(eps):.1f} min={min(eps)} max={max(eps)} | "
              f"acc mean={np.mean(accs):.3f} min={min(accs):.3f} max={max(accs):.3f} | "
              f"stopped early={sum(1 for e in eps if e>=5000)} | params={35*Hv+Hv+Hv*3+3}")

    plot_convergence(curves, Nfill, surname, V)

    plt.figure()
    data_eps = [[results[Hv][0]] * 5 for Hv in [2, H, 2 * H]]
    plt.boxplot(data_eps, labels=[str(Hv) for Hv in [2, H, 2 * H]])
    plt.xlabel("H"); plt.ylabel("epochs")
    plt.title(f"{surname}, V={V}, N_fill={Nfill}")
    plt.grid(True); plt.show()

    plot_confusion(cm_te, Nfill, surname, V)

    acc_noise = noise_test(W1, b1, W2, b2, Xte, yte, mmax=mmax, seed=0)
    plot_noise(acc_noise, Nfill, surname, V)
    below = np.where(acc_noise.mean(1) < 0.8)[0]
    print(f"first m with accuracy<0.8: {below[0] if len(below) else 'never'}")

    plot_weights(W1, Nfill, surname, V)

    show_least_confident(Xte, yte, Pte, 3)

    rej = reject_decisions(Pte, tau=0.6)
    print(f"rejected on test: {rej.sum()} of {len(rej)}, correct among rejected: {(rej & ok).sum()}")

    while True:
        predict_ui(W1, b1, W2, b2, letters, tau=0.6)
        if input("again? (y/n): ").strip().lower() != 'y':
            break

if __name__ == "__main__":
    main()