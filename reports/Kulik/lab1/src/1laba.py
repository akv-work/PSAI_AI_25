import numpy as np
import matplotlib.pyplot as plt

def phi(z):
    return 1.0 / (1.0 + np.exp(-z))

def dphi(a):
    return a * (1.0 - a)

def scale01(v, lo, hi):
    return (v - lo) / (hi - lo)

def unscale01(v, lo, hi):
    return lo + v * (hi - lo)

def feedforward(p, Vh, Bh, Vo, Bo):
    zh = np.dot(p, Vh) - Bh
    ah = phi(zh)
    zo = np.dot(ah, Vo) - Bo
    ao = phi(zo)
    return zh, ah, zo, ao

def infer(p, Vh, Bh, Vo, Bo):
    return feedforward(p, Vh, Bh, Vo, Bo)[3]

def fit_two_layer(P, t, lr=0.5, tol=0.005, n_epochs=20000, beta=0.9):
    rng = np.random.default_rng(1)

    Vh = rng.uniform(-0.1, 0.1, (2, 2))
    Bh = rng.uniform(-0.1, 0.1, 2)
    Vo = rng.uniform(-0.1, 0.1, 2)
    Bo = rng.uniform(-0.1, 0.1)

    dVh = np.zeros_like(Vh)
    dBh = np.zeros_like(Bh)
    dVo = np.zeros_like(Vo)
    dBo = 0.0

    hist = []

    for _ in range(1, n_epochs + 1):
        for p, target in zip(P, t):
            zh, ah, zo, ao = feedforward(p, Vh, Bh, Vo, Bo)

            err_o = ao - target
            d_o = err_o * dphi(ao)

            err_h = dphi(ah) * (Vo * d_o)

            gVo = lr * ah * d_o
            gBo = -lr * d_o
            gVh = lr * np.outer(p, err_h)
            gBh = -lr * err_h

            dVo = -gVo + beta * dVo
            dBo = -gBo + beta * dBo
            dVh = -gVh + beta * dVh
            dBh = -gBh + beta * dBh

            Vo = Vo + dVo
            Bo = Bo + dBo
            Vh = Vh + dVh
            Bh = Bh + dBh

        J = 0.0
        for p, target in zip(P, t):
            ao = infer(p, Vh, Bh, Vo, Bo)
            J += 0.5 * (ao - target) ** 2

        hist.append(J)

        if J <= tol:
            break

    return Vh, Bh, Vo, Bo, hist

def fit_single_layer(P, t, lr=0.5, tol=0.005, n_epochs=20000):
    rng = np.random.default_rng(1)

    W = rng.uniform(-0.1, 0.1, 2)
    B = rng.uniform(-0.1, 0.1)

    hist = []

    for _ in range(1, n_epochs + 1):
        for p, target in zip(P, t):
            z = np.dot(p, W) - B
            a = phi(z)
            d = (a - target) * dphi(a)

            W = W - lr * p * d
            B = B + lr * d

        J = 0.0
        for p, target in zip(P, t):
            z = np.dot(p, W) - B
            a = phi(z)
            J += 0.5 * (a - target) ** 2

        hist.append(J)

        if J <= tol:
            break

    return W, B, hist

def draw_error_curve(hist, title):
    plt.figure()
    plt.plot(hist)
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es")
    plt.title(title)
    plt.grid()
    plt.show()

def hit_rate(preds, targets, lo, hi):
    eps = 0.5 * abs(hi - lo)
    good = sum(1 for p, t in zip(preds, targets) if abs(p - t) < eps)
    return good / len(targets)

def run():
    lo, hi = -4, -9

    a_raw = np.array([-4, -4, -9, -9])
    b_raw = np.array([-4, -9, -4, -9])
    y_raw = np.array([-4, -9, -9, -4])

    Pin = np.column_stack([a_raw, b_raw])
    Pn = scale01(Pin, lo, hi)
    Tn = scale01(y_raw, lo, hi)

    Vh, Bh, Vo, Bo, hist_mlp = fit_two_layer(
        Pn, Tn, lr=0.5, tol=0.005, n_epochs=20000, beta=0.9
    )

    print("\n=== Многослойный персептрон 2-2-1 ===")
    print("Эпох обучения:", len(hist_mlp))
    print("Финальная суммарная ошибка Es:", hist_mlp[-1])

    preds_mlp = []
    print("\nОтветы сети:")
    for p, target in zip(Pn, Tn):
        ao = infer(p, Vh, Bh, Vo, Bo)
        ao_real = unscale01(ao, lo, hi)
        t_real = unscale01(target, lo, hi)
        preds_mlp.append(ao_real)
        print(f"Вход: ({unscale01(p[0], lo, hi):.1f}, "
              f"{unscale01(p[1], lo, hi):.1f}), "
              f"Ожидалось: {t_real:.1f}, Получено: {ao_real:.4f}")

    print("Точность MLP:", hit_rate(preds_mlp, y_raw, lo, hi))
    draw_error_curve(hist_mlp, "Ошибка MLP (Вариант 7)")

    Ws, Bs, hist_slp = fit_single_layer(
        Pn, Tn, lr=0.5, tol=0.005, n_epochs=20000
    )

    print("\n=== Однослойный персептрон ===")
    print("Эпох обучения:", len(hist_slp))
    print("Финальная суммарная ошибка Es:", hist_slp[-1])

    preds_slp = []
    print("\nОтветы персептрона:")
    for p, target in zip(Pn, Tn):
        z = np.dot(p, Ws) - Bs
        a = phi(z)
        a_real = unscale01(a, lo, hi)
        t_real = unscale01(target, lo, hi)
        preds_slp.append(a_real)
        print(f"Вход: ({unscale01(p[0], lo, hi):.1f}, "
              f"{unscale01(p[1], lo, hi):.1f}), "
              f"Ожидалось: {t_real:.1f}, Получено: {a_real:.4f}")

    print("Точность SLP:", hit_rate(preds_slp, y_raw, lo, hi))
    draw_error_curve(hist_slp, "Ошибка однослойного персептрона (Вариант 7)")

    print("\n=== Режим функционирования ===")
    print("Введите пару чисел A B из [-10; 10]. Для выхода введите 'q'.")

    while True:
        raw = input("A B: ").strip()
        if raw.lower() == "q":
            break

        try:
            u, v = map(float, raw.split())
        except ValueError:
            print("Нужно ввести два числа.")
            continue

        if not (-10 <= u <= 10 and -10 <= v <= 10):
            print("Числа должны быть в диапазоне [-10; 10].")
            continue

        p = scale01(np.array([u, v]), lo, hi)
        ao = infer(p, Vh, Bh, Vo, Bo)
        ao_real = unscale01(ao, lo, hi)

        cls = lo if abs(ao_real - lo) < abs(ao_real - hi) else hi

        print(f"Выход сети: {ao_real:.4f}")
        print(f"Ближайший класс: {cls}")

if __name__ == "__main__":
    run()