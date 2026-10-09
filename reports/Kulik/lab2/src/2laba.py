import numpy as np
import matplotlib.pyplot as plt

c_zero, c_one = -4, -9

data = np.array([
    [-4, -4, -4],
    [-4, -9, -9],
    [-9, -4, -9],
    [-9, -9, -4]
], dtype=float)

X_in = data[:, :2] / 10
T_target = ((data[:, 2] - c_zero) / (c_one - c_zero)).reshape(-1, 1)

lr_rate = 0.5
max_epochs = 10000
err_limit = 0.01

def phi(z):
    """Сигмоида."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def dphi(a):
    """Производная сигмоиды по её выходу."""
    return a * (1.0 - a)

def unscale(v):
    """Обратный пересчёт из (0;1) в шкалу [c0; c1]."""
    return v * (c_one - c_zero) + c_zero

def nearest_class(v):
    """Ближайший класс c0 или c1."""
    return c_zero if abs(v - c_zero) < abs(v - c_one) else c_one

def init_weights(seed):
    """Инициализация малых случайных весов для заданного seed."""
    rng = np.random.RandomState(seed)
    Vh = rng.randn(2, 2) * 0.1
    Vo = rng.randn(2, 1) * 0.1
    Bh = np.zeros((1, 2))
    Bo = np.zeros((1, 1))
    return Vh, Vo, Bh, Bo

def feedforward(x, Vh, Vo, Bh, Bo):
    """Прямой проход 2-2-1."""
    ah = phi(x @ Vh + Bh)
    ao = phi(ah @ Vo + Bo)
    return ah, ao

def train_mse(seed):
    Vh, Vo, Bh, Bo = init_weights(seed)
    hist = []

    for _ in range(max_epochs):
        err = 0.0

        for i in range(len(X_in)):
            x, y = X_in[i:i+1], T_target[i:i+1]
            ah, ao = feedforward(x, Vh, Vo, Bh, Bo)

            err += np.sum((y - ao) ** 2)

            d_out = (y - ao) * dphi(ao)
            d_hid = d_out @ Vo.T * dphi(ah)

            Vo += lr_rate * ah.T @ d_out
            Bo += lr_rate * d_out
            Vh += lr_rate * x.T @ d_hid
            Bh += lr_rate * d_hid

        hist.append(err)
        if err <= err_limit:
            break

    return Vh, Vo, Bh, Bo, hist

def train_bce(seed):
    Vh, Vo, Bh, Bo = init_weights(seed)
    hist = []

    for _ in range(max_epochs):
        err = 0.0

        for i in range(len(X_in)):
            x, y = X_in[i:i+1], T_target[i:i+1]
            ah, ao = feedforward(x, Vh, Vo, Bh, Bo)

            p = np.clip(ao, 1e-12, 1 - 1e-12)
            err += np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))

            d_out = ao - y
            d_hid = d_out @ Vo.T * dphi(ah)

            Vo -= lr_rate * ah.T @ d_out
            Bo -= lr_rate * d_out
            Vh -= lr_rate * x.T @ d_hid
            Bh -= lr_rate * d_hid

        hist.append(err)
        if err <= err_limit:
            break

    return Vh, Vo, Bh, Bo, hist

def predict_norm(a, b, model):
    """Нормированный выход сети для точки (a, b)."""
    Vh, Vo, Bh, Bo = model
    x = np.array([[a / 10, b / 10]])
    _, ao = feedforward(x, Vh, Vo, Bh, Bo)
    return ao[0, 0]

def predict_real(a, b, model):
    """Выход сети в исходной шкале [c0; c1]."""
    return unscale(predict_norm(a, b, model))

def compute_metrics(model):
    """Accuracy и MAE в исходной шкале."""
    preds = np.array([predict_real(row[0], row[1], model) for row in data])
    classes = np.array([nearest_class(v) for v in preds])
    accuracy = np.mean(classes == data[:, 2])
    mae = np.mean(np.abs(preds - data[:, 2]))
    return accuracy, mae

seeds = [1, 2, 3, 4, 5]
compare_seed = 3

runs_mse = []
runs_bce = []

print("Лабораторная работа №2")
print("Вариант 7: c0 = -4, c1 = -9")

print("\n--- Конфигурация А (MSE) ---")
for s in seeds:
    *model, hist = train_mse(s)
    acc, mae = compute_metrics(model)
    runs_mse.append((s, model, hist))
    print(f"seed={s}: эпох={len(hist)}, ошибка={hist[-1]:.6f}, "
          f"accuracy={acc:.2f}, MAE={mae:.4f}")

print("\n--- Конфигурация Б (BCE) ---")
for s in seeds:
    *model, hist = train_bce(s)
    acc, mae = compute_metrics(model)
    runs_bce.append((s, model, hist))
    print(f"seed={s}: эпох={len(hist)}, ошибка={hist[-1]:.6f}, "
          f"accuracy={acc:.2f}, MAE={mae:.4f}")

_, model_mse, hist_mse = next(x for x in runs_mse if x[0] == compare_seed)
_, model_bce, hist_bce = next(x for x in runs_bce if x[0] == compare_seed)

acc_mse, mae_mse = compute_metrics(model_mse)
acc_bce, mae_bce = compute_metrics(model_bce)

print("\n=== Сравнение на seed =", compare_seed, "===")
print("MSE:  эпох =", len(hist_mse),
      "| ошибка =", round(hist_mse[-1], 6),
      "| accuracy =", acc_mse,
      "| MAE =", round(mae_mse, 6))
print("BCE:  эпох =", len(hist_bce),
      "| ошибка =", round(hist_bce[-1], 6),
      "| accuracy =", acc_bce,
      "| MAE =", round(mae_bce, 6))

def print_predictions(model, title):
    print(f"\nПредсказания {title}")
    for row in data:
        norm = predict_norm(row[0], row[1], model)
        real = unscale(norm)
        cls = nearest_class(real)
        print(f"({row[0]:5.0f},{row[1]:5.0f}) | "
              f"y_norm={norm:.4f} | y={real:7.3f} | "
              f"class={cls:5.0f} | expected={row[2]:5.0f}")

print_predictions(model_mse, "MSE")
print_predictions(model_bce, "BCE")

plt.figure(figsize=(10, 5))
plt.plot(hist_mse, label="MSE")
plt.plot(hist_bce, label="BCE")
plt.axhline(err_limit, linestyle="--", label="Ee = 0.01")
plt.xlabel("Номер эпохи")
plt.ylabel("Суммарная ошибка Es")
plt.title(f"Сходимость MSE и BCE (seed={compare_seed}, вариант 7)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
x_pos = np.arange(len(seeds))
bar_w = 0.35

epochs_mse = [len(h) for _, _, h in runs_mse]
epochs_bce = [len(h) for _, _, h in runs_bce]

hatch_mse = ["" if h[-1] <= err_limit else "//" for _, _, h in runs_mse]
hatch_bce = ["" if h[-1] <= err_limit else "//" for _, _, h in runs_bce]

for i in range(len(seeds)):
    plt.bar(x_pos[i] - bar_w / 2, epochs_mse[i], bar_w,
            label="MSE" if i == 0 else "",
            hatch=hatch_mse[i], color="steelblue")

for i in range(len(seeds)):
    plt.bar(x_pos[i] + bar_w / 2, epochs_bce[i], bar_w,
            label="BCE" if i == 0 else "",
            hatch=hatch_bce[i], color="orange")

plt.xlabel("Seed")
plt.ylabel("Количество эпох")
plt.title("Сравнение числа эпох по 5 запускам (вариант 7)")
plt.xticks(x_pos, seeds)
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

def plot_boundary(model, title):
    grid = np.linspace(-10, 10, 200)
    A, B = np.meshgrid(grid, grid)

    Z = np.array([[predict_real(a, b, model) for a in grid] for b in grid])

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(A, B, Z, levels=30, alpha=0.8)
    plt.colorbar(contour, label="Выход сети y (шкала [c0; c1])")

    plt.contour(A, B, Z,
                levels=[(c_zero + c_one) / 2],
                colors="black", linewidths=2)

    plt.scatter(data[data[:, 2] == c_zero, 0],
                data[data[:, 2] == c_zero, 1],
                color="blue", marker="o", s=100,
                label=f"Класс {c_zero}")

    plt.scatter(data[data[:, 2] == c_one, 0],
                data[data[:, 2] == c_one, 1],
                color="red", marker="o", s=100,
                label=f"Класс {c_one}")

    plt.xlabel("A")
    plt.ylabel("B")
    plt.title(title)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_boundary(model_mse, f"Разделяющая граница MSE (seed={compare_seed})")
plot_boundary(model_bce, f"Разделяющая граница BCE (seed={compare_seed})")

plt.figure(figsize=(7, 5))
plt.bar(["MSE", "BCE"], [mae_mse, mae_bce], color=["steelblue", "orange"])
plt.ylabel("Средняя абсолютная ошибка (MAE)")
plt.title("Сравнение точности восстановления шкалы [c0; c1]")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

print("\n=== Режим функционирования (BCE) ===")
print("Введите A B из диапазона [-10; 10].")
print("Для выхода введите: q")

while True:
    raw = input("A B: ").strip()
    if raw == "q":
        break

    try:
        a, b = map(float, raw.split())
    except ValueError:
        print("Нужно ввести два числа через пробел.")
        continue

    if not (-10 <= a <= 10 and -10 <= b <= 10):
        print("A и B должны быть в диапазоне [-10; 10].")
        continue

    norm_val = predict_norm(a, b, model_bce)
    real_val = unscale(norm_val)
    cls_val = nearest_class(real_val)

    print(f"Нормализованный выход: {norm_val:.4f}")
    print(f"Выход в шкале [c0; c1]: {real_val:.4f}")
    print(f"Ближайший класс: {cls_val:g}")