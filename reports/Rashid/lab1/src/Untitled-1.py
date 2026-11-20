import numpy as np
C0, C1 = -10.0, 9.0

EE = 0.001
MAX_EPOCHS = 10000
LEARNING_RATE = 1.0
SEED = 0

def sigmoid(s):
    s = np.clip(s, -60, 60)
    return 1.0 / (1.0 + np.exp(-s))

def normalize_input(x):
    return (x - C0) / (C1 - C0)

def denormalize_output(y):
    return C0 + y * (C1 - C0)

# ================== Обучающая выборка (XOR) ==================
X_raw = np.array([
    [-10, -10],
    [-10,   9],
    [  9, -10],
    [  9,   9]
], dtype=float)

T_raw = np.array([-10, 9, 9, -10], dtype=float)

X = normalize_input(X_raw)
T = normalize_input(T_raw)

# ================== Прямой проход МСП 2-2-1 ==================
def mlp_forward(x, W1, b1, W2, b2):
    h = sigmoid(x @ W1 + b1)     # скрытый слой (2 нейрона)
    y = sigmoid(h @ W2 + b2)     # выходной нейрон
    return h, y

# ================== Обучение МСП ==================
def train_mlp():
    rng = np.random.default_rng(SEED)
    W1 = rng.uniform(-0.5, 0.5, (2, 2))
    b1 = rng.uniform(-0.5, 0.5, 2)
    W2 = rng.uniform(-0.5, 0.5, (2, 1))
    b2 = rng.uniform(-0.5, 0.5, 1)

    Es = np.inf
    for epoch in range(1, MAX_EPOCHS + 1):
        # онлайн-режим: веса обновляются после каждого примера
        for x, t in zip(X, T):
            h, y = mlp_forward(x, W1, b1, W2, b2)
            delta_out = (y - t) * y * (1.0 - y)
            delta_hidden = (W2[:, 0] * delta_out[0]) * h * (1.0 - h)

            W2 -= LEARNING_RATE * np.outer(h, delta_out)
            b2 -= LEARNING_RATE * delta_out
            W1 -= LEARNING_RATE * np.outer(x, delta_hidden)
            b1 -= LEARNING_RATE * delta_hidden

        outputs = np.array([mlp_forward(x, W1, b1, W2, b2)[1][0] for x in X])
        Es = 0.5 * np.sum((T - outputs) ** 2)
        if Es <= EE:
            return epoch, Es, W1, b1, W2, b2
    return MAX_EPOCHS, Es, W1, b1, W2, b2

# ================== Обучение однослойного персептрона ==================
def train_single_layer():
    rng = np.random.default_rng(SEED)
    W = rng.uniform(-0.5, 0.5, 2)
    b = float(rng.uniform(-0.5, 0.5))

    Es = np.inf
    for epoch in range(1, MAX_EPOCHS + 1):
        for x, t in zip(X, T):
            y = sigmoid(x @ W + b)
            delta = (y - t) * y * (1.0 - y)
            W -= LEARNING_RATE * delta * x
            b -= LEARNING_RATE * delta

        outputs = sigmoid(X @ W + b)
        Es = 0.5 * np.sum((T - outputs) ** 2)
        if Es <= EE:
            return epoch, Es, W, b
    return MAX_EPOCHS, Es, W, b

def class_by_nearest(value):
    return C0 if abs(value - C0) < abs(value - C1) else C1

# ================== Обучение ==================
mlp_epoch, mlp_Es, W1, b1, W2, b2 = train_mlp()
slp_epoch, slp_Es, Ws, bs = train_single_layer()

# ================== Оценка качества ==================
mlp_norm = np.array([mlp_forward(x, W1, b1, W2, b2)[1][0] for x in X])
mlp_raw = denormalize_output(mlp_norm)
mlp_classes = np.array([class_by_nearest(v) for v in mlp_raw])
mlp_accuracy = np.mean(mlp_classes == T_raw)

slp_norm = sigmoid(X @ Ws + bs)
slp_raw = denormalize_output(slp_norm)
slp_classes = np.array([class_by_nearest(v) for v in slp_raw])
slp_accuracy = np.mean(slp_classes == T_raw)

print("=== МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН 2-2-1 ===")
print("Эпохи:", mlp_epoch)
print("Es:", mlp_Es)
print("Accuracy:", mlp_accuracy)
print("Выходы:", mlp_raw)

print("\n=== ОДНОСЛОЙНЫЙ ПЕРСЕПТРОН ===")
print("Эпохи:", slp_epoch)
print("Es:", slp_Es)
print("Accuracy:", slp_accuracy)
print("Выходы:", slp_raw)

# ================== Режим функционирования ==================
def predict(A, B):
    raw = np.array([A, B], dtype=float)
    x = normalize_input(raw)
    _, y = mlp_forward(x, W1, b1, W2, b2)
    value = denormalize_output(y[0])
    cls = class_by_nearest(value)
    return value, cls

while True:
    try:
        A = float(input("\nВведите A [-10; 10] (q для выхода): "))
        B = float(input("Введите B [-10; 10]: "))
        if not (-10 <= A <= 10 and -10 <= B <= 10):
            print("Ошибка: A и B должны находиться в диапазоне [-10; 10].")
            continue
        value, cls = predict(A, B)
        print(f"Выход сети: {value:.6f}")
        print(f"Ближайший класс: {cls:g}")
    except ValueError:
        break