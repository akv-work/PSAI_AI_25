import numpy as np
import matplotlib.pyplot as plt

C0 = 3.0
C1 = -8.0
TARGET_ERROR_MSE = 0.05
TARGET_ERROR_BCE = 0.05
MAX_EPOCHS = 10000
LEARNING_RATE = 0.5

X_raw = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]], dtype=float)
y_raw = np.array([[C0], [C1], [C1], [C0]], dtype=float)


def normalize_x(X):
    return (C0 - X) / (C0 - C1)


def normalize_y(y):
    return (C0 - y) / (C0 - C1)


def denormalize_y(y_norm):
    return C0 - y_norm * (C0 - C1)


X_norm = normalize_x(X_raw)
y_norm = normalize_y(y_raw)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(s):
    return s * (1.0 - s)


class MLP_MSE:
    def __init__(self, lr, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = lr
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.random.randn(1, 2) * 0.5
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.random.randn(1, 1) * 0.5

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step(self, x, y_true):
        pred = self.forward(x)
        error = y_true - pred
        delta2 = error * sigmoid_derivative(pred)
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)

        self.W2 += self.lr * np.dot(self.a1.T, delta2)
        self.b2 += self.lr * delta2
        self.W1 += self.lr * np.dot(x.T, delta1)
        self.b1 += self.lr * delta1
        return 0.5 * np.sum(error ** 2)


class MLP_BCE:
    def __init__(self, lr, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = lr
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.random.randn(1, 2) * 0.5
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.random.randn(1, 1) * 0.5

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step(self, x, y_true):
        pred = self.forward(x)
        pred_clipped = np.clip(pred, 1e-15, 1 - 1e-15)
        delta2 = y_true - pred
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)

        self.W2 += self.lr * np.dot(self.a1.T, delta2)
        self.b2 += self.lr * delta2
        self.W1 += self.lr * np.dot(x.T, delta1)
        self.b1 += self.lr * delta1

        bce = -(y_true * np.log(pred_clipped) + (1 - y_true) * np.log(1 - pred_clipped))
        return np.sum(bce)


def train_model(model_class, target_error, seed=None):
    model = model_class(lr=LEARNING_RATE, seed=seed)
    history = []
    for epoch in range(MAX_EPOCHS):
        indices = np.arange(len(X_norm))
        np.random.shuffle(indices)
        epoch_error = 0.0
        for i in indices:
            epoch_error += model.train_step(X_norm[i:i + 1], y_norm[i:i + 1])
        history.append(epoch_error)
        if epoch_error <= target_error:
            return model, history, epoch + 1, True
    return model, history, MAX_EPOCHS, False


def calc_mae(model):
    mae = 0.0
    for i in range(len(X_norm)):
        pred_norm = model.forward(X_norm[i:i + 1])[0][0]
        pred_real = denormalize_y(pred_norm)
        mae += abs(pred_real - y_raw[i][0])
    return mae / len(X_norm)


def calc_accuracy(model):
    correct = 0
    for i in range(len(X_norm)):
        pred_norm = model.forward(X_norm[i:i + 1])[0][0]
        pred_real = denormalize_y(pred_norm)
        true_val = y_raw[i][0]
        pred_class = C0 if abs(pred_real - C0) < abs(pred_real - C1) else C1
        true_class = C0 if abs(true_val - C0) < abs(true_val - C1) else C1
        if pred_class == true_class:
            correct += 1
    return correct / len(X_norm) * 100


print("=== Обучение эталонных моделей (seed=42) ===")
model_mse, hist_mse, ep_mse, conv_mse = train_model(MLP_MSE, TARGET_ERROR_MSE, seed=42)
print(f"MSE: эпох = {ep_mse}, Es = {hist_mse[-1]:.5f}, сошлась = {conv_mse}")

model_bce, hist_bce, ep_bce, conv_bce = train_model(MLP_BCE, TARGET_ERROR_BCE, seed=42)
print(f"BCE: эпох = {ep_bce}, Es = {hist_bce[-1]:.5f}, сошлась = {conv_bce}")

print("\n=== Серия из 5 запусков ===")
seeds = [10, 42, 123, 777, 2024]
mse_epochs, bce_epochs = [], []
mse_conv_list, bce_conv_list = [], []

for s in seeds:
    _, _, e_mse, c_mse = train_model(MLP_MSE, TARGET_ERROR_MSE, seed=s)
    _, _, e_bce, c_bce = train_model(MLP_BCE, TARGET_ERROR_BCE, seed=s)
    mse_epochs.append(e_mse)
    bce_epochs.append(e_bce)
    mse_conv_list.append(c_mse)
    bce_conv_list.append(c_bce)
    print(f"seed={s:4d}: MSE {'V' if c_mse else 'X'} {e_mse} эпох, BCE {'V' if c_bce else 'X'} {e_bce} эпох")

mae_mse = calc_mae(model_mse)
mae_bce = calc_mae(model_bce)
acc_mse = calc_accuracy(model_mse)
acc_bce = calc_accuracy(model_bce)

print(f"\n== Итоговые метрики ==")
print(f"MSE: Accuracy = {acc_mse:.1f}%, MAE = {mae_mse:.4f}")
print(f"BCE: Accuracy = {acc_bce:.1f}%, MAE = {mae_bce:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(hist_mse, label='Конфигурация A (MSE)', color='blue', linewidth=2)
plt.plot(hist_bce, label='Конфигурация B (BCE)', color='red', linewidth=2)
plt.axhline(TARGET_ERROR_MSE, color='blue', linestyle=':', linewidth=1.5, label=f'Порог MSE ({TARGET_ERROR_MSE})')
plt.axhline(TARGET_ERROR_BCE, color='red', linestyle=':', linewidth=1.5, label=f'Порог BCE ({TARGET_ERROR_BCE})')
plt.yscale('log')
plt.title('Сходимость: Es от номера эпохи (представительный запуск)')
plt.xlabel('Эпоха')
plt.ylabel('Суммарная ошибка Es (лог. шкала)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(3000, MAX_EPOCHS))
plt.show()

x = np.arange(len(seeds))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
bars_mse = ax.bar(x - width / 2, mse_epochs, width, label='MSE', color='skyblue', edgecolor='black')
bars_bce = ax.bar(x + width / 2, bce_epochs, width, label='BCE', color='lightcoral', edgecolor='black')

for bar, conv in zip(bars_mse, mse_conv_list):
    if not conv:
        bar.set_hatch('///')
        bar.set_label('MSE (не сошлась)')
for bar, conv in zip(bars_bce, bce_conv_list):
    if not conv:
        bar.set_hatch('///')
        bar.set_label('BCE (не сошлась)')

ax.set_title('Разброс числа эпох по 5 запускам\n(штриховка = не сошлась за MAX_EPOCHS)')
ax.set_xlabel('Номер запуска (seed)')
ax.set_ylabel('Эпохи')
ax.set_xticks(x)
ax.set_xticklabels([f'Run {i + 1}\n(seed={s})' for i, s in enumerate(seeds)])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

xx, yy = np.meshgrid(np.linspace(C1 - 2, C0 + 2, 300), np.linspace(C1 - 2, C0 + 2, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_norm = normalize_x(grid_points)

Z_mse = np.array([model_mse.forward(p.reshape(1, 2))[0][0] for p in grid_norm]).reshape(xx.shape)
Z_bce = np.array([model_bce.forward(p.reshape(1, 2))[0][0] for p in grid_norm]).reshape(xx.shape)

Z_mse_real = denormalize_y(Z_mse)
Z_bce_real = denormalize_y(Z_bce)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
cmap = plt.cm.coolwarm

c1 = ax1.contourf(xx, yy, Z_mse_real, levels=100, cmap=cmap, alpha=0.85)
ax1.contour(xx, yy, Z_mse_real, levels=12, colors='k', linewidths=0.4, alpha=0.35)
ax1.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw.flatten(), cmap=cmap, edgecolors='white', linewidths=2, s=180, marker='o',
            label='Точки XOR', zorder=5)
ax1.set_title('Разделяющая поверхность (MSE)')
ax1.set_xlabel('Вход A')
ax1.set_ylabel('Вход B')
ax1.legend(loc='upper right', framealpha=0.9)
fig.colorbar(c1, ax=ax1, label='Выход сети (шкала [c1; c0])')

c2 = ax2.contourf(xx, yy, Z_bce_real, levels=100, cmap=cmap, alpha=0.85)
ax2.contour(xx, yy, Z_bce_real, levels=12, colors='k', linewidths=0.4, alpha=0.35)
ax2.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw.flatten(), cmap=cmap, edgecolors='white', linewidths=2, s=180, marker='o',
            label='Точки XOR', zorder=5)
ax2.set_title('Разделяющая поверхность (BCE)')
ax2.set_xlabel('Вход A')
ax2.set_ylabel('Вход B')
ax2.legend(loc='upper right', framealpha=0.9)
fig.colorbar(c2, ax=ax2, label='Выход сети (шкала [c1; c0])')

plt.suptitle('Визуализация разделяющей поверхности (вариант 9: c0=3, c1=-8)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
bars = plt.bar(['Конфигурация A\n(MSE)', 'Конфигурация B\n(BCE)'], [mae_mse, mae_bce], color=['skyblue', 'lightcoral'],
               width=0.6, edgecolor='black')
plt.title('Сравнение точности восстановления шкалы\nСредняя абсолютная ошибка (MAE) в шкале [c1; c0]', fontsize=12,
          fontweight='bold')
plt.ylabel('MAE', fontsize=11)
plt.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, [mae_mse, mae_bce]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', va='bottom',
             fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n== РЕЖИМ ФУНКЦИОНИРОВАНИЯ (BCE) ==")
print(f"{'Вход (A,B)':<15} | {'y_hat':<10} | {'y в [c1;c0]':<12} | Ближайший класс")
print("=" * 65)

test_cases = [(C0, C0), (C0, C1), (C1, C0), (C1, C1), (0, 0), (5, -5), (-5, 5)]
for a, b in test_cases:
    x_in = normalize_x(np.array([[a, b]]))
    out_norm = model_bce.forward(x_in)[0][0]
    out_real = denormalize_y(out_norm)
    closest = C0 if abs(out_real - C0) < abs(out_real - C1) else C1
    print(f"({a:>4}, {b:>4}) | {out_norm:>8.4f} | {out_real:>10.4f} | {closest}")

print("\n--- Интерактивный режим (exit для выхода) ---")
while True:
    try:
        s = input("Вход A B: ")
        if s.lower() == 'exit':
            break
        a, b = map(float, s.split())
        x_in = normalize_x(np.array([[a, b]]))
        out_norm = model_bce.forward(x_in)[0][0]
        out_real = denormalize_y(out_norm)
        closest = C0 if abs(out_real - C0) < abs(out_real - C1) else C1

        print(f"Норм. выход: {out_norm:.4f}")
        print(f"Выход в шкале [{C0}; {C1}]: {out_real:.4f}")
        print(f"Ближайший класс: {closest}\n")
    except ValueError:
        print("Введите два числа через пробел или 'exit'.")