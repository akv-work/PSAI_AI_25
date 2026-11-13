import numpy as np
import matplotlib.pyplot as plt
C0 = 3.0
C1 = -8.0
TARGET_ERROR = 0.01
MAX_EPOCHS = 10000
LEARNING_RATE = 0.5
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))
def sigmoid_derivative(s):
    return s * (1.0 - s)
def normalize(val, c_min=C1, c_max=C0):
    return (val - c_min) / (c_max - c_min)
def denormalize(val, c_min=C1, c_max=C0):
    return val * (c_max - c_min) + c_min
X_raw = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]])
y_raw = np.array([C0, C1, C1, C0]).reshape(-1, 1)
X_norm = normalize(X_raw)
y_norm = normalize(y_raw)
class MLP:
    def __init__(self, lr):
        self.lr = lr
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.random.randn(1, 2) * 0.5
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.random.randn(1, 1) * 0.5
    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    def train_step_online(self, x, y_true):
        pred = self.forward(x)
        error = y_true - pred
        delta2 = error * sigmoid_derivative(pred)
        delta1 = (delta2 @ self.W2.T) * sigmoid_derivative(self.a1)
        self.W2 += self.lr * (self.a1.T @ delta2)
        self.b2 += self.lr * delta2
        self.W1 += self.lr * (x.T @ delta1)
        self.b1 += self.lr * delta1
        return 0.5 * np.sum(error ** 2)
class SLP:
    def __init__(self, lr):
        self.lr = lr
        self.W = np.random.randn(2, 1) * 0.5
        self.b = np.random.randn(1, 1) * 0.5
    def forward(self, x):
        self.z = x @ self.W + self.b
        self.a = sigmoid(self.z)
        return self.a
    def train_step_online(self, x, y_true):
        pred = self.forward(x)
        error = y_true - pred
        delta = error * sigmoid_derivative(pred)
        self.W += self.lr * (x.T @ delta)
        self.b += self.lr * delta
        return 0.5 * np.sum(error ** 2)
def train(model, X, y, name):
    history = []
    for epoch in range(MAX_EPOCHS):
        idx = np.random.permutation(len(X))
        ep_err = 0.0
        for i in idx:
            ep_err += model.train_step_online(X[i:i+1], y[i:i+1])
        history.append(ep_err)
        if ep_err <= TARGET_ERROR:
            print(f"[{name}] Сходимость на эпохе {epoch+1}, Es = {ep_err:.5f}")
            return history
    print(f"[{name}] НЕ сошёлся за {MAX_EPOCHS} эпох, Es = {history[-1]:.5f}")
    return history
np.random.seed(5)
print("Обучение MLP 2-2-1...")
mlp = MLP(LEARNING_RATE)
mlp_hist = train(mlp, X_norm, y_norm, "MLP")
print("\nОбучение SLP 2-1...")
slp = SLP(LEARNING_RATE)
slp_hist = train(slp, X_norm, y_norm, "SLP")
def accuracy(model):
    correct = 0
    for i in range(4):
        out_norm = model.forward(X_norm[i:i+1])[0, 0]
        out_real = denormalize(out_norm)
        pred_class = C0 if abs(out_real - C0) < abs(out_real - C1) else C1
        if pred_class == float(y_raw[i, 0]):
            correct += 1
    return correct / 4
print(f"\nAccuracy MLP: {accuracy(mlp)*100:.1f}%")
print(f"Accuracy SLP: {accuracy(slp)*100:.1f}%")
def run_mode(model, name, points):
    print(f"\n--- {name} ---")
    print(f"{'A':>5} | {'B':>5} | {'y_hat':>8} | {'класс':>6}")
    print("-" * 34)
    for a, b in points:
        x = normalize(np.array([[a, b]], dtype=float))
        out_norm = model.forward(x)[0, 0]
        out_real = denormalize(out_norm)
        pred_class = C0 if abs(out_real - C0) < abs(out_real - C1) else C1
        print(f"{a:>5} | {b:>5} | {out_real:>8.2f} | {pred_class:>6}")
test_points = [(3, 3), (3, -8), (-8, 3), (-8, -8), (0, 0), (5, -5), (-2, 7), (10, -10), (1, 1), (-9, -9)]
run_mode(mlp, "MLP 2-2-1", test_points)
run_mode(slp, "SLP 2-1", test_points)
plt.figure(figsize=(10, 5))
plt.plot(mlp_hist, label="MLP 2-2-1", color="blue", linewidth=2)
plt.plot(slp_hist, label="SLP 2-1", color="red", linestyle="--", linewidth=2)
plt.axhline(TARGET_ERROR, color="green", linestyle=":", label="Порог ошибки")
plt.title("Динамика суммарной ошибки (Вариант 9: c0=3, c1=-8)")
plt.xlabel("Эпоха")
plt.ylabel("Es")
plt.legend()
plt.grid(alpha=0.3)
plt.xlim(0, min(3000, MAX_EPOCHS))
plt.show()