import numpy as np
import matplotlib.pyplot as plt
import copy

C0 = -3.0
C1 = -8.0

X_raw = np.array([[-3.0, -3.0], [-3.0, -8.0], [-8.0, -3.0], [-8.0, -8.0]])
Y_raw = np.array([-3.0, -8.0, -8.0, -3.0])

def scale_input(val):
    return val / 10.0

def scale_target_mse(val):
    return 0.1 + (val - C1) / (C0 - C1) * 0.8

def inverse_scale_mse(val_scaled):
    return C1 + (val_scaled - 0.1) / 0.8 * (C0 - C1)

def scale_target_bce(val):
    return (val - C1) / (C0 - C1)

def inverse_scale_bce(val_scaled):
    return val_scaled * (C0 - C1) + C1

X = np.array([[scale_input(a), scale_input(b)] for a, b in X_raw])
Y_mse = np.array([scale_target_mse(y) for y in Y_raw]).reshape(-1, 1)
Y_bce = np.array([scale_target_bce(y) for y in Y_raw]).reshape(-1, 1)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

class MLP:
    def __init__(self, lr=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = lr
        self.w1 = np.random.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, 2))
        self.w2 = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def forward(self, x):
        h = sigmoid(np.dot(x, self.w1) + self.b1)
        out = sigmoid(np.dot(h, self.w2) + self.b2)
        return h, out

    def train(self, X, Y, mode='mse', epochs=5000, Ee=0.05):
        history = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = X[i].reshape(1, -1)
                y_i = Y[i].reshape(1, 1)
                
                h, output = self.forward(x_i)
                
                if mode == 'mse':
                    error_out = y_i - output
                    total_error += np.sum(error_out ** 2) / 2
                    delta_out = error_out * sigmoid_deriv(output)
                elif mode == 'bce':
                    out_clipped = np.clip(output, 1e-15, 1 - 1e-15)
                    total_error += -np.sum(y_i * np.log(out_clipped) + (1 - y_i) * np.log(1 - out_clipped))
                    delta_out = y_i - output
                
                error_hidden = np.dot(delta_out, self.w2.T)
                delta_hidden = error_hidden * sigmoid_deriv(h)
                
                self.w2 += self.lr * np.dot(h.T, delta_out)
                self.b2 += self.lr * delta_out.flatten()
                self.w1 += self.lr * np.dot(x_i.T, delta_hidden)
                self.b1 += self.lr * delta_hidden.flatten()
                
            history.append(total_error)
            if total_error <= Ee:
                return epoch + 1, total_error, history
        return epochs, total_error, history

def evaluate(model, X, Y_raw, mode):
    correct = 0
    abs_errors = []
    
    for i in range(len(X)):
        x_i = X[i].reshape(1, -1)
        _, out = model.forward(x_i)
        
        y_pred_scaled = out[0][0]
        if mode == 'mse':
            y_pred_real = inverse_scale_mse(y_pred_scaled)
        else:
            y_pred_real = inverse_scale_bce(y_pred_scaled)
            
        y_true = Y_raw[i]
        abs_errors.append(abs(y_pred_real - y_true))
        
        pred_class = C0 if abs(y_pred_real - C0) < abs(y_pred_real - C1) else C1
        if pred_class == y_true:
            correct += 1
            
    return (correct / len(X)) * 100, np.mean(abs_errors)

seeds = [42, 7, 123, 99, 2024]
max_epochs = 5000
Ee_mse = 0.05
Ee_bce = 0.1

results_A = []
results_B = []
best_models = {}

print("--- ЗАПУСК ЭКСПЕРИМЕНТОВ (5 СИДОВ) ---")
for seed in seeds:
    mlp_A = MLP(lr=0.5, seed=seed)
    ep_A, err_A, hist_A = mlp_A.train(X, Y_mse, mode='mse', epochs=max_epochs, Ee=Ee_mse)
    acc_A, mae_A = evaluate(mlp_A, X, Y_raw, 'mse')
    results_A.append({'seed': seed, 'epochs': ep_A, 'error': err_A, 'acc': acc_A, 'mae': mae_A, 'converged': err_A <= Ee_mse})
    
    mlp_B = MLP(lr=0.5, seed=seed)
    ep_B, err_B, hist_B = mlp_B.train(X, Y_bce, mode='bce', epochs=max_epochs, Ee=Ee_bce)
    acc_B, mae_B = evaluate(mlp_B, X, Y_raw, 'bce')
    results_B.append({'seed': seed, 'epochs': ep_B, 'error': err_B, 'acc': acc_B, 'mae': mae_B, 'converged': err_B <= Ee_bce})
    
    if seed == 42:
        best_models['A'] = copy.deepcopy(mlp_A)
        best_models['B'] = copy.deepcopy(mlp_B)
        best_models['hist_A'] = hist_A
        best_models['hist_B'] = hist_B

print(f"{'Конфиг':<8} | {'Seed':<5} | {'Эпохи':<6} | {'Ошибка Es':<10} | {'Accuracy':<9} | {'MAE (шкала)':<12}")
print("-" * 65)
for i in range(5):
    ra, rb = results_A[i], results_B[i]
    print(f"A (MSE)  | {ra['seed']:<5} | {ra['epochs']:<6} | {ra['error']:<10.5f} | {ra['acc']:<8.1f}% | {ra['mae']:<12.4f}")
    print(f"B (BCE)  | {rb['seed']:<5} | {rb['epochs']:<6} | {rb['error']:<10.5f} | {rb['acc']:<8.1f}% | {rb['mae']:<12.4f}")
    print("-" * 65)

plt.rcParams['figure.dpi'] = 100

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(best_models['hist_A'], label='MSE (Конфигурация А)', color='blue', linewidth=2)
ax1.plot(best_models['hist_B'], label='BCE (Конфигурация Б)', color='red', linewidth=2, linestyle='--')
ax1.axhline(Ee_mse, color='blue', linestyle=':', alpha=0.5, label='Порог Ee (MSE)')
ax1.axhline(Ee_bce, color='red', linestyle=':', alpha=0.5, label='Порог Ee (BCE)')
ax1.set_title('Сходимость MLP 2-2-1 (Seed 42)')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Суммарная ошибка Es')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab2_convergence.png')

fig2, ax2 = plt.subplots(figsize=(8, 5))
x = np.arange(len(seeds))
width = 0.35
epochs_A = [r['epochs'] for r in results_A]
epochs_B = [r['epochs'] for r in results_B]

rects1 = ax2.bar(x - width/2, epochs_A, width, label='MSE (А)', color='skyblue', edgecolor='black')
rects2 = ax2.bar(x + width/2, epochs_B, width, label='BCE (Б)', color='salmon', edgecolor='black')

for i, r in enumerate(results_A):
    if not r['converged']: rects1[i].set_hatch('//')
for i, r in enumerate(results_B):
    if not r['converged']: rects2[i].set_hatch('//')

ax2.set_ylabel('Количество эпох')
ax2.set_title('Сравнение количества эпох до сходимости по 5 сидам')
ax2.set_xticks(x)
ax2.set_xticklabels([f'Seed {s}' for s in seeds])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('lab2_epochs.png')

fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
grid_X = np.c_[xx.ravel(), yy.ravel()]
grid_X_scaled = np.array([[scale_input(a), scale_input(b)] for a, b in grid_X])

Z_A = np.array([best_models['A'].forward(x.reshape(1, -1))[1][0][0] for x in grid_X_scaled])
Z_A_real = inverse_scale_mse(Z_A).reshape(xx.shape)
contour_A = ax3.contourf(xx, yy, Z_A_real, levels=20, cmap='coolwarm', alpha=0.8)
ax3.scatter(X_raw[:, 0], X_raw[:, 1], c=Y_raw, cmap='coolwarm', s=200, edgecolors='k', vmin=C1, vmax=C0)
ax3.set_title('Разделяющая поверхность (MSE)')
ax3.set_xlabel('A'); ax3.set_ylabel('B')
plt.colorbar(contour_A, ax=ax3, label='Выходное значение ŷ')

Z_B = np.array([best_models['B'].forward(x.reshape(1, -1))[1][0][0] for x in grid_X_scaled])
Z_B_real = inverse_scale_bce(Z_B).reshape(xx.shape)
contour_B = ax4.contourf(xx, yy, Z_B_real, levels=20, cmap='coolwarm', alpha=0.8)
ax4.scatter(X_raw[:, 0], X_raw[:, 1], c=Y_raw, cmap='coolwarm', s=200, edgecolors='k', vmin=C1, vmax=C0)
ax4.set_title('Разделяющая поверхность (BCE)')
ax4.set_xlabel('A'); ax4.set_ylabel('B')
plt.colorbar(contour_B, ax=ax4, label='Выходное значение ŷ')
plt.tight_layout()
plt.savefig('lab2_surfaces.png')

fig4, ax5 = plt.subplots(figsize=(6, 4))
mae_A_mean = np.mean([r['mae'] for r in results_A])
mae_B_mean = np.mean([r['mae'] for r in results_B])
ax5.bar(['MSE (А)', 'BCE (Б)'], [mae_A_mean, mae_B_mean], color=['skyblue', 'salmon'], edgecolor='black')
ax5.set_title('Средняя абсолютная ошибка (MAE) в шкале [c0, c1]')
ax5.set_ylabel('MAE')
for i, v in enumerate([mae_A_mean, mae_B_mean]):
    ax5.text(i, v + 0.05, f"{v:.4f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('lab2_mae.png')

plt.show()

print("\n--- РЕЖИМ ФУНКЦИОНИРОВАНИЯ СЕТИ (Конфигурация Б - BCE) ---")
test_inputs = [
    (-3, -3), (-3, -8), (-8, -3), (-8, -8),
    (0, 0), (5, -5), (-5, 5), (-10, 10)
]

print(f"{'Вход A':>6} | {'Вход B':>6} | {'Выход ŷ (0-1)':>14} | {'Выход ŷ (реал)':>14} | {'Ближе к':>8}")
print("-" * 65)
for a, b in test_inputs:
    x_scaled = np.array([[scale_input(a), scale_input(b)]])
    _, out = best_models['B'].forward(x_scaled)
    y_scaled = out[0][0]
    y_real = inverse_scale_bce(y_scaled)
    
    class_name = f"C0 ({C0})" if abs(y_real - C0) < abs(y_real - C1) else f"C1 ({C1})"
    print(f"{a:>6} | {b:>6} | {y_scaled:>14.4f} | {y_real:>14.3f} | {class_name:>8}")