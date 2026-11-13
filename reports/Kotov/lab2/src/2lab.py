import numpy as np
import matplotlib.pyplot as plt


def sigmoid(s):
    s = np.clip(s, -500, 500) # Защита от переполнения
    return 1 / (1 + np.exp(-s))

def sigmoid_der(s):
    y = sigmoid(s)
    return y * (1 - y)

def norm(x, c0, c1):
    return (x - c0) / (c1 - c0)

def denorm(y, c0, c1):
    return c0 + y * (c1 - c0)

def mlp_forward(x, W1, T1, W2, T2):
    S1 = np.dot(x, W1) - T1
    h = sigmoid(S1)
    S2 = np.dot(h, W2) - T2
    y = sigmoid(S2)
    return S1, h, S2, y


def train_mlp_mse(X, y_raw, alpha=0.8, Ee=0.001, max_epochs=10000, seed=42):
    np.random.seed(seed)
    W1 = np.random.uniform(-1.0, 1.0, (2, 2))
    T1 = np.random.uniform(-1.0, 1.0, 2)
    W2 = np.random.uniform(-1.0, 1.0, 2)
    T2 = np.random.uniform(-1.0, 1.0, 1)[0]
    
    errors = []
    epoch = 0
    
    while True:
        Es = 0.0
        for xi, yi_raw in zip(X, y_raw):
            S1, h, S2, y_pred = mlp_forward(xi, W1, T1, W2, T2)
            
            # --- ОТЛИЧИЕ КОНФИГУРАЦИИ А (MSE) №1: РАСЧЕТ ОШИБКИ ---
            # Ошибка считается по MSE на СЫРЫХ данных (yi_raw).
            Es += 0.5 * (y_pred - yi_raw) ** 2
            
            # --- ОТЛИЧИЕ КОНФИГУРАЦИИ А (MSE) №2: ГРАДИЕНТ ВЫХОДНОГО СЛОЯ ---
            # Для MSE градиент должен содержать производную сигмоиды: (y^ - y) * F'(S)
            delta2 = (y_pred - yi_raw) * sigmoid_der(S2)
            
            delta1 = sigmoid_der(S1) * (W2 * delta2)
            
            W2 -= alpha * h * delta2
            T2 += alpha * delta2
            W1 -= alpha * np.outer(xi, delta1)
            T1 += alpha * delta1
            
        errors.append(Es)
        epoch += 1
        
        if Es <= Ee or epoch >= max_epochs:
            break
            
    return W1, T1, W2, T2, errors, epoch

def train_mlp_bce(X, y_norm, alpha=0.8, Ee=0.001, max_epochs=10000, seed=42):
    np.random.seed(seed)
    W1 = np.random.uniform(-1.0, 1.0, (2, 2))
    T1 = np.random.uniform(-1.0, 1.0, 2)
    W2 = np.random.uniform(-1.0, 1.0, 2)
    T2 = np.random.uniform(-1.0, 1.0, 1)[0]
    
    errors = []
    epoch = 0
    
    while True:
        Es = 0.0
        for xi, yi_norm in zip(X, y_norm):
            S1, h, S2, y_pred = mlp_forward(xi, W1, T1, W2, T2)
            
            eps = 1e-15 # Защита от логарифма нуля
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            
            # --- ОТЛИЧИЕ КОНФИГУРАЦИИ Б (BCE) №1: РАСЧЕТ ОШИБКИ ---
            # Считаем бинарную кросс-энтропию на нормализованных целевых значениях (0 и 1)
            loss = - (yi_norm * np.log(y_pred_clipped) + (1 - yi_norm) * np.log(1 - y_pred_clipped))
            Es += loss
            
            # --- ОТЛИЧИЕ КОНФИГУРАЦИИ Б (BCE) №2: ГРАДИЕНТ ВЫХОДНОГО СЛОЯ ---
            # Производная функции потерь BCE и производная сигмоиды сокращаются. Остается только простая разность: (y^ - y)
            delta2 = (y_pred - yi_norm)
            
            delta1 = sigmoid_der(S1) * (W2 * delta2)
            
            W2 -= alpha * h * delta2
            T2 += alpha * delta2
            W1 -= alpha * np.outer(xi, delta1)
            T1 += alpha * delta1
            
        errors.append(Es)
        epoch += 1
        
        if Es <= Ee or epoch >= max_epochs:
            break
            
    return W1, T1, W2, T2, errors, epoch

def evaluate_network(X, y_raw, y_norm, c0, c1, W1, T1, W2, T2, is_bce):
    correct = 0
    mae = 0.0
    
    for i, xi in enumerate(X):
        _, _, _, y_pred = mlp_forward(xi, W1, T1, W2, T2)
        
        if is_bce:
            y_pred_real = denorm(y_pred, c0, c1)
            pred_class = c1 if y_pred >= 0.5 else c0
        else:
            y_pred_real = y_pred
            pred_class = c1 if y_pred >= (c0 + c1)/2 else c0
            
        mae += abs(y_pred_real - y_raw[i])
        
        if abs(pred_class - y_raw[i]) < 1e-5:
            correct += 1
            
    accuracy = (correct / len(X)) * 100
    mae = mae / len(X)
    return accuracy, mae

def main():

    c0 = 1.0
    c1 = 8.0
    
    X_raw = np.array([[1.0, 1.0], [1.0, 8.0], [8.0, 1.0], [8.0, 8.0]])
    y_raw = np.array([1.0, 8.0, 8.0, 1.0])
    
    X = norm(X_raw, c0, c1)
    y_norm = norm(y_raw, c0, c1) 
    
    seeds = [99, 42, 100, 2023, 777]
    max_epochs = 5000
    
    epochs_mse, epochs_bce = [], []
    mae_mse_list, mae_bce_list = [], []
    
    best_bce_weights = None
    best_mse_weights = None
    rep_errors_mse, rep_errors_bce = [], [] 
    
    print("=== ЗАПУСК ОБУЧЕНИЯ ДЛЯ 5 РАЗНЫХ SEED ===")
    for i, seed in enumerate(seeds):
        W1_a, T1_a, W2_a, T2_a, errs_a, ep_a = train_mlp_mse(X, y_raw, max_epochs=max_epochs, seed=seed)
        acc_a, mae_a = evaluate_network(X, y_raw, y_norm, c0, c1, W1_a, T1_a, W2_a, T2_a, is_bce=False)
        epochs_mse.append(ep_a)
        mae_mse_list.append(mae_a)
        
        W1_b, T1_b, W2_b, T2_b, errs_b, ep_b = train_mlp_bce(X, y_norm, max_epochs=max_epochs, seed=seed)
        acc_b, mae_b = evaluate_network(X, y_raw, y_norm, c0, c1, W1_b, T1_b, W2_b, T2_b, is_bce=True)
        epochs_bce.append(ep_b)
        mae_bce_list.append(mae_b)
        
        print(f"Seed {seed:4d} | MSE: эпох={ep_a}, Acc={acc_a:3.0f}%, MAE={mae_a:.2f} | BCE: эпох={ep_b}, Acc={acc_b:3.0f}%, MAE={mae_b:.2f}")
        
        if i == 0:
            best_mse_weights = (W1_a, T1_a, W2_a, T2_a)
            best_bce_weights = (W1_b, T1_b, W2_b, T2_b)
            rep_errors_mse = errs_a
            rep_errors_bce = errs_b


    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(rep_errors_mse, color='red', label='MSE (Конфиг А)')
    ax1.plot(rep_errors_bce, color='green', label='BCE (Конфиг Б)')
    ax1.set_title("Сходимость (Seed = 42)")
    ax1.set_xlabel("Эпохи")
    ax1.set_ylabel("Суммарная ошибка Es")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = plt.subplot(2, 3, 2)
    width = 0.35
    ind = np.arange(len(seeds))
    ax2.bar(ind - width/2, epochs_mse, width, label='MSE', color='red', alpha=0.7)
    ax2.bar(ind + width/2, epochs_bce, width, label='BCE', color='green', alpha=0.7)
    # Штриховка для тех, кто уперся в лимит (не сошелся)
    for i, ep in enumerate(epochs_mse):
        if ep >= max_epochs: ax2.bar(ind[i] - width/2, ep, width, color='red', hatch='//')
    ax2.set_title("Разброс числа эпох по запускам")
    ax2.set_xticks(ind)
    ax2.set_xticklabels([f"S={s}" for s in seeds])
    ax2.set_ylabel("Кол-во эпох")
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(["MSE (Конф. А)", "BCE (Конф. Б)"], [np.mean(mae_mse_list), np.mean(mae_bce_list)], color=['red', 'green'])
    ax3.set_title("Средняя абсолютная ошибка (MAE)")
    ax3.set_ylabel("Ошибка в исходной шкале")
    ax3.grid(True, axis='y', linestyle='--', alpha=0.5)

    ax4 = plt.subplot(2, 3, 5)
    xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_norm = norm(grid_points, c0, c1)
    
    Z_bce = []
    W1, T1, W2, T2 = best_bce_weights
    for pt in grid_norm:
        _, _, _, y_pred = mlp_forward(pt, W1, T1, W2, T2)
        Z_bce.append(y_pred)
    Z_bce = np.array(Z_bce).reshape(xx.shape)
    
    contour = ax4.contourf(xx, yy, Z_bce, cmap='coolwarm', alpha=0.8, levels=20)
    ax4.scatter(X_raw[:, 0], X_raw[:, 1], c=y_norm, cmap='coolwarm', edgecolors='k', s=100)
    ax4.set_title("Поверхность скрытого слоя (BCE)")
    ax4.set_xlabel("X1")
    ax4.set_ylabel("X2")
    fig.colorbar(contour, ax=ax4, label=r"Нормализованный выход $\hat{y} \in (0, 1)$")

    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ (Ввод чисел от -10 до 10, 'q' для выхода)")
    print("="*50)
    
    W1, T1, W2, T2 = best_bce_weights # Используем успешную конфигурацию Б
    
    while True:
        val = input(f"Введите X1 и X2 (через пробел) или 'q': ").strip()
        if val.lower() == 'q':
            break
        try:
            items = val.split()
            a, b = float(items[0]), float(items[1])
            
            x_raw = np.array([a, b])
            x_norm = norm(x_raw, c0, c1)
            
            _, _, _, y_pred_norm = mlp_forward(x_norm, W1, T1, W2, T2)
            y_pred_real = denorm(y_pred_norm, c0, c1)
            
            pred_class = c1 if y_pred_norm >= 0.5 else c0
            
            print("-" * 30)
            print(f"Выход (нормализованный 0..1): {y_pred_norm:.4f}")
            print(f"Выход (шкала [{c0}; {c1}]):      {y_pred_real:.4f}")
            print(f"Определен класс:              {int(pred_class)}")
            print("-" * 30)
            
        except (ValueError, IndexError):
            print("Ошибка ввода. Введите два числа через пробел (например: 1 8).")

if __name__ == "__main__":
    main()