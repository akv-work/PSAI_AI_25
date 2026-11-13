import numpy as np
import matplotlib.pyplot as plt

def sigmoid(s):
    s = np.clip(s, -500, 500)
    return 1 / (1 + np.exp(-s))

def sigmoid_der(s):
    y = sigmoid(s)
    return y * (1 - y)

def relu(s):
    return np.maximum(0, s)

def relu_der(s):
    return np.where(s > 0, 1.0, 0.0)

def norm(x, c0, c1):
    return (x - c0) / (c1 - c0)

def denorm(y, c0, c1):
    return c0 + y * (c1 - c0)

def mlp_forward(x, W1, T1, W2, T2, hidden_act='sigmoid'):
    S1 = np.dot(x, W1) - T1
    
    if hidden_act == 'relu':
        h = relu(S1)
    else:
        h = sigmoid(S1)
        
    S2 = np.dot(h, W2) - T2
    y = sigmoid(S2) 
    return S1, h, S2, y

def train_mlp(X, y_norm, hidden_act='sigmoid', alpha=0.1, Ee=0.001, max_epochs=10000, seed=42):
    np.random.seed(seed)
    W1 = np.random.uniform(-0.5, 0.5, (2, 2))
    T1 = np.random.uniform(-0.5, 0.5, 2)
    W2 = np.random.uniform(-0.5, 0.5, 2)
    T2 = np.random.uniform(-0.5, 0.5, 1)[0]
    
    errors = []
    epoch = 0
    
    while True:
        Es = 0.0
        for xi, yi_norm in zip(X, y_norm):
            S1, h, S2, y_pred = mlp_forward(xi, W1, T1, W2, T2, hidden_act)
            
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            
            loss = - (yi_norm * np.log(y_pred_clipped) + (1 - yi_norm) * np.log(1 - y_pred_clipped))
            Es += loss
            
            delta2 = (y_pred - yi_norm)
            
            if hidden_act == 'relu':
                delta1 = relu_der(S1) * (W2 * delta2)
            else:
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

def evaluate_network(X, y_raw, y_norm, c0, c1, W1, T1, W2, T2, hidden_act):
    correct = 0
    mae = 0.0
    
    for i, xi in enumerate(X):
        _, _, _, y_pred_norm = mlp_forward(xi, W1, T1, W2, T2, hidden_act)
        y_pred_real = denorm(y_pred_norm, c0, c1)
        pred_class = c1 if y_pred_norm >= 0.5 else c0
            
        mae += abs(y_pred_real - y_raw[i])
        if abs(pred_class - y_raw[i]) < 1e-5:
            correct += 1
            
    accuracy = (correct / len(X)) * 100
    mae = mae / len(X)
    return accuracy, mae

def main():
    c0, c1 = 1.0, 8.0 
    
    X_raw = np.array([[1.0, 1.0], [1.0, 8.0], [8.0, 1.0], [8.0, 8.0]])
    y_raw = np.array([1.0, 8.0, 8.0, 1.0])
    
    X = norm(X_raw, c0, c1)
    y_norm = norm(y_raw, c0, c1)
    
    seeds = [100, 42, 2023, 777, 99]
    max_epochs = 5000
    
    epochs_sig, epochs_relu = [], []
    mae_sig_list, mae_relu_list = [], []
    
    best_sig_weights = None
    best_relu_weights = None
    rep_errors_sig, rep_errors_relu = [], []
    
    print("=== СРАВНЕНИЕ АКТИВАЦИЙ НА 5 СИДАХ ===")
    for i, seed in enumerate(seeds):
        W1_s, T1_s, W2_s, T2_s, errs_s, ep_s = train_mlp(X, y_norm, 'sigmoid', max_epochs=max_epochs, seed=seed)
        acc_s, mae_s = evaluate_network(X, y_raw, y_norm, c0, c1, W1_s, T1_s, W2_s, T2_s, 'sigmoid')
        epochs_sig.append(ep_s)
        mae_sig_list.append(mae_s)
        
        W1_r, T1_r, W2_r, T2_r, errs_r, ep_r = train_mlp(X, y_norm, 'relu', max_epochs=max_epochs, seed=seed)
        acc_r, mae_r = evaluate_network(X, y_raw, y_norm, c0, c1, W1_r, T1_r, W2_r, T2_r, 'relu')
        epochs_relu.append(ep_r)
        mae_relu_list.append(mae_r)
        
        print(f"Seed {seed:4d} | SIGMOID: эпох={ep_s:4d}, Acc={acc_s:3.0f}%, MAE={mae_s:.2f} | RELU: эпох={ep_r:4d}, Acc={acc_r:3.0f}%, MAE={mae_r:.2f}")
        
        if i == 0: 
            best_sig_weights = (W1_s, T1_s, W2_s, T2_s)
            best_relu_weights = (W1_r, T1_r, W2_r, T2_r)
            rep_errors_sig = errs_s
            rep_errors_relu = errs_r

    fig = plt.figure(figsize=(16, 12))
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(rep_errors_sig, color='blue', label='Sigmoid (Конф. А)')
    ax1.plot(rep_errors_relu, color='orange', label='ReLU (Конф. B)')
    ax1.set_title(f"Сходимость (Seed = {seeds[0]})")
    ax1.set_xlabel("Эпохи")
    ax1.set_ylabel("Ошибка Es (BCE)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = plt.subplot(2, 2, 2)
    width = 0.35
    ind = np.arange(len(seeds))
    ax2.bar(ind - width/2, epochs_sig, width, label='Sigmoid', color='blue', alpha=0.7)
    ax2.bar(ind + width/2, epochs_relu, width, label='ReLU', color='orange', alpha=0.7)
    
    for i, ep in enumerate(epochs_sig):
        if ep >= max_epochs: ax2.bar(ind[i] - width/2, ep, width, color='blue', hatch='//')
    for i, ep in enumerate(epochs_relu):
        if ep >= max_epochs: ax2.bar(ind[i] + width/2, ep, width, color='orange', hatch='//')
        
    ax2.set_title("Разброс числа эпох по запускам")
    ax2.set_xticks(ind)
    ax2.set_xticklabels([f"S={s}" for s in seeds])
    ax2.set_ylabel("Кол-во эпох")
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_norm = norm(grid_points, c0, c1)
    
    ax3 = plt.subplot(2, 2, 3)
    Z_sig = [mlp_forward(pt, *best_sig_weights, 'sigmoid')[3] for pt in grid_norm]
    Z_sig = np.array(Z_sig).reshape(xx.shape)
    contour1 = ax3.contourf(xx, yy, Z_sig, cmap='coolwarm', alpha=0.8, levels=20)
    ax3.scatter(X_raw[:, 0], X_raw[:, 1], c=y_norm, cmap='coolwarm', edgecolors='k', s=100)
    ax3.set_title("Разделяющая поверхность (Скрытый = Sigmoid)")
    fig.colorbar(contour1, ax=ax3)

    ax4 = plt.subplot(2, 2, 4)
    Z_relu = [mlp_forward(pt, *best_relu_weights, 'relu')[3] for pt in grid_norm]
    Z_relu = np.array(Z_relu).reshape(xx.shape)
    contour2 = ax4.contourf(xx, yy, Z_relu, cmap='coolwarm', alpha=0.8, levels=20)
    ax4.scatter(X_raw[:, 0], X_raw[:, 1], c=y_norm, cmap='coolwarm', edgecolors='k', s=100)
    ax4.set_title("Разделяющая поверхность (Скрытый = ReLU)")
    fig.colorbar(contour2, ax=ax4)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ (Сеть ReLU). Ввод от -10 до 10, 'q' для выхода")
    print("="*50)
    
    while True:
        val = input(f"Введите X1 и X2 (через пробел) или 'q': ").strip()
        if val.lower() == 'q': break
        try:
            a, b = map(float, val.split())
            x_raw = np.array([a, b])
            x_norm = norm(x_raw, c0, c1)
            
            _, _, _, y_pred_norm = mlp_forward(x_norm, *best_relu_weights, 'relu')
            y_pred_real = denorm(y_pred_norm, c0, c1)
            pred_class = c1 if y_pred_norm >= 0.5 else c0
            
            print(f" Норм. выход: {y_pred_norm:.4f} | Исходная шкала: {y_pred_real:.4f} | Класс: {int(pred_class)}")
        except:
            print("Ошибка ввода. Пример: 1 8")

if __name__ == "__main__":
    main()