import numpy as np
import matplotlib.pyplot as plt

# Параметры Варианта 9
A_VAL, B_VAL, C_VAL, D_VAL = 0.9, 0.33, 0.45, 0.75
VAR_P, VAR_N = 7, 4
TOTAL_POINTS = 200
SPLIT_TRAIN = 140       # k=0...139 - train, k=140...199 - test

STEP_LR = 0.1          # Шаг обучения из диапазона [0.01; 0.3]
EPOCH_LIMIT = 20000
ERR_TARGET = 1e-4

def generate_target_series():
    # Генерируем временной ряд по формуле Варианта 9
    time_steps = np.arange(TOTAL_POINTS)
    signal_values = A_VAL * np.cos(B_VAL * time_steps) + C_VAL * np.sin(D_VAL * time_steps)
    return time_steps, signal_values

def sigm_func(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def d_sigm_func(a_val):
    return a_val * (1 - a_val)

def scale_min_max(raw_data, low_bound, high_bound):
    return (raw_data - low_bound) / (high_bound - low_bound)

def unscale_min_max(norm_data, low_bound, high_bound):
    return norm_data * (high_bound - low_bound) + low_bound

def generate_window_dataset(norm_series, window_size, start_k, end_k):
    # Вход - window_size предыдущих значений, цель - следующее значение
    in_features, target_vals, step_indices = [], [], []
    for step_i in range(start_k, end_k + 1):
        in_features.append(norm_series[step_i - window_size:step_i])
        target_vals.append(norm_series[step_i])
        step_indices.append(step_i)
    return np.array(in_features), np.array(target_vals).reshape(-1, 1), np.array(step_indices)

def initialize_network_params(seed_value, in_dim, hidden_dim):
    rng_gen = np.random.RandomState(seed_value)
    weights_1 = rng_gen.randn(in_dim, hidden_dim) * 0.1
    weights_2 = rng_gen.randn(hidden_dim, 1) * 0.1
    biases_1 = np.zeros((1, hidden_dim))
    biases_2 = np.zeros((1, 1))
    return [weights_1, weights_2, biases_1, biases_2]

def forward_propagation(inputs, net_params):
    w1, w2, b1, b2 = net_params
    h_layer = sigm_func(inputs @ w1 + b1)
    out_layer = h_layer @ w2 + b2      # Выход линейный, f(s)=s
    return h_layer, out_layer

def train_network(X_tr, Y_tr, seed_val, p_size, n_size, max_epochs=EPOCH_LIMIT, lr_rate=STEP_LR, target_err=ERR_TARGET):
    model_params = initialize_network_params(seed_val, p_size, n_size)
    w1, w2, b1, b2 = model_params
    loss_history = []
    sample_count = len(X_tr)
    shuffler = np.random.RandomState(seed_val + 10000)   # Свой генератор для перемешивания

    for epoch_idx in range(max_epochs):
        permuted_indices = shuffler.permutation(sample_count)   # Перемешиваем порядок примеров каждую эпоху
        total_sq_err = 0.0

        for idx in permuted_indices:
            x_sample, y_sample = X_tr[idx:idx + 1], Y_tr[idx:idx + 1]
            h_out, model_pred = forward_propagation(x_sample, model_params)

            total_sq_err += np.sum((y_sample - model_pred) ** 2)

            # Градиент: линейный выход -> (y-out), сигм. скрытый -> умножаем на h(1-h)
            grad_out = (y_sample - model_pred)
            grad_hidden = grad_out @ w2.T * d_sigm_func(h_out)

            w2 += lr_rate * h_out.T @ grad_out
            b2 += lr_rate * grad_out
            w1 += lr_rate * x_sample.T @ grad_hidden
            b1 += lr_rate * grad_hidden

        mean_sq_err = total_sq_err / sample_count   # Средняя ошибка по выборке
        loss_history.append(mean_sq_err)

        if mean_sq_err <= target_err:
            break

    return model_params, loss_history

def predict_scaled_sample(window_sample, net_params):
    _, pred_out = forward_propagation(window_sample.reshape(1, -1), net_params)
    return pred_out[0, 0]

def compute_metrics(net_params, X_data, Y_data, min_bound, max_bound):
    # Считаем метрики в исходной шкале
    preds_scaled = np.array([predict_scaled_sample(X_data[i], net_params) for i in range(len(X_data))])
    preds_orig = unscale_min_max(preds_scaled, min_bound, max_bound)
    targets_orig = unscale_min_max(Y_data.ravel(), min_bound, max_bound)

    diff_errors = preds_orig - targets_orig
    mae_val = np.mean(np.abs(diff_errors))
    rmse_val = np.sqrt(np.mean(diff_errors ** 2))

    # Ряд колеблется около нуля, поэтому берём sMAPE вместо MAPE
    denom_vals = np.abs(targets_orig) + np.abs(preds_orig)
    denom_vals[denom_vals == 0] = 1e-9
    smape_val = np.mean(2 * np.abs(diff_errors) / denom_vals) * 100

    return mae_val, rmse_val, smape_val, preds_orig, targets_orig

def multi_step_recursive_forecast(net_params, full_scaled_series, window_size, start_step, forecast_steps):
    # Рекурсивный прогноз: каждое предсказание подставляется в окно как вход
    curr_window = full_scaled_series[start_step - window_size:start_step].copy()
    multi_preds_scaled = []
    for _ in range(forecast_steps):
        next_pred = predict_scaled_sample(curr_window, net_params)
        multi_preds_scaled.append(next_pred)
        curr_window = np.append(curr_window[1:], next_pred)
    return np.array(multi_preds_scaled)


def format_window_string(norm_window):
    # Ровный однострочный вывод окна без переносов numpy
    return "[" + ", ".join(f"{v:.4f}" for v in norm_window) + "]"


def output_sample_prediction(norm_window, net_params, min_bound, max_bound, ground_truth=None):
    # Печатает один пример в едином формате: вход, прогноз норм., прогноз в шкале ряда, ошибка
    scaled_p = predict_scaled_sample(norm_window, net_params)
    orig_p = unscale_min_max(scaled_p, min_bound, max_bound)

    print(f"    вход (норм.):     {format_window_string(norm_window)}")
    print(f"    прогноз (норм.):  {scaled_p:.4f}")
    print(f"    прогноз (ряд):    {orig_p:.4f}")
    if ground_truth is not None:
        print(f"    истинное x(k):    {ground_truth:.4f}")
        print(f"    абс. ошибка:      {abs(orig_p - ground_truth):.4f}")
    print()
    return orig_p

def interactive_mode(net_params, window_size, orig_series, norm_series, min_bound, max_bound, train_boundary):
    # Интерактивный режим: пользователь вводит либо k, либо window_size значений ряда
    print(f"\nРежим функционирования (сеть p={window_size})")
    print(f"Введите номер k ({window_size}..{TOTAL_POINTS-1}) чтобы взять окно из ряда,")
    print(f"или {window_size} чисел через пробел - своё окно в исходной шкале.")
    print("Для выхода: letmeleavepls")

    while True:
        user_input = input("> ").strip()

        if user_input == "letmeleavepls":
            break

        input_tokens = user_input.split()

        if len(input_tokens) == 1:
            try:
                k_idx = int(input_tokens[0])
            except ValueError:
                print("Ошибка: ожидался номер k"); continue

            if k_idx < window_size or k_idx >= TOTAL_POINTS:
                print(f"k должен быть в диапазоне [{window_size}; {TOTAL_POINTS-1}]"); continue

            window_norm = norm_series[k_idx - window_size:k_idx]
            dataset_region = "train" if k_idx < train_boundary else "test"
            print(f"  k={k_idx} ({dataset_region})")
            output_sample_prediction(window_norm, net_params, min_bound, max_bound, ground_truth=orig_series[k_idx])

        elif len(input_tokens) == window_size:
            try:
                raw_inputs = np.array([float(val) for val in input_tokens])
            except ValueError:
                print("Ошибка: нужны числа"); continue

            window_norm = scale_min_max(raw_inputs, min_bound, max_bound)
            print(f"  своё окно: {raw_inputs}")
            output_sample_prediction(window_norm, net_params, min_bound, max_bound, ground_truth=None)

        else:
            print(f"Нужно либо одно число (k), либо {window_size} чисел (окно)")

if __name__ == "__main__":
    k_steps, x_series = generate_target_series()
    min_x_tr, max_x_tr = x_series[:SPLIT_TRAIN].min(), x_series[:SPLIT_TRAIN].max()
    x_scaled_series = scale_min_max(x_series, min_x_tr, max_x_tr)

    print(f"Train: k=0..{SPLIT_TRAIN-1} ({SPLIT_TRAIN} точек); Test: k={SPLIT_TRAIN}..{TOTAL_POINTS-1} ({TOTAL_POINTS-SPLIT_TRAIN} точек)")
    print(f"x_min(train)={min_x_tr:.4f}, x_max(train)={max_x_tr:.4f}")
    print(f"x_norm(test) диапазон: [{x_scaled_series[SPLIT_TRAIN:].min():.4f}; {x_scaled_series[SPLIT_TRAIN:].max():.4f}]  "
          f"(выход за [0;1] ожидаем, т.к. min/max считаны только по train)")

    plt.figure(figsize=(12, 4))
    plt.plot(k_steps, x_series)
    plt.axvline(SPLIT_TRAIN - 0.5, linestyle="--", color="black", label="граница train/test")
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.title("Ряд x(k), вариант 9")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot0_series.png", dpi=110)
    plt.show()

    X_train, Y_train, k_train = generate_window_dataset(x_scaled_series, VAR_P, VAR_P, SPLIT_TRAIN - 1)
    X_test, Y_test, k_test = generate_window_dataset(x_scaled_series, VAR_P, SPLIT_TRAIN, TOTAL_POINTS - 1)
    print(f"Обучающих примеров: {len(X_train)}; тестовых примеров: {len(X_test)}")

    # Базовый эксперимент
    base_model_params, base_loss_history = train_network(X_train, Y_train, seed_val=1, p_size=VAR_P, n_size=VAR_N)
    mae_tr, rmse_tr, smape_tr, *_ = compute_metrics(base_model_params, X_train, Y_train, min_x_tr, max_x_tr)
    mae_te, rmse_te, smape_te, preds_te_base, true_te_base = compute_metrics(base_model_params, X_test, Y_test, min_x_tr, max_x_tr)
    print(f"\nБазовый эксперимент p={VAR_P}, N={VAR_N}, seed=1: "
          f"эпох={len(base_loss_history)}, Es={base_loss_history[-1]:.8f}")
    print(f"Train: MAE={mae_tr:.4f} RMSE={rmse_tr:.4f} sMAPE={smape_tr:.2f}%")
    print(f"Test:  MAE={mae_te:.4f} RMSE={rmse_te:.4f} sMAPE={smape_te:.2f}%")

    # Сюда складываем все обученные модели, чтобы потом честно выбрать лучшую по test MAE
    all_candidate_models = [(VAR_P, VAR_N, 1, base_model_params, base_loss_history, mae_te)]

    # Исследование p (N фиксировано)
    print("\nИсследование p:")
    p_histories_to_plot = {}
    for p_val in [2, 4, VAR_P, 2 * VAR_P]:
        X_tr_p, Y_tr_p, _ = generate_window_dataset(x_scaled_series, p_val, p_val, SPLIT_TRAIN - 1)
        X_te_p, Y_te_p, _ = generate_window_dataset(x_scaled_series, p_val, SPLIT_TRAIN, TOTAL_POINTS - 1)
        epochs_records, mae_records, unstopped_count = [], [], 0
        for seed_idx in range(1, 6):
            m_params, h_hist = train_network(X_tr_p, Y_tr_p, seed_val=seed_idx, p_size=p_val, n_size=VAR_N)
            m_mae, *_ = compute_metrics(m_params, X_te_p, Y_te_p, min_x_tr, max_x_tr)
            epochs_records.append(len(h_hist)); mae_records.append(m_mae)
            if h_hist[-1] > ERR_TARGET:
                unstopped_count += 1
            if p_val != 2:   # p=2 не сходится вообще, в кандидаты на "лучшую" не берём
                all_candidate_models.append((p_val, VAR_N, seed_idx, m_params, h_hist, m_mae))
            if seed_idx == 1 and p_val in (4, 14):
                p_histories_to_plot[f"p={p_val}, N={VAR_N}"] = h_hist
            print(f"    p={p_val}, seed={seed_idx}: эпох={len(h_hist)}, "
                  f"{'сошлось' if h_hist[-1] <= ERR_TARGET else 'НЕ сошлось'}", flush=True)
        print(f"  p={p_val:2d}: эпох {np.mean(epochs_records):.0f}±{np.std(epochs_records):.0f} | "
              f"test MAE {np.mean(mae_records):.4f}±{np.std(mae_records):.4f} | не сошлось {unstopped_count}/5")

    # Исследование N (p фиксировано)
    print("\nИсследование N:")
    for n_val in [1, 2, VAR_N, 2 * VAR_N]:
        epochs_records, mae_train_records, mae_test_records, unstopped_count = [], [], [], 0
        for seed_idx in range(1, 6):
            m_params, h_hist = train_network(X_train, Y_train, seed_val=seed_idx, p_size=VAR_P, n_size=n_val)
            m_train_mae, *_ = compute_metrics(m_params, X_train, Y_train, min_x_tr, max_x_tr)
            m_test_mae, *_ = compute_metrics(m_params, X_test, Y_test, min_x_tr, max_x_tr)
            epochs_records.append(len(h_hist)); mae_train_records.append(m_train_mae); mae_test_records.append(m_test_mae)
            if h_hist[-1] > ERR_TARGET:
                unstopped_count += 1
            all_candidate_models.append((VAR_P, n_val, seed_idx, m_params, h_hist, m_test_mae))
            print(f"    N={n_val}, seed={seed_idx}: эпох={len(h_hist)}, "
                  f"{'сошлось' if h_hist[-1] <= ERR_TARGET else 'НЕ сошлось'}", flush=True)
        print(f"  N={n_val}: эпох {np.mean(epochs_records):.0f}±{np.std(epochs_records):.0f} | "
              f"train MAE {np.mean(mae_train_records):.4f} | test MAE {np.mean(mae_test_records):.4f}±{np.std(mae_test_records):.4f} | "
              f"не сошлось {unstopped_count}/5")

    # Выбираем лучшую по тестовому MAE конфигурацию из всех прогнанных
    all_candidate_models.sort(key=lambda item: item[5])
    best_p, best_N, best_seed, best_model_params, best_loss_history, best_mae = all_candidate_models[0]
    print(f"\nЛучшая конфигурация по тестовому MAE: p={best_p}, N={best_N}, seed={best_seed}, "
          f"test MAE={best_mae:.4f}, эпох={len(best_loss_history)}")

    X_train_best, Y_train_best, k_train_best = generate_window_dataset(x_scaled_series, best_p, best_p, SPLIT_TRAIN - 1)
    X_test_best, Y_test_best, k_test_best = generate_window_dataset(x_scaled_series, best_p, SPLIT_TRAIN, TOTAL_POINTS - 1)
    mae_trb, rmse_trb, smape_trb, preds_trb, true_trb = compute_metrics(best_model_params, X_train_best, Y_train_best, min_x_tr, max_x_tr)
    mae_teb, rmse_teb, smape_teb, preds_teb, true_teb = compute_metrics(best_model_params, X_test_best, Y_test_best, min_x_tr, max_x_tr)
    print(f"Train (лучшая): MAE={mae_trb:.4f} RMSE={rmse_trb:.4f} sMAPE={smape_trb:.2f}%")
    print(f"Test  (лучшая): MAE={mae_teb:.4f} RMSE={rmse_teb:.4f} sMAPE={smape_teb:.2f}%")

    # Рекурсивный прогноз на 30 шагов для лучшей конфигурации
    rec_preds_scaled = multi_step_recursive_forecast(best_model_params, x_scaled_series, best_p, SPLIT_TRAIN, 30)
    rec_preds_orig = unscale_min_max(rec_preds_scaled, min_x_tr, max_x_tr)
    true_rec_vals = x_series[SPLIT_TRAIN:SPLIT_TRAIN + 30]
    rec_abs_errors = np.abs(rec_preds_orig - true_rec_vals)
    train_data_span = x_series[:SPLIT_TRAIN].max() - x_series[:SPLIT_TRAIN].min()
    error_threshold = 0.05 * train_data_span
    exceed_indices = np.where(rec_abs_errors > error_threshold)[0]
    first_exceed_step = exceed_indices[0] + 1 if len(exceed_indices) > 0 else None
    print(f"\nРекурсивный прогноз (p={best_p}, N={best_N}): размах train={train_data_span:.4f}, "
          f"порог 5%={error_threshold:.4f}")
    print(f"Первый шаг превышения порога: {first_exceed_step if first_exceed_step else 'не превышен за 30 шагов'}")

    # Графики
    plt.figure(figsize=(12, 5))
    plt.plot(k_steps, x_series, label="Исходный ряд", color="gray", alpha=0.6)
    plt.plot(k_train_best, preds_trb, label="Прогноз (train)", color="blue")
    plt.plot(k_test_best, preds_teb, label="Прогноз (test)", color="red")
    plt.axvline(SPLIT_TRAIN - 0.5, linestyle="--", color="black", label="граница train/test")
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.title(f"Ряд и одношаговый прогноз (лучшая конфигурация p={best_p}, N={best_N})")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot1_series_forecast.png", dpi=110)
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(base_loss_history, label=f"Базовая (p={VAR_P},N={VAR_N})")
    for plot_label, hist_vals in p_histories_to_plot.items():
        plt.plot(hist_vals, label=plot_label)
    plt.axhline(ERR_TARGET, linestyle="--", color="black", label=f"Ee={ERR_TARGET}")
    plt.yscale("log")
    plt.xlabel("Эпоха"); plt.ylabel("Es (лог. шкала)")
    plt.title("Сходимость: базовая + 2 конфигурации из исследования p")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot2_convergence.png", dpi=110)
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_teb, preds_teb, alpha=0.7)
    scatter_limits = [min(true_teb.min(), preds_teb.min()), max(true_teb.max(), preds_teb.max())]
    plt.plot(scatter_limits, scatter_limits, "k--", label="y = x")
    plt.xlabel("Эталон"); plt.ylabel("Прогноз")
    plt.title("Эталон vs прогноз (тест, лучшая конфигурация)")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("plot3_scatter.png", dpi=110)
    plt.show()

    plt.figure(figsize=(12, 5))
    step_numbers = np.arange(1, 31)
    plt.subplot(1, 2, 1)
    plt.plot(k_steps[SPLIT_TRAIN:SPLIT_TRAIN + 30], true_rec_vals, label="Эталон", marker="o", ms=3)
    plt.plot(k_steps[SPLIT_TRAIN:SPLIT_TRAIN + 30], rec_preds_orig, label="Рекурсивный прогноз", marker="x", ms=3)
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.title(f"Рекурсивный прогноз, 30 шагов (p={best_p}, N={best_N})")
    plt.legend(); plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(step_numbers, rec_abs_errors, marker="o", ms=3)
    plt.axhline(error_threshold, linestyle="--", color="red", label="5% размаха")
    plt.xlabel("Номер шага"); plt.ylabel("|ошибка|")
    plt.title("Рост ошибки рекурсивного прогноза")
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig("plot4_recursive.png", dpi=110)
    plt.show()

    # Демонстрация режима функционирования (не менее 3 train + 3 test + 1-2 своих окон)
    print(f"\nДемонстрация режима функционирования (лучшая модель p={best_p}, N={best_N}):")

    train_demo_indices = [20, 70, 130]
    test_demo_indices = [150, 170, 195]

    for k_demo in train_demo_indices + test_demo_indices:
        dataset_region = "train" if k_demo < SPLIT_TRAIN else "test"
        print(f"  k={k_demo} ({dataset_region})")
        output_sample_prediction(x_scaled_series[k_demo - best_p:k_demo], best_model_params, min_x_tr, max_x_tr, ground_truth=x_series[k_demo])

    custom_outlier_window = x_series[50:50 + best_p].copy()
    custom_outlier_window[best_p // 2] += 3.0
    print("  своё окно 1 (с выбросом)")
    output_sample_prediction(scale_min_max(custom_outlier_window, min_x_tr, max_x_tr), best_model_params, min_x_tr, max_x_tr)

    custom_flat_window = np.full(best_p, 0.3)
    print("  своё окно 2 (ровное, norm=0.3)")
    output_sample_prediction(custom_flat_window, best_model_params, min_x_tr, max_x_tr)

    print("Все графики сохранены (plot0..plot4_*.png) в рабочей директории.\n")

    # Интерактивный режим работы сети
    interactive_mode(best_model_params, best_p, x_series, x_scaled_series, min_x_tr, max_x_tr, SPLIT_TRAIN)