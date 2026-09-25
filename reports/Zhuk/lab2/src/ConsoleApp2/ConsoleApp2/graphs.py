import pandas as pd
import matplotlib.pyplot as plt

# Ошибка обучения

data = pd.read_csv(r"bin\Debug\net10.0\history.csv")

plt.figure()

plt.plot(data["Epoch"], data["MSE"], label="MSE")
plt.plot(data["Epoch"], data["BCE"], label="BCE")

plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Изменение ошибки при обучении")
plt.legend()
plt.grid()

plt.show()

# Количество эпох

data = pd.read_csv(r"bin\Debug\net10.0\epochs.csv")

plt.figure()

plt.bar(data["Seed"] - 0.2, data["MSE"],
        width=0.4, label="MSE")

plt.bar(data["Seed"] + 0.2, data["BCE"],
        width=0.4, label="BCE")

plt.xlabel("Seed")
plt.ylabel("Количество эпох")
plt.title("Количество эпох до сходимости")
plt.xticks(data["Seed"])
plt.legend()
plt.grid(axis="y")

plt.show()

# MAE

data = pd.read_csv(r"bin\Debug\net10.0\mae.csv")

plt.figure()

plt.bar(data["Seed"] - 0.2, data["MSE"],
        width=0.4, label="MSE")

plt.bar(data["Seed"] + 0.2, data["BCE"],
        width=0.4, label="BCE")

plt.xlabel("Seed")
plt.ylabel("MAE")
plt.title("Средняя абсолютная ошибка")
plt.xticks(data["Seed"])
plt.legend()
plt.grid(axis="y")

plt.show()

# Accuracy

data = pd.read_csv(r"bin\Debug\net10.0\accuracy.csv")

plt.figure()

plt.bar(data["Seed"] - 0.2, data["MSE"],
        width=0.4, label="MSE")

plt.bar(data["Seed"] + 0.2, data["BCE"],
        width=0.4, label="BCE")

plt.xlabel("Seed")
plt.ylabel("Accuracy, %")
plt.title("Точность классификации")
plt.xticks(data["Seed"])
plt.legend()
plt.grid(axis="y")

plt.show()

# Разделяющая поверхность MSE

data = pd.read_csv(r"bin\Debug\net10.0\surface_mse.csv")

plt.figure()

for class_value in [1, 8]:
    class_data = data[data["Class"] == class_value]

    plt.scatter(
        class_data["A"],
        class_data["B"],
        s=5,
        label="c" + str(0 if class_value == 1 else 1)
    )

plt.scatter(
    [1, 1, 8, 8],
    [1, 8, 1, 8],
    s=80,
    marker="x",
    label="Обучающие точки"
)

plt.xlabel("A")
plt.ylabel("B")
plt.title("Разделяющая поверхность MSE")
plt.legend()
plt.grid()
plt.show()


# Разделяющая поверхность BCE

data = pd.read_csv(r"bin\Debug\net10.0\surface_bce.csv")

plt.figure()

for class_value in [1, 8]:
    class_data = data[data["Class"] == class_value]

    plt.scatter(
        class_data["A"],
        class_data["B"],
        s=5,
        label="c" + str(0 if class_value == 1 else 1)
    )

plt.scatter(
    [1, 1, 8, 8],
    [1, 8, 1, 8],
    s=80,
    marker="x",
    label="Обучающие точки"
)

plt.xlabel("A")
plt.ylabel("B")
plt.title("Разделяющая поверхность BCE")
plt.legend()
plt.grid()
plt.show()