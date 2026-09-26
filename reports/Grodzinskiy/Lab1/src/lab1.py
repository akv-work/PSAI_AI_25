import numpy as np

c0 = 7
c1 = -7

data = np.array([
    [ 7,  7,  7],
    [ 7, -7, -7],
    [-7,  7, -7],
    [-7, -7,  7]
], dtype=float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def dsigmoid(y):
    return y * (1 - y)

def normalize(x):
    return (x + 10) / 20

def denormalize(y):
    return y * 20 - 10

def get_class(y):
    return c0 if abs(y - c0) < abs(y - c1) else c1

X = np.array([[normalize(v) for v in row[:2]] for row in data])
Y = np.array([[normalize(row[2])] for row in data])

np.random.seed(1)

W1 = np.random.randn(2, 2) * 0.1
W2 = np.random.randn(2, 1) * 0.1
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

lr = 0.5
maxepoch = 10000
errorlimit = 0.01

for epoch in range(maxepoch):
    error = 0
    for i in range(len(X)):
        x = X[i:i+1]
        y = Y[i:i+1]

        h = sigmoid(x @ W1 + b1)
        out = sigmoid(h @ W2 + b2)

        error += np.sum((y - out) ** 2)

        d_out = (y - out) * dsigmoid(out)
        d_h = d_out @ W2.T * dsigmoid(h)

        W2 += lr * h.T @ d_out
        b2 += lr * d_out
        W1 += lr * x.T @ d_h
        b1 += lr * d_h

    if error < errorlimit:
        break

def predict(a, b):
    x = np.array([[normalize(a), normalize(b)]])
    h = sigmoid(x @ W1 + b1)
    y = sigmoid(h @ W2 + b2)
    return denormalize(y[0][0])

correct = 0
print("Вариант 4 - c0=7, c1=-7\n")
print("MLP 2-2-1")
print("Эпохи:", epoch + 1)
print("Ошибка:", round(error, 6))
print("\nРезультаты:")

for row in data:
    result = predict(row[0], row[1])
    cls = get_class(result)
    print(f"({row[0]:2.0f},{row[1]:2.0f}) | "
          f"{result:7.3f} ({cls:2.0f}), ожидаем {row[2]:2.0f}")
    if cls == row[2]:
        correct += 1
print("Точность:", correct / 4)

W = np.random.randn(2, 1) * 0.1
b = np.zeros((1, 1))

for epoch2 in range(maxepoch):
    err = 0
    for i in range(len(X)):
        x = X[i:i+1]
        y = Y[i:i+1]
        out = sigmoid(x @ W + b)
        err += np.sum((y - out) ** 2)
        delta = (y - out) * dsigmoid(out)
        W += lr * x.T @ delta
        b += lr * delta
    if err < errorlimit:
        break

correct = 0
for row in data:
    x = np.array([[normalize(row[0]), normalize(row[1])]])
    out = sigmoid(x @ W + b)
    value = denormalize(out[0][0])
    if get_class(value) == row[2]:
        correct += 1

print("\nОднослойный персептрон")
print("Эпохи:", epoch2 + 1)
print("Ошибка:", round(err, 6))
print("\nРезультаты:")

for row in data:
    x = np.array([[normalize(row[0]), normalize(row[1])]])
    out = sigmoid(x @ W + b)
    value = denormalize(out[0][0])
    print(f"({row[0]:2.0f},{row[1]:2.0f}) | "
          f"{value:7.3f} ({get_class(value):2.0f}), ожидаем {row[2]:2.0f}")
print("Точность:", correct / 4)

print("\nРежим ввода")
while True:
    s = input("A B [-10 10]: ")
    if s.strip() == "":
        break
    a, b = map(float, s.split())
    answer = predict(a, b)
    print(f"y={answer:.3f}, класс={get_class(answer):g}")