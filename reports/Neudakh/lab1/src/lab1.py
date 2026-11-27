import numpy as np
import math
#  5 XOR 5   = 5   (Класс c0)
#  5 XOR -2  = -2  (Класс c1)
#  -2 XOR 5  = -2  (Класс c1)
#  -2 XOR -2 = 5   (Класс c0)
C0 = 5.0
C1 = -2.0

def scale_input(val):
    return (val + 10.0) / 20.0

def scale_target(val):
    return (val - C1) / (C0 - C1)

def inverse_scale_target(val_scaled):
    return val_scaled * (C0 - C1) + C1

def sigmoid(x):
    x = np.clip(x, -500, 500)  
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

X_raw = np.array([
    [5.0, 5.0],
    [5.0, -2.0],
    [-2.0, 5.0],
    [-2.0, -2.0]
])

Y_raw = np.array([5.0, -2.0, -2.0, 5.0])

X = np.array([[scale_input(a), scale_input(b)] for a, b in X_raw])
Y = np.array([scale_target(y) for y in Y_raw]).reshape(-1, 1)

print("Масштабированные входы (X):\n", np.round(X, 3))
print("Масштабированные выходы (Y):\n", np.round(Y, 3), "\n")

class SingleLayerPerceptron:
    def __init__(self, lr=0.5):
        self.lr = lr
        self.w = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b = np.random.uniform(-0.5, 0.5, (1,))

    def forward(self, x):
        return sigmoid(np.dot(x, self.w) + self.b)

    def train(self, X, Y, epochs=10000, target_error=0.01):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = X[i].reshape(1, -1)
                y_i = Y[i].reshape(1, 1)
                
                output = self.forward(x_i)
                error = y_i - output
                total_error += np.sum(error ** 2) / 2
                
             
                delta = error * sigmoid_deriv(output)
                self.w += self.lr * np.dot(x_i.T, delta)
                self.b += self.lr * delta.flatten()
                
            if total_error <= target_error:
                return epoch + 1, total_error
        return epochs, total_error

class MultiLayerPerceptron:
    def __init__(self, lr=2.0):
        self.lr = lr
        self.w1 = np.random.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, 2))
        self.w2 = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def forward(self, x):
        h = sigmoid(np.dot(x, self.w1) + self.b1)
        out = sigmoid(np.dot(h, self.w2) + self.b2)
        return h, out

    def train(self, X, Y, epochs=10000, target_error=0.01):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = X[i].reshape(1, -1)
                y_i = Y[i].reshape(1, 1)
                
                h, output = self.forward(x_i)
                
                error_out = y_i - output
                total_error += np.sum(error_out ** 2) / 2               
                delta_out = error_out * sigmoid_deriv(output)
                
                error_hidden = np.dot(delta_out, self.w2.T)
                delta_hidden = error_hidden * sigmoid_deriv(h)
                
                self.w2 += self.lr * np.dot(h.T, delta_out)
                self.b2 += self.lr * delta_out.flatten()
                
                self.w1 += self.lr * np.dot(x_i.T, delta_hidden)
                self.b1 += self.lr * delta_hidden.flatten()
                
            if total_error <= target_error:
                return epoch + 1, total_error
        return epochs, total_error

max_epochs = 10000
target_error = 0.01

print(" ОБУЧЕНИЕ ОДНОСЛОЙНОГО ПЕРСЕПТРОНА (SLP) ")
slp = SingleLayerPerceptron(lr=0.5)
slp_epochs, slp_error = slp.train(X, Y, epochs=max_epochs, target_error=target_error)

print(" ОБУЧЕНИЕ МНОГОСЛОЙНОГО ПЕРСЕПТРОНА (MLP 2-2-1) ")
mlp = MultiLayerPerceptron(lr=2.0)
mlp_epochs, mlp_error = mlp.train(X, Y, epochs=max_epochs, target_error=target_error)

def evaluate_accuracy(model, is_mlp=False):
    correct = 0
    for i in range(len(X)):
        x_i = X[i].reshape(1, -1)
        y_true = Y_raw[i]
        
        if is_mlp:
            _, out = model.forward(x_i)
        else:
            out = model.forward(x_i)
            
        y_pred_scaled = out[0][0]
        y_pred_real = inverse_scale_target(y_pred_scaled)
       
        pred_class = C0 if abs(y_pred_real - C0) < abs(y_pred_real - C1) else C1
        if pred_class == y_true:
            correct += 1
    return (correct / len(X)) * 100

slp_acc = evaluate_accuracy(slp, is_mlp=False)
mlp_acc = evaluate_accuracy(mlp, is_mlp=True)

print("\n РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ")
print(f"SLP: Эпох = {slp_epochs}, Итоговая ошибка Es = {slp_error:.5f}, Accuracy = {slp_acc}%")
print(f"MLP: Эпох = {mlp_epochs}, Итоговая ошибка Es = {mlp_error:.5f}, Accuracy = {mlp_acc}%")

print("\n РЕЖИМ ФУНКЦИОНИРОВАНИЯ ОБУЧЕННОЙ СЕТИ (MLP) ")
test_inputs = [
    (5, 5), (5, -2), (-2, 5), (-2, -2), 
    (0, 0), (3, -1), (-8, 7), (10, -10), (-5, -5)
]

print(f"{'Вход A':>6} | {'Вход B':>6} | {'Выход ŷ':>10} | {'Ближе к':>8}")
print("-" * 45)
for a, b in test_inputs:
    x_i = np.array([[scale_input(a), scale_input(b)]])
    _, out = mlp.forward(x_i)
    
    y_pred_scaled = out[0][0]
    y_pred_real = inverse_scale_target(y_pred_scaled)
    
    dist_c0 = abs(y_pred_real - C0)
    dist_c1 = abs(y_pred_real - C1)
    class_name = f"C0 ({C0})" if dist_c0 < dist_c1 else f"C1 ({C1})"
    
    print(f"{a:>6} | {b:>6} | {y_pred_real:>10.3f} | {class_name:>8}")

