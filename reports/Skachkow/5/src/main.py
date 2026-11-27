import numpy as np
import matplotlib.pyplot as plt


def load_dataset(path):
    objects = []
    with open(path) as f:
        lines = [line.rstrip('\n') for line in f]
    i = 0
    while i < len(lines):
        if lines[i].strip() and lines[i][0].isdigit():
            parts = lines[i].split()
            cls = int(parts[0])
            inst = int(parts[1])
            grid = []
            for _ in range(7):
                i += 1
                grid.append([1 if c == '#' else 0 for c in lines[i].strip()])
            x = np.array(grid).flatten().astype(float)
            y = np.zeros(3)
            y[cls - 1] = 1.0
            objects.append({
                'cls': cls,
                'inst': inst,
                'x': x,
                'y': y,
                'grid': np.array(grid)
            })
        i += 1
    return objects


def dataset_quality(data):
    Nfill = int(sum(d['x'].sum() for d in data))
    intra = []
    inter = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            d = np.sum(data[i]['x'] != data[j]['x'])
            if data[i]['cls'] == data[j]['cls']:
                intra.append(d)
            else:
                inter.append(d)
    return Nfill, np.mean(intra), np.mean(inter)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

class MLP:
    def __init__(self, H, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.H = H
        self.W1 = np.random.randn(H, 35) * 0.1
        self.b1 = np.zeros(H)
        self.W2 = np.random.randn(3, H) * 0.1
        self.b2 = np.zeros(3)

    def forward(self, x):
        self.h_pre = self.W1 @ x + self.b1
        self.h = sigmoid(self.h_pre)
        self.o_pre = self.W2 @ self.h + self.b2
        self.p = softmax(self.o_pre)
        return self.p

    def predict(self, x):
        p = self.forward(x)
        return np.argmax(p), p

    def backward(self, x, y, eta):
        p = self.forward(x)
        delta_o = p - y
        delta_h = (self.W2.T @ delta_o) * self.h * (1 - self.h)
        dW2 = np.outer(delta_o, self.h)
        db2 = delta_o
        dW1 = np.outer(delta_h, x)
        db1 = delta_h
        self.W2 -= eta * dW2
        self.b2 -= eta * db2
        self.W1 -= eta * dW1
        self.b1 -= eta * db1
        ce = -np.sum(y * np.log(p + 1e-15))
        return ce

    def train(self, train_data, eta=0.05, max_epochs=5000, Ee=0.05, seed=None):
        if seed is not None:
            np.random.seed(seed)
        history = []
        for epoch in range(max_epochs):
            np.random.shuffle(train_data)
            total_ce = 0.0
            for sample in train_data:
                ce = self.backward(sample['x'], sample['y'], eta)
                total_ce += ce
            Es = total_ce / len(train_data)
            history.append(Es)
            if Es <= Ee:
                return epoch + 1, Es, history
        return max_epochs, Es, history


def confusion_matrix(net, data):
    cm = np.zeros((3, 3), dtype=int)
    for d in data:
        pred, _ = net.predict(d['x'])
        true = d['cls'] - 1
        cm[true, pred] += 1
    return cm

def accuracy(net, data):
    correct = 0
    conf_correct = []
    conf_wrong = []
    for d in data:
        pred, p = net.predict(d['x'])
        true = d['cls'] - 1
        conf = p[pred]
        if pred == true:
            correct += 1
            conf_correct.append(conf)
        else:
            conf_wrong.append(conf)
    acc = correct / len(data)
    return (acc,
            np.mean(conf_correct) if conf_correct else 0,
            np.mean(conf_wrong) if conf_wrong else 0)


def add_noise(x, m, seed):
    x_noisy = x.copy()
    if m > 0:
        np.random.seed(seed)
        idxs = np.random.choice(35, m, replace=False)
        x_noisy[idxs] = 1 - x_noisy[idxs]
    return x_noisy


if __name__ == "__main__":
    data = load_dataset("dataset.txt")
    train = [d for d in data if d['inst'] <= 7]
    test = [d for d in data if d['inst'] >= 8]

    Nfill, intra, inter = dataset_quality(data)
    print(f"Nfill = {Nfill}")
    print(f"Intra Hamming: {intra:.2f}")
    print(f"Inter Hamming: {inter:.2f}")

    # Базовый эксперимент H=3, η=0.05
    net = MLP(H=3, seed=42)
    epochs, Es, hist = net.train(train, eta=0.05, seed=42)
    acc_tr, _, _ = accuracy(net, train)
    acc_te, conf_c, conf_w = accuracy(net, test)
    cm_te = confusion_matrix(net, test)

    print(f"Эпох: {epochs}, Es: {Es:.4f}")
    print(f"Train acc: {acc_tr:.3f}, Test acc: {acc_te:.3f}")
    print("CM test:\n", cm_te)
    print(f"Уверенность верные: {conf_c:.3f}, неверные: {conf_w:.3f}")

    # Исследование H
    for H in [2, 3, 6]:
        epochs_list = []
        acc_list = []
        for run in range(5):
            seed = 100 + H * 10 + run
            net_h = MLP(H=H, seed=seed)
            ep, _, _ = net_h.train(train, eta=0.05, seed=seed)
            acc, _, _ = accuracy(net_h, test)
            epochs_list.append(ep)
            acc_list.append(acc)
        print(f"H={H}: epochs mean {np.mean(epochs_list):.1f} "
              f"[{min(epochs_list)}-{max(epochs_list)}], "
              f"acc mean {np.mean(acc_list):.3f}")