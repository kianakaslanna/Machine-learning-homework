import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# 可视化数据
def visualize_data(X, y_encoded):
    random_indices = np.random.choice(X.shape[0], 10, replace=False)
    plt.figure(figsize=(10, 5))

    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx].reshape(20, 20, order='F'), cmap='gray')

        # 显示编码后的标签
        plt.title(f'Label: {np.argmax(y_encoded[idx]) + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sigmoid 导数
def sigmoid_gradient(z):
    s = sigmoid(z)
    return s * (1 - s)

# One-hot 编码
def one_hot_encode(y, num_labels):
    m = y.shape[0]
    Y = np.zeros((m, num_labels))

    for i in range(m):
        label = y[i][0]  # 获取实际标签
        label = 9 if label == 10 else label - 1  # 将10转换为9，其他标签减1
        Y[i, label] = 1

    return Y

# 打印预测结果与真实标签进行对比
def print_comparison(a3, y, num_samples=100):
    random_indices = np.random.choice(a3.shape[0], num_samples, replace=False)
    correct_predictions = 0  # 记录正确预测数量

    for i in random_indices:
        predicted_class = np.argmax(a3[i]) + 1  # 预测的类别
        predicted_class = 0 if predicted_class == 10 else predicted_class

        actual_class = y[i][0]  # 真实标签
        actual_class = 0 if actual_class == 10 else actual_class

        if predicted_class == actual_class:
            correct_predictions += 1

        print(f"样本 {i + 1}: 预测值 -> {np.round(a3[i], 2)}  |  预测类别: {predicted_class}  |  真实类别: {actual_class}")

    accuracy = (correct_predictions / num_samples) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

# 前向传播
def forward(X_bias, theta1, theta2):
    a1 = X_bias  # 输入层
    z2 = a1 @ theta1.T
    a2 = np.insert(sigmoid(z2), 0, 1, axis=1)  # 隐藏层
    z3 = a2 @ theta2.T
    h = sigmoid(z3)  # 输出层
    return z2, a2, z3, h

# 代价函数
def compute_cost(X, Y, theta1, theta2, lamda):
    m = X.shape[0]
    h = forward(X, theta1, theta2)[3]
    cost = -Y * np.log(h) - (1 - Y) * np.log(1 - h)
    cost_sum = np.sum(cost)
    regularization = (lamda / (2 * m)) * (np.sum(theta1 ** 2) + np.sum(theta2 ** 2))
    J = (1 / m) * cost_sum + regularization
    return J

# 反向传播
def backward(X, Y, theta1, theta2, lamda, m):
    delta_3 = np.zeros(theta2.shape)  # 输出层的权重
    delta_2 = np.zeros(theta1.shape)  # 隐藏层的权重

    z2, a2, z3, a3 = forward(X, theta1, theta2)  # 前向传播
    error3 = a3 - Y  # 输出层误差
    error2 = error3.dot(theta2)[:, 1:] * sigmoid_gradient(z2)  # 隐藏层误差

    delta_2 += error2.T.dot(X)  # 更新隐藏层的delta
    delta_3 += error3.T.dot(a2)  # 更新输出层的delta

    D3 = (1 / m) * delta_3 + (lamda / m) * np.concatenate([np.zeros((theta2.shape[0], 1)), theta2[:, 1:]], axis=1)
    D2 = (1 / m) * delta_2 + (lamda / m) * np.concatenate([np.zeros((theta1.shape[0], 1)), theta1[:, 1:]], axis=1)

    return D2, D3  # 返回更新后的梯度

# 梯度下降
def gradient_descent(X, Y, theta1, theta2, alpha, lamda, num_iters, m, tolerance=1e-5):
    J_history = []  # 用于记录每次迭代的代价
    for i in range(num_iters):
        D2, D3 = backward(X, Y, theta1, theta2, lamda, m)
        theta1 -= alpha * D2
        theta2 -= alpha * D3
        J = compute_cost(X, Y, theta1, theta2, lamda)  # 计算代价
        J_history.append(J)  # 记录代价
        print(f"Iteration {i}: J is {J:.3f}")

        if i > 0 and abs(J_history[-1] - J_history[-2]) < tolerance:  # 检查收敛条件
            print(f"Converged at iteration {i}")
            break

    # 绘制代价变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(J_history)), J_history, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('J')
    plt.title('J vs. Iterations')
    plt.grid(True)
    plt.show()


# 加载数据
data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']
X_bias = np.insert(X, 0, 1, axis=1)  # 添加偏置项

# 数据划分：70%训练集，30%测试集
X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.3, random_state=42)
y_encoded = one_hot_encode(y_train, 10)

# 参数设置
input_layer_size = 400  # 输入层特征数
hidden_layer_size = 25  # 隐藏层节点数
num_labels = 10  # 输出层标签数
alpha = 0.5
lamda = 0.1
num_iters = 10000
m = X_train.shape[0]  # 使用训练集的样本数量

# 随机初始化权重参数
epsilon_init = 0.12
theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1) * 2 * epsilon_init - epsilon_init
theta2 = np.random.rand(num_labels, hidden_layer_size + 1) * 2 * epsilon_init - epsilon_init

# 梯度下降
gradient_descent(X_train, y_encoded, theta1, theta2, alpha, lamda, num_iters, m, tolerance=1e-5)

# 预测
a3 = forward(X_test, theta1, theta2)[3]
print_comparison(a3, y_test, num_samples=1000)
