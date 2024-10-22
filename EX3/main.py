import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#可视化数据
def DateVisualization(X,y):
    # 随机选择10个样本的索引
    random_indices = np.random.choice(X.shape[0], 10, replace=False)

    # 可视化随机选择的样本
    plt.figure(figsize=(10, 5))

    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx].reshape(20, 20, order='F'), cmap='gray')  # 使用列优先顺序重塑
        plt.title(f'Label: {y[idx][0]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 代价函数
def CostFunction(X, Y, theta, lamda):
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-15, 1 - 1e-15)  # 限制 h 的范围
    cost = -Y * np.log(h) - (1 - Y) * np.log(1 - h)
    J = np.mean(cost) + lamda * np.sum(theta[1:]**2) / (2 * m)
    return J

#one_hot编码
def one_hot_encode(y, num_labels):
    m = y.shape[0]
    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] % num_labels] = 1  # 使用y[i] % num_labels确保标签在范围内
    return Y

# 梯度下降
def Gradient_Descent(X, Y, theta, alpha, lamda, num_iters, m, tolerance=1e-5):
    J_history = []  # 用于记录每次迭代的代价
    for i in range(num_iters):
        h = sigmoid(X @ theta)  # 计算预测值
        h = np.clip(h, 1e-15, 1 - 1e-15)  # 限制 h 的范围
        gradient = (1 / m) * (X.T @ (h - Y))  # 计算梯度
        gradient[1:] += (lamda / m) * theta[1:]  # 正则化
        theta -= alpha * gradient  # 更新参数
        J = CostFunction(X, Y, theta, lamda)  # 计算代价
        J_history.append(J)  # 记录代价
        print(f"Iteration {i}: J is {J:.3f}")

        # 检查收敛条件
        if i > 0 and abs(J_history[-1] - J_history[-2]) < tolerance:
            print(f"Converged at iteration {i}")
            break  # 如果收敛，停止迭代

    # 绘制代价随迭代次数变化的曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(J_history)), J_history, color='blue')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost J')
    plt.title('Cost J vs. Iterations')
    plt.grid(True)
    plt.show()

    return theta

#预测苹果
def evaluate_predictions(X, y, theta):
    X = np.insert(X, 0, 1, axis=1)  # 维度变为 (m, n+1)
    # 随机选择10个样本的索引
    random_indices = np.random.choice(X.shape[0], 1000, replace=False)
    X_random = X[random_indices]  # 选择随机样本
    predictions = sigmoid(X @ theta)  # 维度为 (m, num_labels)
    # 使用训练好的 theta 进行预测
    predicted_labels = np.argmax(predictions, axis=1)

    # 获取真实标签
    true_labels = y[random_indices].flatten()  # 展平为一维数组

    # 打印预测结果和真实结果
    correct_predictions = 0
    for i, idx in enumerate(random_indices):
        print(f"Sample {idx}: Predicted Label: {predicted_labels[idx]}, True Label: {true_labels[i]}")
        if predicted_labels[idx] == true_labels[i]:
            correct_predictions += 1

    # 计算准确度
    accuracy = correct_predictions / len(random_indices) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# 加载数据
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
y[y == 10] = 0
#参数设置
m = X.shape[0]
theta = np.zeros([401,10])
X_new = np.insert(X, 0, 1, axis=1)
Y_encoded = one_hot_encode(y,10)
alpha = 0.2
num_iters = 10000
lamda = 0.1
#计算最终的theta
final_theta = Gradient_Descent(X_new, Y_encoded, theta,alpha,lamda,num_iters,m)

#对模型进行评估
evaluate_predictions(X, y, final_theta)
