import numpy as np
import matplotlib.pyplot as plt


def normalize(X):
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    normalized_X = 2 * (X - min_vals) / (max_vals - min_vals) - 1
    return normalized_X


# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def CostFunction(X, Y, theta):
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-15, 1 - 1e-15)  # 限制 h 的范围
    cost = -Y * np.log(h) - (1 - Y) * np.log(1 - h)
    J = np.mean(cost)  # 计算代价
    return J


def Gradient_Descent(X, Y, theta, alpha, num_iters):
    for i in range(num_iters):
        h = sigmoid(X @ theta)  # 计算预测值
        h = np.clip(h, 1e-15, 1 - 1e-15)  # 限制 h 的范围
        gradient = (1 / m) * (X.T @ (h - Y))  # 计算梯度
        theta -= alpha * gradient  # 更新参数
        J = CostFunction(X, Y, theta)  # 计算代价
        print(f"Iteration {i}: J is {J}, theta is {theta}, gradient is {gradient}")
    return theta  # 返回更新后的参数


def plotDecisionBoundary(theta, X, y):
    # 计算决策边界
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    x2_min, x2_max = X[:, 2].min(), X[:, 2].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))

    Z = sigmoid(np.c_[np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()] @ theta)
    Z = Z.reshape(xx1.shape)

    # 绘制数据点
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap='RdYlBu', marker='o', edgecolor='k')

    # 绘制决策边界
    plt.contour(xx1, xx2, Z, levels=[0.5], linewidths=1, colors='blue')

    plt.xlabel('Feature 1 (Normalized)')
    plt.ylabel('Feature 2 (Normalized)')
    plt.title('Decision Boundary with Predictions')
    plt.show()


# 导入数据
data = np.loadtxt("ex2data1.txt", delimiter=',')
X = data[:, [0, 1]]
Y = data[:, 2]

# 设计参数
theta = np.array([0, 0, 0], dtype=float)
X_norma = normalize(X)
X_new = np.insert(X_norma, 0, 1, axis=1)
m = data.shape[0]
alpha = 0.1
num_iters = 20000

# 梯度下降
theta_final = Gradient_Descent(X_new, Y, theta, alpha, num_iters)

# 绘制原始图像和预测分割线
plotDecisionBoundary(theta_final, X_new, Y)
