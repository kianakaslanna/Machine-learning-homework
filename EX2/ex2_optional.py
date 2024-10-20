import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

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

def Gradient_Descent(X, Y, theta, alpha, lamda, num_iters):
    for i in range(num_iters):
        h = sigmoid(X @ theta)  # 计算预测值
        h = np.clip(h, 1e-15, 1 - 1e-15)  # 限制 h 的范围
        gradient = (1 / m) * (X.T @ (h - Y))  # 计算未正则化的梯度
        gradient[1:] += (lamda / m) * theta[1:]  # 正则化
        theta -= alpha * gradient  # 更新参数
        J = CostFunction(X, Y, theta, lamda)  # 计算代价
        print(f"Iteration {i}: J is {J}")
    return theta  # 返回更新后的参数

# 导入数据
data = np.loadtxt("ex2data2.txt", delimiter=',')
X = data[:, [0, 1]]
Y = data[:, 2]

# 设计参数
poly = PolynomialFeatures(degree=6)  # 选择适当的多项式阶数
X_poly = poly.fit_transform(X)
m = data.shape[0]
theta = np.zeros(X_poly.shape[1], dtype=float)  # theta的维度应与X_poly的列数相匹配

# 设置超参数
alpha = 0.1
lamda = 0.01
num_iters = 20000

# 梯度下降
theta_final = Gradient_Descent(X_poly, Y, theta, alpha, lamda, num_iters)

# 绘制原始图像
plt.figure(figsize=(10, 6))
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], marker='o', label='Class 1', color='blue')
plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], marker='x', label='Class 0', color='red')

# 绘制预测分割线
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))

# 计算决策边界
Z = sigmoid(poly.transform(np.c_[xx1.ravel(), xx2.ravel()]) @ theta_final)
Z = Z.reshape(xx1.shape)

# 绘制决策边界
plt.contour(xx1, xx2, Z, levels=[0.5], linewidths=1, colors='blue')

# 设置坐标轴范围
plt.xlim(-1, 1.5)
plt.ylim(-0.8, 1.2)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot with Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()
