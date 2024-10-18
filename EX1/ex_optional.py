import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    mu = np.mean(X, axis=0)  # 计算每一列的均值
    sigma = np.std(X, axis=0, ddof=0)  # 计算每一列的标准差
    # 标准化
    X_norm = (X - mu) / sigma
    return X_norm,mu,sigma

# 定义成本函数  square error
def square_error(X, Y, theta):
    error = X @ theta - Y  # 计算预测值与实际值的误差
    J = (1 / (2 * len(Y))) * np.sum(error ** 2)  # 计算标量损失
    return J

# 定义梯度下降算法
def gradient_descent(X, Y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        error = X @ theta - Y  # 计算误差
        gradient = (1 / len(Y)) * (X.T @ error)  # 计算梯度
        theta -= alpha * gradient  # 更新 theta
        J_history[i] = square_error(X, Y, theta)  # 记录当前成本
        print(f"Iteration {i + 1}, Cost: {J_history[i]}")  # 输出当前迭代的成本
    print("Over")
    return theta, J_history

# 导入数据
data = np.loadtxt('ex1data2.txt', delimiter=',')
m = data.shape[0]
X = data[:, [0,1]]
Y = data[:, 2]

#标准化X
X,mu,sigma=normalize(X)
X_new = np.insert(X, 0, 1, axis=1)
theta = np.zeros(3)

#设置学习率与迭代次数
alpha = 0.1
num_iters = 50

#使用梯度下降算法计算出最终的theta与损失函数
theta, J_history = gradient_descent(X_new, Y, theta, alpha, num_iters)
# 绘制 J_history
plt.figure(figsize=(10, 6))
plt.plot(range(num_iters), J_history, label='Cost J', color='blue')
plt.title('Cost J vs. Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost J')
plt.grid(True)
plt.legend()
plt.show()


# 预测数据
x_predict = np.array([1, 1203, 3])  # 包含偏置项的输入特征
# 标准化
x_predict_norma = (x_predict[1:] - mu) / sigma  # 只对特征部分进行标准化
x_predict_norma = np.hstack([1, x_predict_norma])  # 添加偏置项
# 预测
predicted_Y = x_predict_norma @ theta
print(f'predicted_Y is {predicted_Y}')
# 精度分析
actual_Y = 239500  # 实际值
print(f'actual_Y is {actual_Y}')
det=100*(actual_Y-predicted_Y)/actual_Y
print(f'the error is {det}%')