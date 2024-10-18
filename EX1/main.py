import numpy as np
import matplotlib.pyplot as plt
import ml_functions

# 导入数据
data = np.loadtxt('ex1data1.txt', delimiter=',')
m = data.shape[0]
X = data[:, 0]
Y = data[:, 1]
# 在 X 前添加一列全为 1
X_new = np.insert(X.reshape(-1, 1), 0, 1, axis=1)
# 初始化 theta
theta = np.zeros(2)
#设置学习率与迭代次数
alpha = 0.01
num_iters = 1500
#使用梯度下降算法计算出最终的theta与损失函数
theta, J_history = ml_functions.gradient_descent(X_new, Y, theta, alpha, num_iters)
#绘制theta与J的梯度图像
ml_functions.J_map(X_new,Y,theta,J_history)

# 绘制回归直线与原始数据的图像
predicted_Y = X_new @ theta
plt.scatter(X, Y, label="training data")
plt.xlabel("population")
plt.ylabel("profit")
plt.plot(X, predicted_Y, color='red', label='Regression Line')
plt.legend()
plt.show()
#对数据进行预测
predicted_Y = [1,3.5] @ theta
print(predicted_Y)
