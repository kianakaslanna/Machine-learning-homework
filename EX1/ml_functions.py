import numpy as np
import matplotlib.pyplot as plt

# 定义成本函数  square error
def square_error(X, Y, theta):
    error = X @ theta - Y  # 计算预测值与实际值的误差
    J = (1 / (2 * len(Y))) * np.sum(error ** 2)  # 计算损失函数
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

def J_map(X, Y,theta,J_history):
    # 生成示例数据，替换为你的实际数据
    theta0 = np.linspace(-10, 10, num=100)  # theta0 的范围
    theta1 = np.linspace(-1, 4, num=100)  # theta1 的范围

    #   创建一个网格
    theta0_grid, theta1_grid = np.meshgrid(theta0, theta1)

    # 计算每对 (theta0, theta1) 对应的损失函数值 J
    J_values = np.zeros(theta0_grid.shape)

    for i in range(theta0_grid.shape[0]):
        for j in range(theta0_grid.shape[1]):
         J_values[i, j] = square_error(X, Y, np.array([theta0_grid[i, j], theta1_grid[i, j]]))

    # 绘制三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0_grid, theta1_grid, J_values, cmap='viridis', alpha=0.7)
    ax.scatter(theta[0], theta[1], J_history[-1], color='r', s=50)  # 绘制最终参数位置
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Cost J')
    ax.set_title('3D Surface Plot of Cost Function')
    plt.show()