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

#one_hot编码
def one_hot_encode(y, num_labels):
    m = y.shape[0]
    Y = np.zeros((m, num_labels))
    for i in range(m):
        Y[i, y[i] % num_labels] = 1  # 使用y[i] % num_labels确保标签在范围内
    return Y

def evaluate_predictions(a3,y):
    # 获取实际标签，假设你已经在之前的代码中定义了 true_labels
    true_labels = y.flatten()  # 确保将 y 展平为一维数组
    true_labels[true_labels == 10] = 0  # 将10映射为0

    predicted_labels = np.argmax(a3, axis=1) + 1  # 获取当前行的预测标签

    # 将预测标签中10映射为0
    predicted_labels[predicted_labels == 10] = 0  # 注意：这里是因为索引从0开始，9对应标签10

    # 打印每一行最大值对应的标签
    correct_predictions = 0
    incorrect_samples = []  # 存储不正确的样本

    for i in range(len(predicted_labels)):
        actual_label = true_labels[i]
        predicted_label = predicted_labels[i]

        # 计算正确预测的数量
        if predicted_label == actual_label:
            correct_predictions += 1
        else:
            incorrect_samples.append((i, predicted_label, actual_label))

    # 计算准确率
    accuracy = correct_predictions / len(predicted_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# 加载处理数据
data = loadmat('ex3data1.mat')
weights = loadmat('ex3weights.mat')
X = data['X']
y = data['y']
y[y == 10] = 0
# 将权重导入到 theta1 和 theta2 中
theta1 = weights['Theta1']  # 25x401 的矩阵
theta2 = weights['Theta2']  # 10x26 的矩阵
#数据处理
m = X.shape[0]
X_new = np.insert(X, 0, 1, axis=1)
Y_encoded = one_hot_encode(y,10)
#计算隐藏层
a2= sigmoid(X_new @ theta1.T)
#插入偏置常量
a2_new = np.insert(a2, 0, 1, axis=1)
#计算输出层
a3 = sigmoid(a2_new @ theta2.T)
#计算准确率
evaluate_predictions(a3,y)

