import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), 0)

if __name__ == "__main__":
    print("# 1 生成我喜欢猫的one-hot(二进制向量)编码\n")
    vocabulary = {"我":0, "喜欢": 1, "猫":2}
    one_hot_vectors = np.eye(len(vocabulary))

    for word, idx in vocabulary.items():
        print(f"{word}: {one_hot_vectors[idx]}")

    print("# 2 初始化权重矩阵 w1 和 w2\n")
    np.random.seed(0)
    w1 = np.random.rand(3, 3)
    w2 = np.random.rand(3, 3)
    print(f"w1:{w1}\n")
    print(f"w2:{w2}\n")

    print("# 正向传播\n")
    input_vector = one_hot_vectors[vocabulary['我']]
    h = np.dot(w1.T, input_vector)
    u = np.dot(w2.T, h)
    print(f"u:{u}")
    print("# 3 softmax 函数（它通过对向量中的每个元素进行指数运算，并将结果归一化，使得每个元素都在0到1之间，并且所有元素的和等于1）\n")

    y_pred = softmax(u)
    print("# 3 softmax 结果 \n")
    print(y_pred)

    print("# 4 计算损失函数和梯度\n")
    E1 = one_hot_vectors[vocabulary['喜欢']] - y_pred
    E2 = one_hot_vectors[vocabulary['猫']] - y_pred
    E = E1 + E2
    loss = -np.sum(one_hot_vectors[vocabulary['喜欢']]) * np.log(y_pred) + np.sum(one_hot_vectors[vocabulary['猫']]) * np.log(y_pred)
    print(f"loss={loss}\n")

    print("# 5 反向传播和更新权重\n")
    dw2 = np.outer(h, E)
    w2 = w2 - 0.01 * dw2

    dw1 = np.outer(input_vector, np.dot(w2, E))
    w1 = w1 - 0.01 * dw1
    print(f"w1={w1}\n")
    print(f"w2={w2}\n")

    print("# 6 输出‘我’的向量\n")
    print(w1[:, vocabulary["我"]])
