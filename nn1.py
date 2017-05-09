import numpy as np
import pickle
import lib.mathfunc as mathfunc
import input_mnist

def get_train_data():
    mnist = input_mnist.read_data_sets("MNIST_data/", one_hot=False)
    return mnist.train.images, mnist.train.labels

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = mathfunc.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = mathfunc.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = mathfunc.softmax(a3)

    return y

x, label = get_train_data()
network = init_network()

accuracy_cnt = 0
batch_size = 100
for i in range(0, len(x), batch_size):
    y = predict(network, x[i:i+batch_size])
    p = np.argmax(y, axis = 1)
    accuracy_cnt += np.sum(p == label[i:i+batch_size])

print("Acuuracy:" + str(float(accuracy_cnt) / len(x)))
