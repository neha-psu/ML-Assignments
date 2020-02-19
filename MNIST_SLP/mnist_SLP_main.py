#%matplotlib notebook
import matplotlib.pyplot as plt
import mnist_SLP

p = mnist_SLP.perceptron_mnist('mnist_train.csv', 'mnist_test.csv')
#p.re_init()
epoch = 70  # number of iterations
eta = 0.1 # learning rate
p.learn(epoch, eta)

# print confusion matrix

actual = p.test_target_value
predicted = p.predict()
conf_matrix= p.confuse_matrix(actual, predicted)
print("Confusion matrix for learning rate ", eta, "\n")
print(conf_matrix.astype(int))

# Plot the Accuray graph

plt.plot(p.train_accuracy, label='Train')
plt.plot(p.test_accuracy, label='Test')
plt.ylabel('Accuracy in %')
plt.xlabel('epoch')
plt.title('For Learning rate 0.1')
plt.legend()
plt.show()