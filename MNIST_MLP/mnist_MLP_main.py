#%matplotlib notebook
import matplotlib.pyplot as plt
import mnist_MLP

epoch=50
eta = 0.1 # learning rate

# vary momentum for experiment 2 (0,0.25,0.5,0.9)
print("Input the momentum")
momentum = float(input())
print("momentum for the experiment is: ",momentum)

# vary number of hidden inputs for experiment-1 (20,50,100)
print("Input the number of hidden nodes: ")
n = int(input())
print("hidden nodes for the experiment is: ",n)

p = mnist_MLP.perceptron_mnist_MLP(n, 'mnist_train.csv', 'mnist_test.csv')
p.learn(epoch,eta,momentum)

# Plot the Accuray graph

plt.plot(p.train_accuracy, label='Train')
plt.plot(p.test_accuracy, label='Test')
plt.ylabel('Accuracy in %')
plt.xlabel('epoch')
plt.title('For Learning rate 0.1')
plt.legend()
plt.show()