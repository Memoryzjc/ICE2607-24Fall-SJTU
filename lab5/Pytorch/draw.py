import numpy as np
import matplotlib.pyplot as plt

test_acc = [
    49.00, 49.33, 59.46, 62.40, 50.84, 46.37, 67.64, 71.78, 69.67, 68.13, 75.05, 74.38, 74.13, 76.61, 78.68, 72.58, 80.79, 77.77, 76.53, 77.43, 85.43, 85.37, 85.71, 85.64, 85.76, 85.93, 86.02, 85.83, 85.99, 86.21
]

plt.plot(range(1, 31), test_acc)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy of ResNet20 in CIFAR-10')
plt.grid()
plt.savefig('./test_acc.png')