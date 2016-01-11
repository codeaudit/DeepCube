import numpy as np
import matplotlib.pyplot as plt

x = 1000 * np.arange(100)
y1 = 0.010 * np.arange(35) + 0.3 + 0.18 * np.random.randn(35)
y2 = 0.017 * np.arange(35, 70) + 0.2 + 0.12 * np.random.randn(35)
y3 = 0.012 * np.arange(70, 100) + 0.8 + 0.17 * np.random.randn(30)

y = np.zeros((100,))
y[:35] = y1
y[35:70] = y2
y[70:100] = y3

y = y / 10
for i in range(y.shape[0]):
    y[i] = max(y[i], 0)

plt.plot(x, y, 'r-')
plt.show()
