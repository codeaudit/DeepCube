import numpy as np
import matplotlib.pyplot as plt

x = 1000 * np.arange(300)
y1 = 0.013 * np.arange(35) + 0.3 + 0.18 * np.random.randn(35)
y2 = 0.023 * np.arange(35, 70) + 0.4 + 0.13 * np.random.randn(35)
y3 = 0.019 * np.arange(70, 100) + 0.9 + 0.18 * np.random.randn(30)

y = np.zeros((100,))
y[:35] = y1
y[35:70] = y2
y[70:100] = y3

y = y / 10

z1 = 0.017 * np.arange(35) + 1.2 + 0.18 * np.random.randn(35)
z2 = 0.020 * np.arange(35, 70) + 1.5 + 0.20 * np.random.randn(35)
z3 = 0.022 * np.arange(70, 100) + 1.7 + 0.17 * np.random.randn(30)

z = np.zeros((100,))
z[:35] = z1
z[35:70] = z2
z[70:100] = z3

z = z / 10

t1 = 0.017 * np.arange(35) + 1.8 + 0.18 * np.random.randn(35)
t2 = 0.023 * np.arange(35, 70) + 2.0 + 0.12 * np.random.randn(35)
t3 = 0.022 * np.arange(70, 100) + 2.2 + 0.17 * np.random.randn(30)

t = np.zeros((100,))
t[:35] = t1
t[35:70] = t2
t[70:100] = t3

t = t / 10

# for i in range(y.shape[0]):
#     y[i] = max(y[i], 0)

u = 0.01 * np.ones((300,)) + 0.01 * np.random.randn(300) + 0.00017 * np.arange(300)

for i in range(300):
    u[i] = max(u[i], 0)

plt.plot(x, np.concatenate((np.concatenate((y, z)), t)), 'r-')
plt.plot(x, u, 'b-')
plt.show()
