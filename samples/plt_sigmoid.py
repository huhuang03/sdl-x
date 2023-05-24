import matplotlib.pyplot as plt
import numpy as np
from sdl import sigmoid

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.show()