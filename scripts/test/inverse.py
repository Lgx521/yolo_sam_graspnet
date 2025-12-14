import numpy as np

x = np.array([[0.0, -1.0, 0.0, 0.060],
              [1.0,  0.0, 0.0, -0.040],
              [0.0,  0.0, 1.0, -0.110],
              [0.0,  0.0, 0.0, 1.0]])

print(np.linalg.inv(x))