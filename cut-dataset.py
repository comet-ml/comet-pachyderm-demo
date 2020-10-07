import numpy as np
f = np.load("mnist.npz", allow_pickle=True)
np.savez("mnist-small.npz", x_train=f["x_train"][:30000], y_train=f["y_train"][:30000], x_test=f["x_test"][:5000], y_test=f["y_test"][:5000])
