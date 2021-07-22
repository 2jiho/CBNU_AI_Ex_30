from os import sched_yield
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import json

# Data Load
with open("Data.txt", "r") as f:
    lines = f.readlines()
X = []
Y = []
for line in lines:
    data = json.loads(line)
    X.append(
        [
            data["acelerometer"]["x"],
            data["acelerometer"]["y"],
            data["acelerometer"]["z"],
        ]
    )
    Y.append(1 if data["shock"] else 0)
(X, Y) = np.array(X), np.array(Y)


# X, Y = make_blobs(n_samples=3000, n_features=3)  # Test Data
clf = svm.SVC(kernel="linear")
clf.fit(X, Y)
newData = [X[-10:]]
z = (
    lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y)
    / clf.coef_[0][2]
)
tmp = np.linspace(-2, 2, 51)
x, y = np.meshgrid(tmp, tmp)
# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x, y, z(x, y))
ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], "ob")
ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], "sr")
plt.show()
