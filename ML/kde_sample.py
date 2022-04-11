import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# データ生成。 今回はラベルは不要なのでデータだけ取得する。
data, _ = make_moons(
    n_samples=200,
    noise=0.1,
    random_state=0
)

data_kde = np.vstack([data[:, 0], data[:, 1]]).T 
#bandwidth = 1.0
#kde_skl = KernelDensity(bandwidth=bandwidth) 
#kde_skl.fit(data_kde) 

# score_samples() returns the log-likelihood of the samples 
#z = np.exp(kde_skl.score_samples(xy_sample)) 

fig = plt.figure(facecolor="w")
ax = fig.add_subplot(111, title="sample data")
ax.scatter(data[:, 0], data[:, 1])
fig.savefig('sample_data.png')

X = np.vstack([data[:, 0], data[:, 1]]).T

#bw_method = 'silverman'
bw_method = 0.25
kde = gaussian_kde(X.T,bw_method=bw_method)

# 等高線を引く領域のx座標とy座標のリストを用意する
x = np.linspace(-1.5, 2.5, 500)
y = np.linspace(-0.8, 1.3, 500)
# メッシュに変換
xx, yy = np.meshgrid(x, y)
# kdeが受け取れる形に整形
meshdata = np.vstack([xx.ravel(), yy.ravel()])
# 高さのデータ計算
z = kde.evaluate(meshdata)

print(z.shape)
print(z.max())
print(z.min())

# 可視化
fig = plt.figure(facecolor="w")
ax = fig.add_subplot(111, title="KDE")
#ax.contourf(xx, yy, z.reshape(len(y), len(x)), cmap="Blues", alpha=0.5)
ax.pcolormesh(xx, yy, z.reshape(len(y), len(x)))
ax.scatter(data[:, 0], data[:, 1], c="b")
fig.savefig('kde_result_%s.png' % str(bw_method))

#plt.show()