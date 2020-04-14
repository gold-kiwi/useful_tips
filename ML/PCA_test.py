import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def PCA_for_Subspace(data):

    print('PCA Start')
    print(data)

    pca = PCA()
    np_data = np.array(data)
    pca.fit(np.array(np_data))

    U = np.zeros((np_data.shape[1] , np_data.shape[1]))

    print('主成分の次元ごとの寄与率')
    print(pca.explained_variance_ratio_)
    print('固有ベクトル')
    print(pca.components_)

    U = pca.components_
    return U

def main():
    x = np.array([[1, 2, 5], [2, 4, 5], [3, 6, 2], [4, 8,7], [1,4,7]])
    y = np.array([[1], [5], [3], [4], [2]])
    N = x.shape[0]
    print(x,N)




    x = np.array([[1, 2], [2, 4], [3, 6]])



    R = 0
    for i in range(3):
        R += np.dot(x[i],x[i].transpose())

    print(R / N)

    x = np.array([[1, 2, 5], [2, 4, 5], [3, 6, 2], [4, 8,7], [1,4,7]])


    a = np.array([[1,2,3],[4,5,6]])
    conv_a = np.cov(x.T)
    print('自己共分散行列')
    print(conv_a)



    aTa = np.dot( a.transpose() , a)
    print(aTa)

    component_count = 3
    pca = PCA(n_components = component_count)
    pca.fit(aTa)
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(aTa)

    print('主成分の次元ごとの寄与率を出力する')
    print(pca.explained_variance_ratio_)
    print('Eigen Vector')
    print(pca.components_)


    U = PCA_for_Subspace(aTa)
    print(U)
    '''
    # グラフ描画サイズを設定する
    plt.figure(figsize=(12, 4))

    # 元データをプロットする
    plt.subplot(1, 2, 1)
    plt.scatter(features[:, 0], features[:, 1])
    plt.title('origin')
    plt.xlabel('x')
    plt.ylabel('y')
    '''

    '''
    # 主成分分析する

    component_count = 2

    pca = PCA(n_components = component_count)
    pca.fit(features_x)
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(features_x)
    '''



    '''
    # 主成分をプロットする
    plt.subplot(1, 2, 2)
    plt.scatter(transformed[:, 0], transformed[:, 1])
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    '''


    '''
    print('主成分の次元ごとの寄与率を出力する')
    print(pca.explained_variance_ratio_)
    print('U')
    print(pca.components_)


    print('')
    U = pca.components_
    P = np.dot(U , U.transpose())
    print(P)

    print('')
    U_small = U[0:2,:]
    print(U_small)
    '''


if __name__ == '__main__':
    main()
