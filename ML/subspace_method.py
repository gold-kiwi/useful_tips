import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange

#pathに保存されているMNISTデータを読み込む
def Load_MNIST_DATA_as_Numpy(path):

    X_train = np.load(path + 'mnist_X_train.npy')
    y_train = np.load(path + 'mnist_y_train.npy')
    X_test = np.load(path + 'mnist_X_test.npy')
    y_test = np.load(path + 'mnist_y_test.npy')

    return (X_train, y_train, X_test, y_test)

#MNISTデータをクラスごとに分ける
def Divide_MNIST_DATA_for_Each_Class(X_train, y_train, X_test, y_test):

    mnist_class_count = 50

    X_train_for_each_class = []
    y_train_for_each_class = []
    X_test_for_each_class = []
    y_test_for_each_class = []

    for i in range(mnist_class_count):
        X_train_for_each_class.append([])
        y_train_for_each_class.append([])
        X_test_for_each_class.append([])
        y_test_for_each_class.append([])

    # train data
    for i in range(y_train.shape[0]):
        X_train_for_each_class[y_train[i]].append(X_train[i])
        y_train_for_each_class[y_train[i]].append(y_train[i])

    for i in range(y_test.shape[0]):
        X_test_for_each_class[y_test[i]].append(X_test[i])
        y_test_for_each_class[y_test[i]].append(y_test[i])

    return (
        np.array(X_train_for_each_class),
        np.array(y_train_for_each_class),
        np.array(X_test_for_each_class),
        np.array(y_test_for_each_class)
        )

def PCA_for_Subspace_Method(data):
    pca = PCA()
    pca.fit(np.array(data))

    np_data = np.array(data)

    U = np.zeros((np_data.shape[1] , np_data.shape[1]))

    #print('主成分の次元ごとの寄与率')
    #print(pca.explained_variance_ratio_)
    #print('固有ベクトル')
    #print(pca.components_)

    U = pca.components_
    return U

def Make_Subspace(data,data_path,class_count,component_count):
    U = []
    for i in range(class_count):
        U.append([])
    for i in range(class_count):
        U[i] = PCA_and_Eigen(data[i] , len(data[1]), component_count)
    np.save(data_path + 'U_for_mnist.npy',U)

    return U

def Inference(input_vector, base_vectors, U, class_count):
    cos_list = np.zeros((class_count))
    for i in range(class_count):
        P = np.dot(U[i],U[i].transpose())
        projected_input_vector = np.dot(P,input_vector)
        cos_list[i] = Calc_Cosine_Similarity_(U.components_ , projected_input_vector)

    return ( np.argmax(cos_list[i]) , cos_list[i] )

def main():
    mnist_data_path = './MNIST_DATA/'
    component_count = 35
    mnist_class_count = 10
    data_size = 28 * 28

    # DATA LOAD
    [X_train, y_train, X_test, y_test] = Load_MNIST_DATA_as_Numpy(mnist_data_path)
    print('DATA LOADED')

    # DIVIDE DATA for EACH CLASS
    [X_train_for_each_class,
        y_train_for_each_class,
        X_test_for_each_class,
        y_test_for_each_class
        ] = Divide_MNIST_DATA_for_Each_Class(X_train, y_train, X_test, y_test)
    print('DATA DIVIED')

    # DATA SAVE
    np.save(mnist_data_path + 'X_train_for_each_class.npy',X_train_for_each_class)
    np.save(mnist_data_path + 'y_train_for_each_class.npy',y_train_for_each_class)
    np.save(mnist_data_path + 'X_test_for_each_class.npy',X_test_for_each_class)
    np.save(mnist_data_path + 'y_test_for_each_class.npy',y_test_for_each_class)
    print('DATA SAVED')



    # PCA
    U_Group = np.zeros((mnist_class_count,component_count,data_size))
    print('Training Process')
    for i in trange(mnist_class_count):
        train_data = np.array(X_train_for_each_class[i])
        #train_data = train_data[0:5000,:]
        tmp_U = PCA_for_Subspace_Method(train_data)
        np_tmp_U = np.array(tmp_U)
        U_Group[i,0:component_count,:] = np_tmp_U[0:component_count,:]



    result_matrix = np.zeros((y_test.shape[0],int(mnist_class_count)))
    result_and_answer_list = np.zeros((y_test.shape[0],2)) #0が正解、1が推定結果



    print('Test Process')
    for test_index in trange(len(y_test)):
        norm_list = np.zeros((mnist_class_count))
        for c in range(mnist_class_count):
            projected_vector = np.dot(U_Group[c] , X_test[test_index])
            norm_list[c] = np.linalg.norm(projected_vector)
        result_and_answer_list[test_index,0] = y_test[test_index]
        result_and_answer_list[test_index,1] = np.argmax(norm_list)

        result_matrix[test_index,:] = norm_list

    correct_counter = 0
    for i in range(len(y_test)):
        if result_and_answer_list[i][0] == result_and_answer_list[i][1]:
            correct_counter += 1
    print('accuracy : ', end='')
    print(correct_counter / len(y_test))






if __name__ == '__main__':
    main()
