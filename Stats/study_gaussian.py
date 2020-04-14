# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import ctypes
import time
import cv2
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def Image_Replacement(path1='./test.jpg',path2='./IMG_5109.JPG'):

    #平均と分散の初期値
    mu = np.array([0,0])
    sigma = np.array([[1,0],
                      [0,1]])

    path_to_graph_image = './gaussian.png'

    plt.ion()

    while True:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = y = np.arange(-20, 20, 0.5)
        X, Y = np.meshgrid(x, y)

        #行列式
        det = np.linalg.det(sigma)
        #逆行列
        inv_sigma = np.linalg.inv(sigma)
        #ガウス二次元確率密度を返す関数
        def f(x, y):
            x_c = np.array([x, y]) - mu
            return np.exp(- x_c.dot(inv_sigma).dot(x_c[np.newaxis, :].T) / 2.0) / (2*np.pi*np.sqrt(det))
        #配列それぞれ対応するものを返す関数に変える
        Z = np.vectorize(f)(X,Y)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)


        plt.draw()
        #plt.show()
        plt.pause(1)

        #plt.savefig( path_to_graph_image )

        # フレームを表示する
        #cv2.imshow('Gaussian' , frame)
        #frame = cv2.imread( path_to_graph_image )

        k = cv2.waitKey(1)&0xff

        if k == ord('q'):
            plt.clf()
            exit()
        elif k == ord('w'):
            plt.clf()
            mu[0] += 1
        elif k == ord('s'):
            plt.clf()
            mu[1] += 1
        elif k == ord('e'):
            plt.clf()
            mu[0] -= 1
        elif k == ord('d'):
            plt.clf()
            mu[1] -= 1
        else:
            plt.close()
            continue




    cv2.destroyAllWindows()

Image_Replacement()
