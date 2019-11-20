# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import pandas as pd


# sml algorithm
def sml():
    # load picture:
    # w1:
    t_d1 = load_picture("D:/ML_project/WebSearchHW/dataset/w1/")
    # w2:
    t_d2 = load_picture("D:/ML_project/WebSearchHW/dataset/w2/")
    # w3:
    t_d3 = load_picture("D:/ML_project/WebSearchHW/dataset/w3/")
    # w4:
    t_d4 = load_picture("D:/ML_project/WebSearchHW/dataset/w4/")

    # ################################USING-MATRIX-TO-CALCULATE###################################

    print("loading picture...")
    # for each picture I in w1/w2/w3/w4:
    # split the picture I in to N regions: 8×8 size, overlap length is 2.
    spl_t_d1 = split_picture(t_d1)
    spl_t_d2 = split_picture(t_d2)
    spl_t_d3 = split_picture(t_d3)
    spl_t_d4 = split_picture(t_d4)

    print("DCT")
    # for each split picture calculate its DCT
    dct_spl_t_d1 = dct(spl_t_d1)
    dct_spl_t_d2 = dct(spl_t_d2)
    dct_spl_t_d3 = dct(spl_t_d3)
    dct_spl_t_d4 = dct(spl_t_d4)

    # # for each DCT picture rearrange it by its channels.
    # r_dct_spl_t_d1 = rearrange(dct_spl_t_d1)
    # r_dct_spl_t_d2 = rearrange(dct_spl_t_d2)
    # r_dct_spl_t_d3 = rearrange(dct_spl_t_d3)
    # r_dct_spl_t_d4 = rearrange(dct_spl_t_d4)

    # # compress the rearranged DCT
    # cp_dct_d1 = compress_dct(dct_spl_t_d1)
    # cp_dct_d2 = compress_dct(dct_spl_t_d2)
    # cp_dct_d3 = compress_dct(dct_spl_t_d3)
    # cp_dct_d4 = compress_dct(dct_spl_t_d4)

    print("K-means")
    # k-means to initialize mu:
    mu_d1, pi_d1 = k_means(dct_spl_t_d1)
    mu_d2, pi_d2 = k_means(dct_spl_t_d2)
    mu_d3, pi_d3 = k_means(dct_spl_t_d3)
    mu_d4, pi_d4 = k_means(dct_spl_t_d4)

    print("image EM")
    # for each picture I, estimate Gaussian distribution with EM algorithm(Using 4 Gaussian distributions)
    mu_d1, sigma_d1, pi_d1 = single_em(dct_spl_t_d1, mu_d1, pi_d1)
    mu_d2, sigma_d2, pi_d2 = single_em(dct_spl_t_d2, mu_d2, pi_d2)
    mu_d3, sigma_d3, pi_d3 = single_em(dct_spl_t_d3, mu_d3, pi_d3)
    mu_d4, sigma_d4, pi_d4 = single_em(dct_spl_t_d4, mu_d4, pi_d4)

    # ################################THE-END-OF-THIS-CALCULATION####################################

    print("class K-means")
    # k-means for ex_em:
    mu_class_d1, pi_class_d1 = k_means_ex_em(mu_d1)
    mu_class_d2, pi_class_d2 = k_means_ex_em(mu_d2)
    mu_class_d3, pi_class_d3 = k_means_ex_em(mu_d3)
    mu_class_d4, pi_class_d4 = k_means_ex_em(mu_d4)

    print("class EM")
    # for each w1/w2/w3/w4, estimate all pictures' Gaussian distributions
    # using extended EM algorithm with 16 Gaussian distributions.
    class_mu_d1, class_sigma_d1, class_pi_d1 = ex_em(mu_d1, mu_class_d1, pi_class_d1)
    class_mu_d2, class_sigma_d2, class_pi_d2 = ex_em(mu_d2, mu_class_d2, pi_class_d2)
    class_mu_d3, class_sigma_d3, class_pi_d3 = ex_em(mu_d3, mu_class_d3, pi_class_d3)
    class_mu_d4, class_sigma_d4, class_pi_d4 = ex_em(mu_d4, mu_class_d4, pi_class_d4)

    # save the converged model information in a python dictionary:
    w1 = {
        # 'img_mu': mu_d1,
        # 'img_sigma': sigma_d1,
        # 'img_pi': pi_d1,
        'class_mu': class_mu_d1,
        'class_sigma': class_sigma_d1,
        'class_pi': class_pi_d1
    }
    df_class_mu_d1 = pd.DataFrame(class_mu_d1)
    df_class_mu_d1.to_csv("classModel/w1/class_mu.csv", index=False)
    df_class_sigma_d1 = pd.DataFrame(class_sigma_d1.reshape(class_sigma_d1.shape[0],
                                                            class_sigma_d1.shape[1] * class_sigma_d1.shape[2]))
    df_class_sigma_d1.to_csv("classModel/w1/class_sigma.csv", index=False)
    df_class_pi_d1 = pd.DataFrame(class_pi_d1)
    df_class_pi_d1.to_csv("classModel/w1/class_pi.csv", index=False)
    w2 = {
        # 'img_mu': mu_d2,
        # 'img_sigma': sigma_d2,
        # 'img_pi': pi_d2,
        'class_mu': class_mu_d2,
        'class_sigma': class_sigma_d2,
        'class_pi': class_pi_d2
    }
    df_class_mu_d2 = pd.DataFrame(class_mu_d2)
    df_class_mu_d2.to_csv("classModel/w2/class_mu.csv", index=False)
    df_class_sigma_d2 = pd.DataFrame(class_sigma_d2.reshape(class_sigma_d2.shape[0],
                                                            class_sigma_d2.shape[1] * class_sigma_d2.shape[2]))
    df_class_sigma_d2.to_csv("classModel/w2/class_sigma.csv", index=False)
    df_class_pi_d2 = pd.DataFrame(class_pi_d2)
    df_class_pi_d2.to_csv("classModel/w2/class_pi.csv", index=False)
    w3 = {
        # 'img_mu': mu_d3,
        # 'img_sigma': sigma_d3,
        # 'img_pi': pi_d3,
        'class_mu': class_mu_d3,
        'class_sigma': class_sigma_d3,
        'class_pi': class_pi_d3
    }
    df_class_mu_d3 = pd.DataFrame(class_mu_d3)
    df_class_mu_d3.to_csv("classModel/w3/class_mu.csv", index=False)
    df_class_sigma_d3 = pd.DataFrame(class_sigma_d3.reshape(class_sigma_d3.shape[0],
                                                            class_sigma_d3.shape[1] * class_sigma_d3.shape[2]))
    df_class_sigma_d3.to_csv("classModel/w3/class_sigma.csv", index=False)
    df_class_pi_d3 = pd.DataFrame(class_pi_d3)
    df_class_pi_d3.to_csv("classModel/w3/class_pi.csv", index=False)
    w4 = {
        # 'img_mu': mu_d4,
        # 'img_sigma': sigma_d4,
        # 'img_pi': pi_d4,
        'class_mu': class_mu_d4,
        'class_sigma': class_sigma_d4,
        'class_pi': class_pi_d4
    }
    df_class_mu_d4 = pd.DataFrame(class_mu_d4)
    df_class_mu_d4.to_csv("classModel/w4/class_mu.csv", index=False)
    df_class_sigma_d4 = pd.DataFrame(class_sigma_d4.reshape(class_sigma_d4.shape[0],
                                                            class_sigma_d4.shape[1] * class_sigma_d4.shape[2]))
    df_class_sigma_d4.to_csv("classModel/w4/class_sigma.csv", index=False)
    df_class_pi_d4 = pd.DataFrame(class_pi_d4)
    df_class_pi_d4.to_csv("classModel/w4/class_pi.csv", index=False)

    print("testing...")
    # test test pictures.
    idx = label("D:/ML_project/WebSearchHW/dataset/test.jpg", w1, w2, w3, w4)
    print("the thing in the test picture is: w" + str(idx))


# load picture function:
def load_picture(path):
    img = np.zeros((10, 300, 450, 3))  # 初始化读取的图像，共10张图片，每张图片300行450列3通道
    for i in range(10):
        img[i, :, :, :] = matplotlib.image.imread(path + str(i + 1) + '.jpg')
    return img


# split picture:
def split_picture(img):
    n_width = int((img.shape[2] - 6) / 2)  # 计算切分后，在列上一共有多少块
    n_height = int((img.shape[1] - 6) / 2)  # 计算切分后，在行上一共有多少块
    n = n_width * n_height  # 所有的一共有多少块
    spl_img = np.zeros((10, n, 8, 8, 3))  # 初始化切分后的图像。每一张图片分为n块，每块像素为8*8，3个通道。
    for i in range(10):  # 对每张图片
        for ni in range(n):
            row = int(ni / n_width)  # 计算处在第几行
            column = ni % n_width  # 计算处在第几列
            spl_img[i, ni, :, :, :] = img[i, row:row + 8, column:column + 8, :]  # 将img的像素放到spl_img中
    return spl_img


# split picture for single image:
def split_picture_single_image(img):
    n_width = int((img.shape[1] - 6) / 2)  # 计算切分后，在列上一共有多少块
    n_height = int((img.shape[0] - 6) / 2)  # 计算切分后，在行上一共有多少块
    n = n_width * n_height  # 所有的一共有多少块
    spl_img = np.zeros((n, 8, 8, 3))  # 初始化切分后的图像。每一张图片分为n块，每块像素为8*8，3个通道。
    for ni in range(n):
        row = int(ni / n_width)  # 计算处在第几行
        column = ni % n_width  # 计算处在第几列
        spl_img[ni, :, :, :] = img[row:row + 8, column:column + 8, :]  # 将img的像素放到spl_img中
    return spl_img


# dct:
def dct(spl_img):
    dct_spl_img = np.zeros(spl_img.shape)
    cmp_dct_spl_img = np.zeros((dct_spl_img.shape[0], dct_spl_img.shape[1], int(dct_spl_img.shape[2] / 4),
                                int(dct_spl_img.shape[3] / 4), int(dct_spl_img.shape[4])))
    fl_cmp_dct = np.zeros((cmp_dct_spl_img.shape[0], cmp_dct_spl_img.shape[1],
                           cmp_dct_spl_img.shape[2] * cmp_dct_spl_img.shape[3], cmp_dct_spl_img.shape[4]))
    shuffle_fl = np.zeros((fl_cmp_dct.shape[0], fl_cmp_dct.shape[1], fl_cmp_dct.shape[2] * fl_cmp_dct.shape[3]))
    for i in range(spl_img.shape[0]):  # 数据集的大小，在一个语义中有多少张图片
        for ni in range(spl_img.shape[1]):  # 一张图片被分为了n个区域，然后对每个区域进行操作
            for c in range(spl_img.shape[4]):  # 一张图片共有三个通道。
                dct_spl_img[i, ni, :, :, c] = cv2.dct(spl_img[i, ni, :, :, c])  # calculate split image's dct
                cmp_dct_spl_img[i, ni, :, :, c] = dct_spl_img[i, ni, 0:2, 0:2, c]  # 压缩图像，只取图像左上角。
                fl_cmp_dct[i, ni, :, c] = cmp_dct_spl_img[i, ni, :, :, c].reshape((2 * 2))  # 将图像像素扁平化
            shuffle_fl[i, ni, :] = fl_cmp_dct[i, ni, :, :].T.reshape((fl_cmp_dct.shape[2] * fl_cmp_dct.shape[3]))
            # 将三个通道交错排列。将每一个区域变为1维。
    return shuffle_fl


# # rearrange:
# def rearrange(dct_img):
#     r_dct_img = np.zeros((dct_img.shape[0], dct_img.shape[1]*dct_img.shape[2]))
#     for i in range(dct_img.shape[0]):
#         r_dct_img[i, :] = dct_img[i, :, :].T.reshape((dct_img.shape[1]*dct_img.shape[2]))
#     return r_dct_img


# # compress the rearranged DCT
# def compress_dct(dct_spl_img):
#     return r_dct_img[:, 0:int(r_dct_img.shape[1]/2)]  # 将数据压缩一半


# K-means for single EM algorithm:
def k_means(shuffle_fl):
    mu = np.random.randn(shuffle_fl.shape[0], 4, shuffle_fl.shape[2])  # 随机初始化四个均值
    kn = int(shuffle_fl.shape[1] / 4)
    pi_all = np.ones((shuffle_fl.shape[0], 4)) / 4  # 初始化每个高斯分布的概率值
    for m in range(shuffle_fl.shape[0]):  # 一共10张图片
        for k in range(mu.shape[1]):
            mu[m, k, :] = np.mean(shuffle_fl[m, k * kn:k * kn + kn, :], axis=0)  # 等分N个区域，并求出四个类中的均值，以此初始化。
    c = np.zeros(shuffle_fl.shape[1])
    for n in range(10):  # 迭代10次。
        for m in range(shuffle_fl.shape[0]):  # 一共10张图片
            for i in range(shuffle_fl.shape[1]):  # 每张图片共有N个区域
                c[i] = np.linalg.norm(shuffle_fl[m, i, :] - mu[m, 0, :], ord=2)
                idx = 0
                for j in range(mu.shape[1]):
                    cm = np.linalg.norm(shuffle_fl[m, i, :] - mu[m, j, :], ord=2)
                    if c[i] > cm:
                        c[i] = cm
                        idx = j
                c[i] = idx
            for j in range(mu.shape[1]):
                mu[m, j, :] = np.zeros((mu.shape[2]))
                cj = (c == j) + 0
                mu[m, j, :] = np.sum(np.tile(cj.reshape((cj.shape[0], 1)),
                                             shuffle_fl.shape[2]) * shuffle_fl[m, :, :], axis=0)
                # print("np.sum(c"+str(j)+"):")
                # print(np.sum(cj))
                if np.sum(cj) != 0:
                    mu[m, j, :] = mu[m, j, :] / np.sum(cj)
                else:
                    mu[m, j, :] = np.mean(shuffle_fl[m, :, :], axis=0)
                pi_all[m, j] = np.sum(cj) / shuffle_fl.shape[1]
        # print("c")
        # print(c)

    return mu, pi_all


# Gaussian distribution:
def gaussian(x, mu, sigma):
    d = int(x.shape[0])
    return np.exp(-np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)),
                          (x - mu)) / 2) / (np.power(2 * np.pi, d / 2) * np.power(np.abs(np.linalg.det(sigma)), 0.5))


# Gaussian distribution (sigma matrix calculating with pseudo inverse):
def gaussian_pinv(x, mu, sigma):
    d = int(x.shape[0])
    return np.exp(-np.dot(np.dot((x - mu).T, np.linalg.pinv(sigma)),
                          (x - mu)) / 2) * np.power(np.abs(np.linalg.det(np.linalg.pinv(sigma))), 0.5) / \
           np.power(2 * np.pi, d / 2)


# single EM algorithm:
def single_em(img, mu_all, pi_all):
    sigma_all = np.random.randn(img.shape[0], 4, img.shape[2], img.shape[2])
    for m in range(img.shape[0]):  # 对每一张图片
        x = img[m, :, :]
        mu = mu_all[m, :, :]
        # 图像的GMM为4个分量：
        pi = pi_all[m, :]
        gamma = np.zeros((x.shape[0], 4))
        sigma = np.random.randn(4, x.shape[1], x.shape[1])
        for k in range(4):
            sigma[k, :, :] = np.dot((x - mu[k, :]).T, x - mu[k, :]) / (x.shape[0] - 1)
        for i in range(3):  # 反复执行EM步骤10次
            # E-step:
            sum_ = np.zeros((x.shape[0]))
            for n in range(x.shape[0]):
                for k in range(4):
                    if np.linalg.det(sigma[k, :, :]) == 0:
                        sum_[n] += pi[k] * gaussian_pinv(x[n, :], mu[k, :], sigma[k, :, :])
                    else:
                        sum_[n] += pi[k] * gaussian(x[n, :], mu[k, :], sigma[k, :, :])
            for n in range(x.shape[0]):
                for k in range(4):
                    if np.linalg.det(sigma[k, :, :]) == 0:
                        gamma[n, k] = pi[k] * gaussian_pinv(x[n, :], mu[k, :], sigma[k, :, :]) / sum_[n]
                    else:
                        gamma[n, k] = pi[k] * gaussian(x[n, :], mu[k, :], sigma[k, :, :]) / sum_[n]
            nk = np.sum(gamma, axis=0)
            # M-step:
            for k in range(4):
                mu[k] = np.sum(np.tile(gamma[:, k].reshape(gamma.shape[0], 1), x.shape[1]) * x, axis=0) / nk[k]
                sigma[k] = np.zeros((x.shape[1], x.shape[1]))
                for n in range(x.shape[0]):
                    x_mu = x[n, :] - mu[k]
                    sigma[k] += gamma[n, k] * np.dot(x_mu.reshape((x_mu.shape[0], 1)),
                                                     x_mu.reshape((x_mu.shape[0], 1)).T)
                sigma[k] = sigma[k] / nk[k]
                pi[k] = nk[k] / x.shape[0]
        mu_all[m, :, :] = mu
        sigma_all[m, :, :, :] = sigma
        pi_all[m, :] = pi
    return mu_all, sigma_all, pi_all


# K-means for extended EM algorithm:
def k_means_ex_em(img_mu):
    mu = np.random.randn(16, img_mu.shape[2])  # 随机初始化16个均值，对应16个高斯混合分布
    c = np.zeros((img_mu.shape[0] * img_mu.shape[1]))
    pi = np.ones(16) / 16
    calc_img_mu = img_mu.reshape((img_mu.shape[0] * img_mu.shape[1], img_mu.shape[2]))
    kn = int(calc_img_mu.shape[0] / 16)
    for k in range(16):
        mu[k, :] = np.mean(calc_img_mu[k * kn:(k + 1) * kn, :], axis=0)
    for n in range(3):  # 迭代100次。
        for i in range(calc_img_mu.shape[0]):  # 数据集总数×图像高斯混合分布的个数
            c[i] = np.linalg.norm(calc_img_mu[i, :] - mu[0, :], ord=2)
            idx = 0
            for j in range(mu.shape[0]):
                cm = np.linalg.norm(calc_img_mu[i, :] - mu[j, :], ord=2)
                if c[i] > cm:
                    c[i] = cm
                    idx = j
            c[i] = idx
        # print("c")
        # print(c)
        for j in range(mu.shape[0]):
            mu[j, :] = np.zeros((mu.shape[1]))
            cj = ((c == j) + 0)
            mu[j, :] = np.sum(np.tile(cj.reshape((cj.shape[0], 1)), calc_img_mu.shape[1]) * calc_img_mu, axis=0)
            if np.sum(cj) != 0:
                mu[j, :] = mu[j, :] / np.sum(cj)
            else:
                mu[j, :] = np.mean(calc_img_mu, axis=0)
            pi[j] = np.sum(cj) / calc_img_mu.shape[0]
    return mu, pi


# extended EM algorithm:
def ex_em(img_mu, class_mu, class_pi):
    class_sigma = np.random.randn(class_mu.shape[0], class_mu.shape[1], class_mu.shape[1])
    all_img_mu = img_mu.reshape((img_mu.shape[0] * img_mu.shape[1], img_mu.shape[2]))
    gamma = np.zeros((all_img_mu.shape[0], class_mu.shape[0]))
    for k in range(class_sigma.shape[0]):
        class_sigma[k, :, :] = np.dot((all_img_mu - class_mu[k, :]).T, all_img_mu - class_mu[k, :]) \
                               / (all_img_mu.shape[0] - 1)

    for em in range(3):  # 反复执行EM步骤100次
        # E-step:
        sum_ = np.zeros((all_img_mu.shape[0]))
        for n in range(all_img_mu.shape[0]):
            for k in range(class_mu.shape[0]):
                if np.linalg.det(class_sigma[k]) == 0:
                    sum_[n] += class_pi[k] * gaussian_pinv(all_img_mu[n, :], class_mu[k, :], class_sigma[k, :, :])
                else:
                    sum_[n] += class_pi[k] * gaussian(all_img_mu[n, :], class_mu[k, :], class_sigma[k, :, :])
        # print("sum:")
        # print(sum_)
        for n in range(all_img_mu.shape[0]):
            for k in range(class_mu.shape[0]):
                if np.linalg.det(class_sigma[k]) == 0:
                    gamma[n, k] = class_pi[k] * gaussian_pinv(all_img_mu[n, :], class_mu[k, :], class_sigma[k, :, :]) \
                                  / sum_[n]
                else:
                    gamma[n, k] = class_pi[k] * gaussian(all_img_mu[n, :], class_mu[k, :], class_sigma[k, :, :]) / \
                                  sum_[n]
        nk = np.sum(gamma, axis=0)
        # print("nk:")
        # print(nk)
        # M-step:
        for k in range(class_sigma.shape[0]):
            if nk[k] == 0:
                class_mu[k] = np.sum(np.tile(gamma[:, k].reshape(gamma.shape[0], 1), all_img_mu.shape[1]) * all_img_mu,
                                     axis=0) / all_img_mu.shape[0]
            else:
                class_mu[k] = np.sum(np.tile(gamma[:, k].reshape(gamma.shape[0], 1), all_img_mu.shape[1]) * all_img_mu,
                                     axis=0) / nk[k]
            class_sigma[k] = np.zeros((class_sigma.shape[1], class_sigma.shape[2]))
            for n in range(all_img_mu.shape[0]):
                ac_mu = all_img_mu[n, :] - class_mu[k, :]
                class_sigma[k, :, :] += gamma[n, k] * np.dot(ac_mu.reshape((ac_mu.shape[0], 1)),
                                                             ac_mu.reshape((ac_mu.shape[0], 1)).T)
            if nk[k] == 0:
                class_sigma[k, :, :] = class_sigma[k, :, :] / all_img_mu.shape[0]
                class_pi[k] = 0
            else:
                class_sigma[k] = class_sigma[k] / nk[k]
                class_pi[k] = nk[k] / all_img_mu.shape[0]

    # for em in range(10):  # 反复执行EM步骤100次
    #     # E-step:
    #     for j in range(img_mu.shape[0]):  # j=1,...,|Di|即语义为w的数据集的总大小
    #         for k in range(img_mu.shape[1]):  # k=1,...,K即图像的混合高斯分布一共有多少个高斯分量，本程序中为4个。书中为8个。
    #             sum_ = 0
    #             for m in range(16):  # 16个高斯混合分量
    #                 sum_ += np.power(gaussian(img_mu[j, k, :], class_mu[m, :], class_sigma[m, :, :]) * np.exp(
    #                     -np.trace(np.dot(np.linalg.inv(class_sigma[m, :, :]), img_sigma[j, k, :, :]))), img_pi[j, k])\
    #                         *class_pi[m]
    #             for m in range(16):
    #                 h[m, j, k] = np.power(gaussian(img_mu[j, k, :], class_mu[m, :], class_sigma[m, :, :]) * np.exp(
    #                     -np.trace(np.dot(np.linalg.inv(class_sigma[m, :, :]), img_sigma[j, k, :, :]))), img_pi[j, k])\
    #                              *class_pi[m] / sum_
    #     # M-step:
    #     for m in range(16):  # 16个高斯混合分量
    #         class_pi[m] = np.sum(h[m, :, :]) / (img_mu.shape[0] * img_mu.shape[1])
    #         wm = np.zeros((img_mu.shape[0], img_mu.shape[1]))
    #         class_mu[m, :] = np.zeros(class_mu.shape[1])
    #         class_sigma[m, :, :] = np.zeros((class_sigma.shape[1], class_sigma.shape[2]))
    #         for j in range(img_mu.shape[0]):
    #             for k in range(img_mu.shape[1]):
    #                 wm[j, k] = h[m, j, k] * img_pi[j, k] / np.sum(h[m, :, :] * img_pi)
    #                 class_mu[m, :] += wm[j, k] * img_mu[j, k, :]
    #                 class_sigma += wm[j, k] * (img_sigma[j, k, :, :] + np.power(img_mu[j, k, :] - class_mu[m, :], 2))
    return class_mu, class_sigma, class_pi


# label the picture:
def label(path, w1, w2, w3, w4):
    # img_mu1 = w1['img_mu']
    # img_sigma1 = w1['img_sigma']
    # img_pi1 = w1['img_pi']
    class_mu1 = w1['class_mu']
    class_sigma1 = w1['class_sigma']
    class_pi1 = w1['class_pi']

    # img_mu2 = w2['img_mu']
    # img_sigma2 = w2['img_sigma']
    # img_pi2 = w2['img_pi']
    class_mu2 = w2['class_mu']
    class_sigma2 = w2['class_sigma']
    class_pi2 = w2['class_pi']

    # img_mu3 = w3['img_mu']
    # img_sigma3 = w3['img_sigma']
    # img_pi3 = w3['img_pi']
    class_mu3 = w3['class_mu']
    class_sigma3 = w3['class_sigma']
    class_pi3 = w3['class_pi']

    # img_mu4 = w4['img_mu']
    # img_sigma4 = w4['img_sigma']
    # img_pi4 = w4['img_pi']
    class_mu4 = w4['class_mu']
    class_sigma4 = w4['class_sigma']
    class_pi4 = w4['class_pi']

    img = matplotlib.image.imread(path)
    spl_img = split_picture_single_image(img)
    spl_img = spl_img.reshape((1, spl_img.shape[0], spl_img.shape[1], spl_img.shape[2], spl_img.shape[3]))
    dct_img = dct(spl_img)

    # print(dct_img.shape)
    # print(class_mu1.shape)
    # print(class_sigma1.shape)

    p = np.zeros(4)
    # q = np.zeros(4)
    for n in range(dct_img.shape[1]):
        for k in range(16):
            if np.linalg.det(class_sigma1[k, :, :]) == 0:
                p[0] += class_pi1[k] * gaussian_pinv(dct_img[0, n, :], class_mu1[k, :], class_sigma1[k, :, :])
            else:
                p[0] += class_pi1[k] * gaussian(dct_img[0, n, :], class_mu1[k, :], class_sigma1[k, :, :])
        # if p[0] != 0:
        #     q[0] += np.log(p[0])
        for k in range(16):
            if np.linalg.det(class_sigma2[k, :, :]) == 0:
                p[1] += class_pi2[k] * gaussian_pinv(dct_img[0, n, :], class_mu2[k, :], class_sigma2[k, :, :])
            else:
                p[1] += class_pi2[k] * gaussian(dct_img[0, n, :], class_mu2[k, :], class_sigma2[k, :, :])
        # if p[1] != 0:
        #     q[1] += np.log(p[1])
        for k in range(16):
            if np.linalg.det(class_sigma3[k, :, :]) == 0:
                p[2] += class_pi3[k] * gaussian_pinv(dct_img[0, n, :], class_mu3[k, :], class_sigma3[k, :, :])
            else:
                p[2] += class_pi3[k] * gaussian(dct_img[0, n, :], class_mu3[k, :], class_sigma3[k, :, :])
        # if p[2] != 0:
        #     q[2] += np.log(p[2])
        for k in range(16):
            if np.linalg.det(class_sigma4[k, :, :]) == 0:
                p[3] += class_pi4[k] * gaussian_pinv(dct_img[0, n, :], class_mu4[k, :], class_sigma4[k, :, :])
            else:
                p[3] += class_pi4[k] * gaussian(dct_img[0, n, :], class_mu4[k, :], class_sigma4[k, :, :])
        # if p[3] != 0:
        #     q[3] += np.log(p[3])

    idx = np.argmax(p)

    return idx + 1
