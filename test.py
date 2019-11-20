from functions import *

# shuffle_fl = np.random.randn(10, 32634, 48)*255
# # mu = k_means(shuffle_fl)
# # print(mu.shape)  # (10, 4, 48)
# mu_all = np.random.randn(10, 4, 48)*255
# #
# mu_all, sigma_all, pi_all = single_em(shuffle_fl, mu_all)
# print(mu_all.shape, sigma_all.shape, pi_all.shape)
# print(mu_all)
# print(sigma_all)
# print(pi_all)
# (10, 4, 48) (10, 4, 48, 48) (10, 4)

# mu_class_d1 = k_means_ex_em(mu_all)
# print(mu_class_d1.shape)
# (16, 48)

# sigma_all = np.random.randn(10, 4, 48, 48)
# pi_all = np.ones((10, 4))/4

# all_mu_all = mu_all.reshape((mu_all.shape[0]*mu_all.shape[1], mu_all.shape[2]))
# mu_class_d1 = np.random.randn(16, 48)*600
# kn = int(all_mu_all.shape[0]/16)
# for k in range(16):
#     mu_class_d1[k, :] = np.mean(all_mu_all, axis=0)
#
# pi_class_d1 = np.ones(16)/16
# # #
# class_mu_d1, class_sigma_d1, class_pi_d1 = ex_em(mu_all, mu_class_d1, pi_class_d1)
# print(class_mu_d1.shape, class_sigma_d1.shape, class_pi_d1.shape)
# # (16, 48) (16, 48, 48) (16,)
# print(class_mu_d1)
# print(class_sigma_d1)
# print(class_pi_d1)


class_mu_w1 = np.array(pd.read_csv('classModel/w1/class_mu.csv'))
class_sigma_w1 = np.array(pd.read_csv('classModel/w1/class_sigma.csv'))
a = int(np.power(class_sigma_w1.shape[1], 0.5))
class_sigma_w1 = class_sigma_w1.reshape((class_sigma_w1.shape[0], a, a))
class_pi_w1 = np.array(pd.read_csv('classModel/w1/class_pi.csv'))
w1 = {
    'class_mu': class_mu_w1,
    'class_sigma': class_sigma_w1,
    'class_pi': class_pi_w1
}

class_mu_w2 = np.array(pd.read_csv('classModel/w2/class_mu.csv'))
class_sigma_w2 = np.array(pd.read_csv('classModel/w2/class_sigma.csv'))
a = int(np.power(class_sigma_w2.shape[1], 0.5))
class_sigma_w2 = class_sigma_w2.reshape((class_sigma_w2.shape[0], a, a))
class_pi_w2 = np.array(pd.read_csv('classModel/w2/class_pi.csv'))
w2 = {
    'class_mu': class_mu_w2,
    'class_sigma': class_sigma_w2,
    'class_pi': class_pi_w2
}

class_mu_w3 = np.array(pd.read_csv('classModel/w3/class_mu.csv'))
class_sigma_w3 = np.array(pd.read_csv('classModel/w3/class_sigma.csv'))
a = int(np.power(class_sigma_w3.shape[1], 0.5))
class_sigma_w3 = class_sigma_w3.reshape((class_sigma_w3.shape[0], a, a))
class_pi_w3 = np.array(pd.read_csv('classModel/w3/class_pi.csv'))
w3 = {
    'class_mu': class_mu_w3,
    'class_sigma': class_sigma_w3,
    'class_pi': class_pi_w3
}

class_mu_w4 = np.array(pd.read_csv('classModel/w4/class_mu.csv'))
class_sigma_w4 = np.array(pd.read_csv('classModel/w4/class_sigma.csv'))
a = int(np.power(class_sigma_w4.shape[1], 0.5))
class_sigma_w4 = class_sigma_w4.reshape((class_sigma_w4.shape[0], a, a))
class_pi_w4 = np.array(pd.read_csv('classModel/w4/class_pi.csv'))
w4 = {
    'class_mu': class_mu_w4,
    'class_sigma': class_sigma_w4,
    'class_pi': class_pi_w4
}

print("testing...")
# test test pictures.
idx = label("D:/ML_project/WebSearchHW/dataset/test.jpg", w1, w2, w3, w4)
print("the thing in the test picture is: w"+str(idx))

# t_d1 = load_picture("D:/ML_project/WebSearchHW/dataset/w1/")
# print("t_d1:")
# print(t_d1)
# spl_t_d1 = split_picture(t_d1)
# print("spl_t_d1")
# print(spl_t_d1)
# dct_spl_t_d1 = dct(spl_t_d1)
# print("dct_spl_t_d1")
# print(dct_spl_t_d1)
# mu_d1, pi_d1 = k_means(dct_spl_t_d1)
# print("mu_d1")
# print(mu_d1)
# print("pi_d1")
# print(pi_d1)
# mu_d1, sigma_d1, pi_d1 = single_em(dct_spl_t_d1, mu_d1, pi_d1)
# print("mu_d1")
# print(mu_d1)
# print("sigma_d1")
# print(sigma_d1)
# print("pi_d1")
# print(pi_d1)
# mu_class_d1, pi_class_d1 = k_means_ex_em(mu_d1)
# print("mu_class_d1")
# print(mu_class_d1)
# print("pi_class_d1")
# print(pi_class_d1)
# class_mu_d1, class_sigma_d1, class_pi_d1 = ex_em(mu_d1, mu_class_d1, pi_class_d1)
# print("class_mu_d1")
# print(class_mu_d1)
# print("class_sigma_d1")
# print(class_sigma_d1)
# print("class_pi_d1")
# print(class_pi_d1)
