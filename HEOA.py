import math
import random
from scipy.special import gamma
import numpy as np
import E_HEOA_model as md
import os
from sympy import symbols, Eq, solve, diff
import tensorflow as tf
from scipy.stats import cauchy
os.environ["CUDA_VISIBLE_DEVICES"]="-2"


def fit_fun(param, X):  # 适应函数,此处为模型训练
    train_data = param['data']
    train_label = param['label']
    model = md.create_model(dropout=X[-2])
    # 传入待优化参数learning_rate
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adadelta(learning_rate=X[-1]))
    # history = model.fit(train_data, train_label, batch_size=32, epochs=2, validation_split=0.2)
    history = model.fit(train_data,train_label,batch_size=1,epochs=1,validation_split=0.2,verbose=1)
    # 获取最小的loss值,优化为loss最小时learning_rate
    val_loss = min(history.history['val_loss'])
    val_loss = np.float64(val_loss)
    return val_loss



class HEOA:
    def __init__(self, model_param, unity_param, constraint_ueq=None):
        self.model_param = model_param  # 模型参数
        self.n_dim = unity_param['n_dim']
        self.size_pop = unity_param['size_pop']
        self.max_iter = unity_param['max_iter']
        self.lb = unity_param['lb']
        self.ub = unity_param['ub']

    # 人类进化优化参数
        self.A = 0.6  # 警告值
        self.LN = 0.4  # 领导者百分比
        self.EN = 0.4  # 探索者百分比
        self.FN = 0.1  # 追随者百分比

        # 初始化种群
        self.X0 = self.heoa_initializationLogistic(self.size_pop, self.n_dim, self.ub, self.lb)
        self.X0 = self.checkBound(self.X0)
        self.X = self.X0
        self.X_new = self.X.copy()  # 存储种群未来的更新

        self.fitness = np.zeros(self.size_pop)
        self.fitness_new = np.zeros(self.size_pop)


        self.GBestX = self.X[0].copy()        # 全局最优
        print("初始化HEOA种群")
        self.GBestF = fit_fun(self.model_param, self.X[0])

        self.now_iter_x_best_heoa = self.GBestX  #本次迭代最优
        self.now_iter_y_best_heoa = self.GBestF

        self.pre_iter_x_best_heoa = self.GBestX  # 上次迭代最优
        self.pre_iter_y_best_heoa = self.GBestF

        self.qi_p_x_best_heoa = self.GBestX      # 拉格朗日计算出来的
        self.qi_p_y_best_heoa = self.GBestF




    def run(self):
        # 类进化优化算法的一些初始量

        jump_factor = abs(self.lb - self.ub) / 1000  # 控制探索的步长
        LNNumber = round(self.size_pop * self.LN)  # 领导者数量
        ENNumber = round(self.size_pop * self.EN)  # 探索者数量
        FNNumber = round(self.size_pop * self.FN)  # 最随者数量
        # 评估适应度
        for i in range(self.size_pop):
            self.fitness[i] = fit_fun(self.model_param,self.X[i,:])
        self.fitness, index = self.fitness[np.argsort(self.fitness)], np.argsort(self.fitness)
        self.GBestF = self.fitness[0]
        for j in range(self.size_pop):
            self.X[j, :] = self.X0[index[j], :]
        self.GBestX = self.X[0, :]
        self.X_new = self.X.copy()
        # Start search
        for i in range(self.max_iter):

            print(f"第{i+1}次迭代")
            R = np.random.rand()
            # Boundary control
            for j in range(self.size_pop):
                self.X_new[j] = self.checkBound(self.X_new[j])
            for j in range(self.size_pop):
                if i < (1 / 4) * self.max_iter:# Explorers阶段
                    self.X_new[j, :] = self.GBestX * (1 - (i+1) / self.max_iter) + (np.mean(self.X[j, :]) - self.GBestX) * np.floor(
                                            np.random.randn() / jump_factor) * jump_factor + 0.2 * (1 - (i+1) / self.max_iter) * (
                                                                self.X[j,:] - self.GBestX) * self.Levy(self.n_dim)
                else: #发展阶段
                    for j in range(LNNumber):  #领导者
                        if R < self.A:
                            self.X_new[j, :] = (0.2 * np.cos(np.pi / 2 * (1 - ((i+1) / self.max_iter))) * self.X[j, :] *
                                                np.exp((-(i+1) * np.random.randn()) / (np.random.rand() * self.max_iter)))
                        else:
                            self.X_new[j, :] = (0.2 * np.cos(np.pi / 2 * (1 - ((i+1) / self.max_iter))) * self.X[j, :] +
                                                np.random.randn() * np.ones(self.n_dim))
                    for j in range(LNNumber, LNNumber + ENNumber):#探索者
                        self.X_new[j, :] = np.random.randn() * np.exp((self.X[-1, :] - self.X[j, :]) / (j ** 2))
                    for j in range(LNNumber + ENNumber, LNNumber + ENNumber + FNNumber):# 跟随者
                        self.X_new[j, :] = (self.X[j, :] + 0.2 * np.cos(np.pi / 2 * (1 - ((i+1) / self.max_iter))) *
                                            np.random.rand(self.n_dim) * (self.X[0, :] - self.X[j, :]))
                    for j in range(LNNumber + ENNumber + FNNumber, self.size_pop):#失败者
                        self.X_new[j, :] = self.GBestX + (self.GBestX - self.X[j, :]) * np.random.randn()
            self.X_new = abs(self.X_new)
            self.X_new = self.checkBound(self.X_new)
            # Update positions
            for j in range(self.size_pop):
                for m in range(self.n_dim):
                    if self.X_new[j][m] == self.ub or self.X_new[j][m] == self.lb:
                        self.X_new[j][m] = self.ub[m] + (random.random()*(self.lb[m] - self.ub[m]))
                self.fitness_new[j] = fit_fun(self.model_param, self.X_new[j, :])
            self.X = self.X_new
            self.fitness = self.fitness_new
            # Sorting and updating
            self.fitness, index = self.fitness[np.argsort(self.fitness)], np.argsort(self.fitness)  # sort
            X1 = self.X.copy()
            for j in range(self.size_pop):
                X1[j, :] = self.X[index[j], :]
            for j in range(self.size_pop):
                self.X[j, :] = X1[j, :]
            self.now_iter_y_best = self.fitness[0]
            self.now_iter_x_best = self.X[0, :]


            print("执行柯西变异")
            self.qi_p_x_best_heoa = self.add_cauchy_mutation(self.now_iter_x_best)
            self.qi_p_y_best_heoa = fit_fun(self.model_param, self.qi_p_x_best_heoa)

            n_iter_x_best = self.now_iter_x_best
            n_iter_y_best = self.now_iter_y_best
            if self.qi_p_y_best_heoa < self.now_iter_y_best:
                n_iter_x_best = self.qi_p_x_best_heoa
                n_iter_y_best = self.qi_p_y_best_heoa
            if n_iter_y_best < self.GBestF:
                self.GBestF = n_iter_y_best
                self.GBestX = n_iter_x_best


            self.pre_iter_x_best_heoa = self.now_iter_x_best
            self.pre_iter_y_best_heoa = self.now_iter_y_best


        return self.GBestX[-1],self.GBestX[-2], self.GBestF.min()

    def add_cauchy_mutation(self,xbest, scale=1):
        print(xbest)
        cauchy_sample = cauchy.rvs(size=xbest.shape, loc=0, scale=scale)
        cauchy_sample = self.ub - abs(cauchy_sample) * (self.ub - self.lb)
        xnewbest = xbest + xbest * cauchy_sample
        xnewbest = np.clip(xnewbest, self.lb, self.ub)
        return xnewbest


    def checkBound(self, x):
        return np.clip(x, self.lb, self.ub)

    def Levy(self,d):
        beta = 1.5
        noise_coeff = 0.05
        sigma = ((gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (
                    gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)*noise_coeff
        u = np.random.randn(d) * sigma
        v = np.random.randn(d)
        step = u / abs(v) ** (1 / beta)
        return step
    # 初始化位置
    def heoa_initializationLogistic(self,pop, dim, ub, lb):
        boundary_no = len(ub)  # number of boundaries
        positions = np.zeros((pop, dim))  # initialize the positions matrix
        for i in range(pop):
            for j in range(dim):
                x0 = np.random.rand()  # 生成一个【0,2】之间的随机数
                x = 4 * x0 * (1 - x0)  # 逻辑映射公式计算x的值
                if boundary_no == 1:
                    positions[i, j] = ((ub - lb) * x + lb)*np.random.rand()  # 线性映射计算初始位置，线性映射将x的值映射到[lb, ub]区间。
                    if positions[i, j] > ub:
                        positions[i, j] = ub
                    if positions[i, j] < lb:
                        positions[i, j] = lb
                else:
                    positions[i, j] = ((ub[j] - lb[j]) * x + lb[j])*np.random.rand()  # 使用线性映射计算初始位置。线性映射将x的值映射到[lb[j], ub[j]]区间。
                    if positions[i, j] > ub[j]:
                        positions[i, j] = ub[j]
                    if positions[i, j] < lb[j]:
                        positions[i, j] = lb[j]
        return positions
