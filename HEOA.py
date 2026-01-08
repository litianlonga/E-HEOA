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


def fit_fun(param, X):
    train_data = param['data']
    train_label = param['label']
    model = md.create_model(dropout=X[-2])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adadelta(learning_rate=X[-1]))
    history = model.fit(train_data,train_label,batch_size=1,epochs=1,validation_split=0.2,verbose=1)
    val_loss = min(history.history['val_loss'])
    val_loss = np.float64(val_loss)
    return val_loss



class HEOA:
    def __init__(self, model_param, unity_param, constraint_ueq=None):
        self.model_param = model_param
        self.n_dim = unity_param['n_dim']
        self.size_pop = unity_param['size_pop']
        self.max_iter = unity_param['max_iter']
        self.lb = unity_param['lb']
        self.ub = unity_param['ub']

    # 人类进化优化参数
        self.A = 0.6
        self.LN = 0.4
        self.EN = 0.4
        self.FN = 0.1

        # 初始化种群
        self.X0 = self.heoa_initializationLogistic(self.size_pop, self.n_dim, self.ub, self.lb)
        self.X0 = self.checkBound(self.X0)
        self.X = self.X0
        self.X_new = self.X.copy()

        self.fitness = np.zeros(self.size_pop)
        self.fitness_new = np.zeros(self.size_pop)


        self.GBestX = self.X[0].copy()
        print("初始化HEOA种群")
        self.GBestF = fit_fun(self.model_param, self.X[0])

        self.now_iter_x_best_heoa = self.GBestX
        self.now_iter_y_best_heoa = self.GBestF

        self.pre_iter_x_best_heoa = self.GBestX
        self.pre_iter_y_best_heoa = self.GBestF

        self.qi_p_x_best_heoa = self.GBestX
        self.qi_p_y_best_heoa = self.GBestF




    def run(self):


        jump_factor = abs(self.lb - self.ub) / 1000
        LNNumber = round(self.size_pop * self.LN)
        ENNumber = round(self.size_pop * self.EN)
        FNNumber = round(self.size_pop * self.FN)
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
            for j in range(self.size_pop):
                self.X_new[j] = self.checkBound(self.X_new[j])
            for j in range(self.size_pop):
                if i < (1 / 4) * self.max_iter:
                    self.X_new[j, :] = self.GBestX * (1 - (i+1) / self.max_iter) + (np.mean(self.X[j, :]) - self.GBestX) * np.floor(
                                            np.random.randn() / jump_factor) * jump_factor + 0.2 * (1 - (i+1) / self.max_iter) * (
                                                                self.X[j,:] - self.GBestX) * self.Levy(self.n_dim)
                else:
                    for j in range(LNNumber):
                        if R < self.A:
                            self.X_new[j, :] = (0.2 * np.cos(np.pi / 2 * (1 - ((i+1) / self.max_iter))) * self.X[j, :] *
                                                np.exp((-(i+1) * np.random.randn()) / (np.random.rand() * self.max_iter)))
                        else:
                            self.X_new[j, :] = (0.2 * np.cos(np.pi / 2 * (1 - ((i+1) / self.max_iter))) * self.X[j, :] +
                                                np.random.randn() * np.ones(self.n_dim))
                    for j in range(LNNumber, LNNumber + ENNumber):
                        self.X_new[j, :] = np.random.randn() * np.exp((self.X[-1, :] - self.X[j, :]) / (j ** 2))
                    for j in range(LNNumber + ENNumber, LNNumber + ENNumber + FNNumber):
                        self.X_new[j, :] = (self.X[j, :] + 0.2 * np.cos(np.pi / 2 * (1 - ((i+1) / self.max_iter))) *
                                            np.random.rand(self.n_dim) * (self.X[0, :] - self.X[j, :]))
                    for j in range(LNNumber + ENNumber + FNNumber, self.size_pop):
                        self.X_new[j, :] = self.GBestX + (self.GBestX - self.X[j, :]) * np.random.randn()
            self.X_new = abs(self.X_new)
            self.X_new = self.checkBound(self.X_new)
            for j in range(self.size_pop):
                for m in range(self.n_dim):
                    if self.X_new[j][m] == self.ub or self.X_new[j][m] == self.lb:
                        self.X_new[j][m] = self.ub[m] + (random.random()*(self.lb[m] - self.ub[m]))
                self.fitness_new[j] = fit_fun(self.model_param, self.X_new[j, :])
            self.X = self.X_new
            self.fitness = self.fitness_new
            self.fitness, index = self.fitness[np.argsort(self.fitness)], np.argsort(self.fitness)  # sort
            X1 = self.X.copy()
            for j in range(self.size_pop):
                X1[j, :] = self.X[index[j], :]
            for j in range(self.size_pop):
                self.X[j, :] = X1[j, :]
            self.now_iter_y_best = self.fitness[0]
            self.now_iter_x_best = self.X[0, :]


            self.qi_p_x_best_heoa = self.add_cauchy_mutation(self.now_iter_x_best,i,self.max_iter)
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

    def add_cauchy_mutation(self, xbest, t, T, scale_min=0.1, scale_max=0.5):
        # 动态调整scale（指数衰减）
        scale = scale_min + (scale_max - scale_min) * np.exp(-3 * t / T)
        cauchy_sample = cauchy.rvs(size=xbest.shape, loc=0, scale=scale)
        cauchy_sample = self.ub - abs(cauchy_sample) * (self.ub - self.lb)
        x_new = xbest + xbest * cauchy_sample
        x_new = np.clip(x_new, self.lb, self.ub)
        return x_new


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
    def heoa_initializationLogistic(self,pop, dim, ub, lb):
        boundary_no = len(ub)
        positions = np.zeros((pop, dim))
        for i in range(pop):
            for j in range(dim):
                x0 = np.random.rand()
                x = 4 * x0 * (1 - x0)
                if boundary_no == 1:
                    positions[i, j] = ((ub - lb) * x + lb)*np.random.rand()
                    if positions[i, j] > ub:
                        positions[i, j] = ub
                    if positions[i, j] < lb:
                        positions[i, j] = lb
                else:
                    positions[i, j] = ((ub[j] - lb[j]) * x + lb[j])*np.random.rand()
                    if positions[i, j] > ub[j]:
                        positions[i, j] = ub[j]
                    if positions[i, j] < lb[j]:
                        positions[i, j] = lb[j]
        return positions
