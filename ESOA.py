import math
import random
from scipy.special import gamma
import numpy as np
import E_HEOA_model as md
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="-2"


def fit_fun(param, X):  # 适应函数,此处为模型训练
    train_data = param['data']
    train_label = param['label']
    model = md.create_model(dropout=X[-2])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adadelta(learning_rate=X[-1]))
    history = model.fit(train_data,train_label,batch_size=1,epochs=1,validation_split=0.2,verbose=1)
    val_loss = min(history.history['val_loss'])
    val_loss = np.float64(val_loss)
    return val_loss



class ESOA:
    def __init__(self, model_param, unity_param, constraint_ueq=None):
        self.model_param = model_param  # 模型参数
        self.n_dim = unity_param['n_dim']
        self.size_pop = unity_param['size_pop']
        self.max_iter = unity_param['max_iter']
        self.lb = unity_param['lb']
        self.ub = unity_param['ub']


    # 白鹭群参数
        self.times = 0
        # adam's learning rate of weight estimate
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.m = np.zeros((self.size_pop, self.n_dim))
        self.v = np.zeros((self.size_pop, self.n_dim))
        self.w = np.random.uniform(-1, 1, size=(self.size_pop, self.n_dim))
        self.g = np.empty_like(self.w)

        # location, fitness, and estimate fitness
        self.x = self.eso_initializationTent(self.size_pop, self.n_dim, self.ub, self.lb)
        self.y = np.empty(self.size_pop)
        self.p_y = self.y.copy()

        # best fitness history and estimate error history
        self.y_hist = []
        self.err = []

        # individual best location, gradient direction, and fitness
        self.x_hist_best = self.x.copy()
        self.g_hist_best = np.empty_like(self.x)
        self.y_hist_best = np.ones(self.size_pop) * np.inf

        # group best location, gradient direction, and fitness
        self.x_global_best = self.x[0].copy()
        self.g_global_best = np.zeros(self.n_dim)

        print("初始化ESOA种群")
        self.y_global_best = fit_fun(self.model_param, self.x[0])
        self.y_global_best = np.float64(self.y_global_best)

        # 本次迭代时小组们最佳的位置与适应度值
        self.now_iter_x_best_esoa = self.x_global_best
        self.now_iter_y_best_esoa = self.y_global_best

        self.pre_iter_x_best_esoa = self.x_global_best  # 上次迭代最优
        self.pre_iter_y_best_esoa = self.y_global_best

        self.qi_p_x_best_esoa = self.x_global_best
        self.qi_p_y_best_esoa = self.y_global_best

        self.hop = self.ub - self.lb



    def run(self):

        # Start search
        for i in range(self.max_iter):

            print(f"第{i+1}次迭代")
        ###  白鹭群算法开始  ###

            print("执行ESOA")
            self.times = i
            self.updateSurface()

            # 根据小队当前位置分裂为白鹭A、B、C个体，分别计算各自代表策略的位置和适应度值
            x_m, y_m, x_n, y_n = self.randomSearch()
            x_o, y_o = self.adviceSearch()

            # 将白鹭A、B、C的适应度进行横向对比，每一组选择最小适应度的记录其位置和适应度
            x_i = np.empty_like(self.x)
            y_i = np.empty_like(self.y)
            x_summary = np.array([x_m, x_n, x_o])
            y_summary = np.column_stack((y_m, y_n, y_o))
            y_summary[y_summary == np.nan] = np.inf
            i_ind = y_summary.argmin(axis=1)
            for i in range(self.size_pop):
                y_i[i] = y_summary[i, i_ind[i]]
                x_i[i, :] = x_summary[i_ind[i]][i]

            # 更新当前每组白鹭小队的适应度与位置
            mask = y_i < self.y
            self.y = np.where(mask, y_i, self.y)
            mask = self.refill(mask)
            self.x = np.where(mask, x_i, self.x)

            # 更新每组白鹭小队的最佳适应度和位置
            mask = y_i < self.y_hist_best
            self.y_hist_best = np.where(mask, y_i, self.y_hist_best)
            mask = self.refill(mask)
            self.x_hist_best = np.where(mask, x_i, self.x_hist_best)

            # 计算本次迭代中本种群中最佳适应度值和位置，用于二次插值
            self.now_iter_x_best = x_i[y_i.argmin(), :]
            self.now_iter_y_best = y_i[y_i.argmin()]
            print("进行高斯变异")
        #高斯变异
            self.qi_p_x_best_esoa = self.gaussian_part(self.now_iter_x_best,self.x_global_best)
            self.qi_p_y_best_esoa = fit_fun(self.model_param,self.qi_p_x_best_esoa)
            # 二次插值预测值和本次迭代种群最佳值做对比，最好的再拿来跟全局最佳值比，以更新全局最佳值
            n_iter_x_best = self.now_iter_x_best
            n_iter_y_best = self.now_iter_y_best
            if self.qi_p_y_best_esoa < self.now_iter_y_best:
                n_iter_x_best = self.qi_p_x_best_esoa
                n_iter_y_best = self.qi_p_y_best_esoa

            # 更新种群最佳位置与适应度
            # 否则，如果当前哪些白鹭小队没有比之前最佳适应度更小，也有0.3的几率去将小队最佳适应度和位置更新为当前情况
            if n_iter_y_best < self.y_global_best:
                self.y_global_best = n_iter_y_best
                self.x_global_best = n_iter_x_best
            else:
                ran = np.random.random(self.size_pop)
                ran = self.refill(ran)
                ran[mask] = 1
                mask = ran < 0.3
                self.x = np.where(mask, x_i, self.x)
                self.y = np.where(mask[:, 0], y_i, self.y)

            # 记录直到本次迭代为止的种群最佳适应度
            self.y_global_best = np.float64(self.y_global_best)
            self.y_hist.append(self.y_global_best.copy())

            # 将上次迭代的小组最佳值和位置更新为本次迭代的小组最佳值和位置，以服务于下次的二次插值计算
            self.pre_iter_x_best_esoa = self.now_iter_x_best
            self.pre_iter_y_best_esoa = self.now_iter_y_best

        print(self.x_global_best)
        print(self.y_global_best)
        return self.x_global_best[-1],self.x_global_best[-2], self.y_global_best.min()


    def gaussian_part(self,x, xbest, mutation_rate=1):
        print(x)
        print(xbest)
        new_pos = np.copy(x)
        if np.random.rand() < mutation_rate:
            sigma = 1
            sqrt_2pi = np.sqrt(2 * np.pi)
            gaussian_values = (1 / (sqrt_2pi * sigma)) * np.exp(-((xbest - x) ** 2) / (2 * sigma ** 2))
            new_pos = np.clip(x * gaussian_values, self.lb, self.ub)
        return new_pos

    def checkBound(self, x):
        return np.clip(x, self.lb, self.ub)


    def eso_initializationTent(self,pop, dim, ub, lb):
        boundary_no = len(ub)
        positions = np.zeros((pop, dim))
        mu = 1
        for i in range(pop):
            for j in range(dim):
                x = np.random.rand()
                if x < mu / 2:
                    x = mu * x
                else:
                    x = mu * (1 - x)
                if boundary_no == 1:
                    positions[i, j] = lb + (ub - lb) * x
                    positions[i, j] = np.clip(positions[i, j], lb, ub)
                else:
                    positions[i, j] = lb[j] + (ub[j] - lb[j]) * x
                    positions[i, j] = np.clip(positions[i, j], lb[j], ub[j])
        return positions

    def refill(self, v):
        v = v.reshape(len(v), 1)
        v = np.tile(v, self.n_dim)
        return v

    # 更新每组小队实际应用梯度的方向
    def gradientEstimate(self, g_temp):
        # 计算每组小队的实际应用梯度方向，应用了当前位置、小队最佳位置、当前适应度值、小队最佳适应度值
        p_d = self.x_hist_best - self.x
        p_d_sum = p_d.sum(axis=1)
        p_d_sum = self.refill(p_d_sum)
        f_p_bias = self.y_hist_best - self.y
        f_p_bias = self.refill(f_p_bias)
        p_d *= f_p_bias
        p_d /= (p_d_sum + np.spacing(1)) * (p_d_sum + np.spacing(1))
        d_p = p_d + self.g_hist_best

        # 计算种群最佳的实际应用梯度方向，应用了当前位置、种群最佳位置、当前适应度值、种群最佳适应度值
        c_d = self.x_global_best - self.x
        c_d_sum = c_d.sum(axis=1)
        c_d_sum = self.refill(c_d_sum)
        f_c_bias = self.y_global_best - self.y
        f_c_bias = self.refill(f_c_bias)
        c_d *= f_c_bias
        c_d /= (c_d_sum + np.spacing(1)) * (p_d_sum + np.spacing(1))
        d_g = c_d + self.g_global_best

        # 搞三个随机数种子，用于决定在计算实际应用梯度时实际梯度、小队梯度方向、全局梯度方向所占更新比例的权重
        # (注意这里与原文有些不符合，原文是只搞了两个随机数种子用于决定三个数值所占有更新比例的权重)
        r1 = np.random.random(self.size_pop)
        r1 = self.refill(r1)
        r2 = np.random.random(self.size_pop)
        r2 = self.refill(r2)
        r3 = np.random.random(self.size_pop)
        r3 = self.refill(r3)

        # 根据实际梯度、小队梯度方向、全局梯度方向来更新每组小队实际应用梯度，并做归一化
        self.g = r1 * g_temp + r2 * d_p + r3 * d_g
        g_sum = self.g.sum(axis=1)
        g_sum = self.refill(g_sum)
        self.g /= (g_sum + np.spacing(1))

    # 基于adam更新w参数
    def weightUpdate(self):
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.g
        self.v = self.beta2 * self.v + (1 - self.beta2) * self.g ** 2
        self.w = self.w - self.m / (np.sqrt(self.v) + np.spacing(1))

    # 根据小组当前位置计算更新基础信息，如当前小队适应度值、最佳适应度值、最佳位置、应用梯度方向、最佳梯度方向
    # ，全局最佳适应度、最佳梯度方向、位置，以及参数w
    def updateSurface(self):
        # 计算每组小队的适应度值
        self.y = np.array([fit_fun(self.model_param, self.x[i, :]) for i in range(self.size_pop)]).reshape(self.size_pop)

        # 计算每组小队对当前位置存在猎物所估计的值(与参数w有关)
        self.p_y = np.sum(self.w * self.x, axis=1)
        self.err.append(np.abs(self.y - self.p_y).min())

        # 计算每组小队的实际梯度(不是实际应用梯度)
        p = self.p_y - self.y
        p = self.refill(p)
        g_temp = p * self.x

        # 根据每组小队的适应度值更新每组小队各自的最佳适应度
        mask = self.y < self.y_hist_best
        self.y_hist_best = np.where(mask, self.y, self.y_hist_best)

        # 根据每组小队的适应度值更新每组小队各自的最佳位置和梯度
        mask = self.refill(mask)
        self.x_hist_best = np.where(mask, self.x, self.x_hist_best)
        self.g_hist_best = np.where(mask, g_temp, self.g_hist_best)

        # 根据每组小队梯度计算其对应的梯度方向
        g_hist_sum = self.refill(np.sqrt((self.g_hist_best ** 2).sum(axis=1)))
        self.g_hist_best /= (g_hist_sum + np.spacing(1))

        # 如果此时小队中有适应度小于全局最佳适应度的，则更新全局的最佳适应度、位置、梯度方向
        if self.y.min() < self.y_global_best:
            self.y_global_best = self.y.min()
            self.x_global_best = self.x[self.y.argmin(), :]
            self.g_global_best = g_temp[self.y.argmin(), :]
            self.g_global_best /= np.sqrt(np.sum(self.g_global_best ** 2))

        # 更新每组小队的实际应用梯度方向
        self.gradientEstimate(g_temp)

        # 更新w参数
        self.weightUpdate()

    # 激进策略（白鹭B、白鹭C，即包含了随机和包围策略）
    def randomSearch(self):

        print("激进策略")

        a = 1
        r1 = a - a * self.times / self.max_iter
        r2 = np.random.uniform(0, 2 * math.pi)
        r3 = np.random.uniform(0, 2)
        r4 = np.random.uniform(0, 1)
        if r4 < 0.5:
            x_n = self.x + r1 * np.sin(r2) * np.abs(r3 * self.pre_iter_x_best_esoa * self.x)
        else:
            x_n = self.x + r1 * np.cos(r2) * np.abs(r3 * self.pre_iter_x_best_esoa * self.x)
        # 计算当前每只白鹭B的适应度值
        y_n = np.array([fit_fun(self.model_param, x_n[i, :]) for i in range(self.size_pop)])

        d = self.x_hist_best - self.x
        d_g = self.x_global_best - self.x
        # r = np.random.uniform(-np.pi / 2, np.pi / 2, size=(size_pop, n_dim))
        r = np.random.uniform(0, 0.5, size=(self.size_pop, self.n_dim))
        r2 = np.random.uniform(0, 0.5, size=(self.size_pop, self.n_dim))
        x_m = (1 - r - r2) * self.x + r * d + r2 * d_g
        x_m = self.checkBound(x_m)
        # 计算当前每只白鹭C的适应度值
        y_m = np.array([fit_fun(self.model_param, x_m[i, :]) for i in range(self.size_pop)])


        # 返回每只白鹭C、白鹭B的位置以及适应度值
        return x_m, y_m, x_n, y_n

    # 坐等策略（白鹭A）
    def adviceSearch(self):
        print("坐等策略")
        # 坐等策略即使用了实际应用梯度方向来计算每个白鹭A的位置
        x_o = self.x + np.exp(-self.times / (0.1 * self.max_iter)) * 0.1 * self.hop * self.g
        x_o = self.checkBound(x_o)
        # 计算每只白鹭A的适应度值
        y_o = np.array([fit_fun(self.model_param, x_o[i, :]) for i in range(self.size_pop)])

        # 返回每只白鹭A的位置以及适应度值
        return x_o, y_o
