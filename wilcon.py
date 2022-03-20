# coding=utf-8
import pandas as pd
import numpy as np
import scipy.stats as stats

class wilconing(object):

    def run(self, alg1, alg2):
        WSRT = np.concatenate((np.array(alg1).reshape(-1, 1), np.array(alg2).reshape(-1, 1)), axis=1)
        diff = (WSRT[:, 0] - WSRT[:, 1]).reshape(-1, 1)

        is_pos = diff.flatten() > 0  # 正值还是负值
        index = np.argsort(-abs(diff.flatten()))  # 按diff 绝对值 从大到小排序

        rank = np.empty([len(index)])
        n = len(index)
        # 排序
        for i in range(len(index)):
            rank[index[i]] = int(n)
            n -= 1
        # 添加正负号
        for i in range(len(rank)):
            if is_pos[i] == False:
                rank[i] = -rank[i]

        R_plus, R_minues = 0, 0
        for i in range(len(rank)):
            if rank[i] > 0:
                R_plus += rank[i]
            else:
                R_minues += rank[i]

        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(alg1, alg2, zero_method='wilcox', correction=False)

        if p_value <= 0.05 and abs(R_plus) > abs(R_minues):
            return 1
        if p_value <= 0.05 and abs(R_plus) < abs(R_minues):
            return -1
        if p_value <= 0.05 and abs(R_plus) == abs(R_minues):
            return 0
        if p_value > 0.05:
            return 0
        if np.isnan(p_value):
            return 0


    def run7(self, alg1, alg2):
        WSRT = np.concatenate((np.array(alg1).reshape(-1, 1), np.array(alg2).reshape(-1, 1)), axis=1)
        diff = (WSRT[:, 0] - WSRT[:, 1]).reshape(-1, 1)
        WSRT = np.concatenate((WSRT, diff), axis=1)

        is_pos = diff.flatten() > 0  # 正值还是负值
        index = np.argsort(-abs(diff.flatten()))  # 按diff 绝对值 从大到小排序

        rank = np.empty([len(index)])
        n = len(index)
        # 排序
        for i in range(len(index)):
            rank[index[i]] = int(n)
            n -= 1

        rank_plus, rank_minus = [], []

        # 添加正负号
        for i in range(len(rank)):
            if is_pos[i] == False:
                rank_minus.append(int(rank[i]))
                rank_plus.append("-")
            else:
                rank_minus.append('-')
                rank_plus.append(int(rank[i]))

        # if diff = 0 重新计算 R+,R-
        for i in range(len(rank)):
            if abs(diff[i]) <= 0.0001:
                if rank_plus[i] == '-':
                    rank_minus[i] = 0.5 * rank_minus[i]
                    rank_plus[i] = rank_minus[i]
                if rank_minus[i] == '-':
                    rank_minus[i] = 0.5 * rank_plus[i]
                    rank_plus[i] = rank_plus[i]

        for i in range(len(WSRT)):
            for j in range(len(WSRT[0])):
                WSRT[i][j] = (str(WSRT[i][j]) + '000')[0:6]
        WSRT = np.concatenate((WSRT, np.array(rank_plus).reshape(-1, 1), np.array(rank_minus).reshape(-1, 1)), axis=1)

        R_plus, R_minues = 0, 0
        for i in range(len(rank)):
            if rank_plus[i] == '-':
                R_minues += rank_minus[i]
            if rank_minus[i] == '-':
                R_plus += rank_plus[i]
            if rank_plus[i] != '-' and rank_minus[i] != '-':
                R_minues += rank_minus[i]
                R_plus += rank_plus[i]

        # Wilcoxon signed-rank test
        statistic, p_value = stats.wilcoxon(alg1, alg2, zero_method='wilcox', correction=False)

        return WSRT, p_value, R_plus, R_minues

