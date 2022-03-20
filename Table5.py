# coding=utf-8
import os
import pandas as pd
import numpy as np

from wilcon import wilconing
from getname import loadname

# Method =["SM", "Bor1", "Bor2", "ADA", "MWM", "ACO", "SMB", "RAB"]
Method =["SM", "Bor1", "Bor2", "ADA", "MWM", "ACO"]

DATASET = ['Abalone_18v9', 'Abalone_17v7', 'Abalone_19v5', 'CTG_PvN', 'CTG_SvN', 'Statland_4v12', 'Statland_5v12', 'Libras_123vA',
'Libras_456vA', 'Libras_789vA', 'Yeast_ME1vCYT', 'Yeast_ME1vNUC', 'Yeast_ME2vCYT', 'Yeast_ME2vNUC', 'Yeast_ME3vCYT', 'Yeast_ME3vNUC',
'Robot_LvF', 'Robot_RvF', 'Ecoil_omvcp', 'Ecoil_imvA', 'Ecoil_ppvcp', 'Ecoil_imvcp', 'Glass_567vA', 'Vehicle_vanvA',
'Vehicle_opelvA', 'Vehicle_saabvA', 'Vehicle_busvA', 'Wine_3vA', 'Wine_1vA', 'Wine_2vA', 'Breast_tissue', 'Breast_cancer',
'Ionosphere', 'Page_4v2', 'Page_5v2', 'Segment_4v123', 'Segment_5v123', 'Segment_6v123', 'Segment_7v123']

# DATASET = ['Abalone_18v9', 'Abalone_17v7', 'Abalone_19v5', 'CTG_PvN', 'CTG_SvN', 'Statland_4v12', 'Statland_5v12', 'Libras_123vA',
# 'Libras_456vA', 'Libras_789vA', 'Yeast_ME1vCYT', 'Yeast_ME1vNUC', 'Yeast_ME2vCYT', 'Yeast_ME2vNUC', 'Yeast_ME3vCYT', 'Yeast_ME3vNUC'
#            ,'Robot_LvF', 'Robot_RvF', 'Ecoil_omvcp', 'Ecoil_imvA', 'Ecoil_ppvcp', 'Ecoil_imvcp', 'Glass_567vA', 'Vehicle_vanvA',
#            'Vehicle_opelvA', 'Vehicle_saabvA', 'Vehicle_busvA','Wine_3vA', 'Wine_1vA', 'Wine_2vA', 'Breast_tissue', 'Breast_cancer',
# 'Ionosphere', 'Page_4v2', 'Page_5v2']

f_out = []
# =================================================  lli1  =============================================
out_row = []
for med in Method:
    a = med
    win, lose, draw = 0, 0, 0
    for dataset in DATASET:
        file_name = loadname(dataset).run()

        data_name2 = file_name + dataset + '/' + med + ".csv"
        data2 = pd.read_csv(data_name2).values[:, 6]
        new_data2 = np.tile(data2, 5)

        data_name1 = file_name + dataset + '/' + 'lli1' + ".csv"
        data1 = pd.read_csv(data_name1).values[:, 6].reshape(-1,1)
        new_data1 = data1.repeat(5)

        value = wilconing().run(new_data1, new_data2)
        if value == 1:
            win += 1
        if value == -1:
            lose += 1
        if value == 0:
            draw += 1
    out = str(win) + '-' + str(lose) + "-" + str(draw)
    out_row.append(out)
f_out.append(out_row)
# =================================================  lli2  =============================================
out_row = []
for med in Method:
    a = med
    win, lose, draw = 0, 0, 0
    for dataset in DATASET:
        file_name = loadname(dataset).run()

        data_name2 = file_name + dataset + '/' + med + ".csv"
        data2 = pd.read_csv(data_name2).values[:, 6]
        new_data2 = np.tile(data2, 5)

        data_name1 = file_name + dataset + '/' + 'lli2' + ".csv"
        data1 = pd.read_csv(data_name1).values[:, 6].reshape(-1,1)
        new_data1 = data1.repeat(5)

        value = wilconing().run(new_data1, new_data2)
        if value == 1:
            win += 1
        if value == -1:
            lose += 1
        if value == 0:
            draw += 1
    out = str(win) + '-' + str(lose) + "-" + str(draw)
    out_row.append(out)
f_out.append(out_row)
# =================================================  lli3  =============================================
out_row = []
for med in Method:
    a = med
    win, lose, draw = 0, 0, 0
    for dataset in DATASET:
        file_name = loadname(dataset).run()

        data_name2 = file_name + dataset + '/' + med + ".csv"
        data2 = pd.read_csv(data_name2).values[:, 6]
        new_data2 = np.tile(data2, 5)

        data_name1 = file_name + dataset + '/' + 'lli3' + ".csv"
        data1 = pd.read_csv(data_name1).values[:, 6].reshape(-1,1)
        new_data1 = data1.repeat(5)

        value = wilconing().run(new_data1, new_data2)
        if value == 1:
            win += 1
        if value == -1:
            lose += 1
        if value == 0:
            draw += 1
    out = str(win) + '-' + str(lose) + "-" + str(draw)
    out_row.append(out)
f_out.append(out_row)
# =================================================  lli4  =============================================
out_row = []
for med in Method:
    a = med
    win, lose, draw = 0, 0, 0
    for dataset in DATASET:
        file_name = loadname(dataset).run()

        data_name2 = file_name + dataset + '/' + med + ".csv"
        data2 = pd.read_csv(data_name2).values[:, 6]
        new_data2 = np.tile(data2, 5)

        data_name1 = file_name + dataset + '/' + 'lli4' + ".csv"
        data1 = pd.read_csv(data_name1).values[:, 6].reshape(-1,1)
        new_data1 = data1.repeat(5)

        value = wilconing().run(new_data1, new_data2)
        if value == 1:
            win += 1
        if value == -1:
            lose += 1
        if value == 0:
            draw += 1
    out = str(win) + '-' + str(lose) + "-" + str(draw)
    out_row.append(out)
f_out.append(out_row)

out = pd.DataFrame(np.array(f_out))
out.to_csv("table5.txt")
