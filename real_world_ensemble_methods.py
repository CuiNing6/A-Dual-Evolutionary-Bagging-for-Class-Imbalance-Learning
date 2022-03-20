import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

names = ['Evo-Bagging', 'OverBagging', 'Random Forest', 'EE-Bagging', 'ANASYN-Bagging',  'SMOTEBoost', 'Adaboost']

x = range(len(names))
F1 =        [0.9866, 0.9210, 0.8798, 0.8954, 0.8808, 0.8765, 0.8695]
ACC =       [0.9844, 0.9823, 0.9768, 0.9576, 0.9641, 0.9576, 0.9610]
AUC =       [0.9847, 0.9289, 0.9042, 0.8591, 0.8593, 0.8591, 0.8531]
Recall =    [1.0,    0.8583, 0.8083, 0.7292, 0.7208, 0.7292, 0.7292]
Precision = [0.9752, 0.9952, 1.0,    0.8900, 0.9710, 0.8900, 0.9833]
Gmean =     [0.9846, 0.9258, 0.9510, 0.9475, 0.9355, 0.9351, 0.9203]


plt.figure(figsize=(8,4.2))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.plot(x, ACC, marker='o', ms=6, label='Accuracy',linewidth=2)
plt.plot(x, F1, marker='*', ms=6, label='F1-score',linestyle='-.',linewidth=2)
plt.plot(x, AUC, marker='s', ms=6, label='AUC',linestyle=':',linewidth=2)
plt.plot(x, Gmean, marker='+', ms=6, label='G-mean',linestyle='--',linewidth=2)
# plt.plot(x, Recall, marker='s', ms=6, label='Recall')
# plt.plot(x, Precision, marker='o', ms=6, label='Precision')

plt.legend(loc='lower left', fontsize=10) # 让图例生效
# plt.grid(axis = 'y')
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')

plt.xticks(x, names,rotation=45)
plt.margins(0.1)
plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0])
plt.subplots_adjust(bottom=0.05)

# plt.xlabel("sampling rate") #X轴标签
plt.ylabel("magnitude") #Y轴标签
# plt.title("A simple plot") #标题
plt.tight_layout()

# plt.savefig("../result/number8.png")

plt.show()