import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

names = ['Bagging', 'SVM', 'MLP', 'GB', 'DT', 'KNN', 'SGD', 'FLD', 'LR']

x = range(len(names))
F1 =        [0.8677, 0.8719, 0.8496, 0.8402, 0.8276, 0.7727, 0.8069, 0.7500, 0.6714]
ACC =       [0.9865, 0.9813, 0.9783, 0.9737, 0.9727, 0.9621, 0.9417, 0.9115, 0.9012]
AUC =       [0.8672, 0.8956, 0.8960, 0.7911, 0.8866, 0.8272, 0.8565, 0.8021, 0.7622]
Recall =    [1.0,    0.7911, 1.0,    0.7344, 0.7822, 0.6544, 1.0,    0.7633, 0.6667]
Precision = [0.7787, 1.0,    0.7519, 1.0,    0.9015, 1.0,    0.7842, 1.0,    1.0   ]
Gmean =     [0.9865, 0.8835, 0.9012, 0.8535, 0.8771, 0.7727, 0.8965, 0.7667, 0.7333]


plt.figure(figsize=(8,4))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.plot(x, ACC, marker='o', ms=6, label='Accuracy',linewidth=2)
plt.plot(x, F1, marker='*', ms=6, label='F1-score',linestyle='-.',linewidth=2)
plt.plot(x, AUC, marker='s', ms=6, label='AUC',linestyle=':',linewidth=2)
plt.plot(x, Gmean, marker='+', ms=6, label='G-mean',linestyle='--',linewidth=2)
# plt.plot(x, Recall, marker='>', ms=6, label='Recall')
# plt.plot(x, Precision, marker='v', ms=6, label='Precision',linestyle=':')

plt.legend(loc='lower left', fontsize=10) # 让图例生效
# plt.grid(axis = 'y')
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')

plt.xticks(x, names,rotation=45)
plt.margins(0.1)
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.subplots_adjust(bottom=0.05)

# plt.xlabel("sampling rate") #X轴标签
plt.ylabel("magnitude") #Y轴标签
# plt.title("A simple plot") #标题
plt.tight_layout()

# plt.savefig("../result/number8.png")

plt.show()