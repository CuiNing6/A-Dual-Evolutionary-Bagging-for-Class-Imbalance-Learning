import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

names = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']

x = range(len(names))
ROS =        [0.9146, 0.9268, 0.9512, 0.9634, 0.9726, 0.9762, 0.9790, 0.9789, 0.9791, 0.9790]
SMOTE =       [0.9230, 0.9472, 0.9625, 0.9737, 0.9832, 0.9862, 0.9873, 0.9874, 0.9873, 0.9874]
SMOTE_borderline1 =       [0.9201, 0.9615, 0.9743, 0.9875, 0.9901, 0.9920, 0.9919, 0.9919, 0.9920, 0.9920]
SMOTE_borderline2 =    [0.9212,    0.9500, 0.9683,    0.9757, 0.9797, 0.9815, 0.9813,    0.9815, 0.9815, 0.9814]
SMOTE_SVM = [0.9268, 0.9583,    0.9638, 0.9767,    0.9837, 0.9875,    0.9875, 0.9875,    0.9874, 0.9875   ]


plt.figure(figsize=(8,4))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.plot(x, ROS, marker='o', ms=6, label='ROS',linewidth=2)
plt.plot(x, SMOTE, marker='*', ms=6, label='SMOTE',linestyle='-.',linewidth=2)
plt.plot(x, SMOTE_borderline1, marker='s', ms=6, label='SMOTE-borderline1',linestyle=':',linewidth=2)
plt.plot(x, SMOTE_borderline2, marker='+', ms=6, label='SMOTE-borderline2',linestyle='--',linewidth=2)
plt.plot(x, SMOTE_SVM, marker='+', ms=6, label='SMOTE_SVM',linestyle='--',linewidth=2)
# plt.plot(x, Recall, marker='>', ms=6, label='Recall')
# plt.plot(x, Precision, marker='v', ms=6, label='Precision',linestyle=':')

plt.legend(loc='lower right', fontsize=10) # 让图例生效
# plt.grid(axis = 'y')
ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')

plt.xticks(x, names)
plt.margins(0.1)
plt.yticks([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
plt.subplots_adjust(bottom=0.05)

plt.xlabel("Number of iterations") #X轴标签
plt.ylabel("AUC") #Y轴标签
# plt.title("A simple plot") #标题
plt.tight_layout()

# plt.savefig("../result/number8.png")

plt.show()