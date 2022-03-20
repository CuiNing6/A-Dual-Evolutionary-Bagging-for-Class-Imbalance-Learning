import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

names = ['Evo-Bagging', 'OverBagging', 'ANASYN-Bagging', 'Adaboost', 'Random Forest', 'EE-Bagging', 'SMOTEBoost']

x = range(len(names))
ACC = [1.975,4.9,4.0,4.05,	3.15,	5.275,	4.6]
AUC = [1.375,4.125,4.3,4.775,	5.65,	3.35,	4.2]

ACC_var = [1.39,1.34,1.21,1.65,1.46,1.675,1.44]
# ACC_var1 = [0.0221,0.0269,0.0275,0.0260,0.0238,0.0327,0.0284]
AUC_var = [0.84,1.55,1.52,1.62,1.04,1.17,1.53]
# AUC_var1 = [0.0213, 0.0752,0.0836,0.0889,0.1101,0.0606,0.0738]

fig = plt.figure(figsize=(8,4))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)
ax1 = fig.add_subplot(111)
line1 = ax1.errorbar(x, ACC,ACC_var, marker='o', ms=6, label='Rank(Accuracy)',linestyle='-',fmt='o',color='b',elinewidth=2,capsize=4)
# ax1.legend(loc=1)
ax1.set_ylabel('Rank(Accuracy)',color='b',size=12,weight='normal')
ax1.set_ylim(0,8)

ax2 = ax1.twinx()
line2 = ax2.errorbar(x, AUC,AUC_var, marker='s',color='r', ms=6, label='Rank(AUC)',linestyle='--',fmt='o',elinewidth=2,capsize=4)
# ax2.legend(loc=2)
ax2.set_ylabel('Rank(AUC)',color='r',size=12,weight='normal')
ax2.set_ylim(0,8)

# added these three lines
# line = line1+line2
# labs = [l.get_label() for l in line]
# ax1.legend(line, labs, loc='lower left')
ax1.set_xticklabels(['']+names,rotation=45)
plt.margins(0.1)
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')

# plt.xticks(x, names,rotation=45)
# plt.margins(0.1)
# plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.subplots_adjust(bottom=0.3)

# plt.xlabel("sampling rate") #X轴标签
# plt.ylabel("magnitude") #Y轴标签
# plt.title("A simple plot") #标题
# plt.tight_layout()

# plt.savefig("../result/number8.png")

plt.show()