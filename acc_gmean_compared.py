import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

names = ['Evo-Bagging', 'OverBagging', 'ANASYN-Bagging', 'Adaboost', 'Random Forest', 'EE-Bagging', 'SMOTEBoost']

x = range(len(names))
ACC = [0.9431,0.9115,0.9189,0.9142,	0.9283,	0.8967,	0.9092]
AUC = [0.9443,0.8066,0.7924,0.7796,	0.7486,	0.8377,	0.8116]

ACC_var = [0.0442,0.0538,0.0550,0.0520,0.0476,0.0633,0.0568]
ACC_var1 = [0.0221,0.0269,0.0275,0.0260,0.0238,0.0327,0.0284]
AUC_var = [0.0426,0.1504,0.1671,0.1778,0.2202,0.1211,0.1475]
AUC_var1 = [0.0213, 0.0752,0.0836,0.0889,0.1101,0.0606,0.0738]

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}

fig = plt.figure(figsize=(8,4))
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)
ax1 = fig.add_subplot(111)
line1 = ax1.errorbar(x, ACC,ACC_var1, marker='o', ms=6, label='Accuracy',linestyle='-',fmt='o',color='b',elinewidth=2,capsize=4)
# ax1.legend(loc=1)
ax1.set_ylabel('Accuracy',color='b',size=12,weight='normal')
ax1.set_ylim(0.6,1)

ax2 = ax1.twinx()
line2 = ax2.errorbar(x, AUC,AUC_var1, marker='s',color='r', ms=6, label='AUC',linestyle='--',fmt='o',elinewidth=2,capsize=4)
# ax2.legend(loc=2)
ax2.set_ylabel('AUC',color='r',size=12,weight='normal')
ax2.set_ylim(0.6,1)

# added these three lines
# line = line1+line2
# labs = [l.get_label() for l in line]
# ax1.legend(line, labs, loc='lower left')
ax1.set_xticklabels(['']+names,rotation=45)
# plt.xticks(x, names,rotation=45)
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