import matplotlib.pyplot as plt
import matplotlib  
# matplotlib.use('Agg')

names = ['1', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']

x = range(len(names))

# maj_recall = [0.84, 0.91, 0.91, 0.92, 0.92, 0.93, 0.92, 0.92, 0.93, 0.93, 0.93, 0.92, 0.92]
Fvalue = [0.78, 0.89, 0.91, 0.92, 0.93, 0.93, 0.92, 0.93, 0.93, 0.94, 0.92, 0.93, 0.92]
Gmean = [0.80, 0.89, 0.91, 0.92, 0.93, 0.93, 0.92, 0.92, 0.93, 0.94, 0.93, 0.92, 0.92]
Recall = [0.75, 0.87, 0.92, 0.92, 0.93, 0.92, 0.93, 0.93, 0.93, 0.92, 0.92, 0.93, 0.93]

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)

# plt.plot(x, maj_recall, marker='*', ms=6, linewidth=2, label='Majority Recall')
plt.plot(x, Fvalue, marker='s', ms=6, linewidth=2, label='F-value')
plt.plot(x, Gmean, marker='*', ms=6, linewidth=2, label='G-mean')
plt.plot(x, Recall, marker='o', ms=6, linewidth=2, label='Recall')

plt.legend(loc='lower right', fontsize=20) # 让图例生效
plt.grid(axis = 'y')
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

plt.xticks(x, names)
plt.margins(0.05)
plt.yticks([0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96])
plt.subplots_adjust(bottom=0.05)

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}
plt.xlabel("Number of base classifiers(The sampling rate of sub-dataset is set to 80%)",font2) #X轴标签
plt.ylabel("Average value of final outputs",font2) #Y轴标签
# plt.title("A simple plot") #标题
plt.tight_layout()

# plt.savefig("../result/number8.png")

plt.show()