import matplotlib.pyplot as plt 
import matplotlib
# mpl.use('Agg')

# matplotlib.use('Agg')

names = ['1', '5', '10', '15', '20', '25', '30', '35', '40', '45']

x = range(len(names))
min_recall = [0.58, 0.75, 0.71, 0.77, 0.77, 0.77, 0.74, 0.75, 0.72, 0.77]
maj_recall = [0.73, 0.76, 0.74, 0.77 , 0.80, 0.78, 0.79, 0.79, 0.76, 0.75]
min_fval = [0.63, 0.75, 0.71, 0.77, 0.78, 0.77, 0.76, 0.77, 0.74, 0.76]
gmean = [0.65, 0.75, 0.72, 0.77, 0.79, 0.78, 0.77, 0.77, 0.74, 0.76]

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)
plt.plot(x, min_recall, marker='o', ms=6, linewidth=2, label='Minority Recall')
plt.plot(x, maj_recall, marker='*', ms=6, linewidth=2, label='Majority Recall')
plt.plot(x, min_fval, marker='s', ms=6, linewidth=2, label='Minority F-val')
plt.plot(x, gmean, marker='+', ms=6, linewidth=2, label='G-mean')


plt.legend(loc='lower right', fontsize=10) # 让图例生效
plt.grid(axis = 'y')
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

plt.xticks(x, names)
plt.margins(0.05)
plt.yticks([0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82])
plt.subplots_adjust(bottom=0.05)

plt.xlabel("Number of base classifiers(The sampling rate of sub-dataset is set to 20%)") #X轴标签
plt.ylabel("Average value of final outputs") #Y轴标签
# plt.title("A simple plot") #标题
plt.tight_layout()

# plt.savefig("../result/number1.png")

plt.show()