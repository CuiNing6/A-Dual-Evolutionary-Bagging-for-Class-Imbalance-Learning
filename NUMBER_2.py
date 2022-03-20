import matplotlib.pyplot as plt 
import matplotlib  
# matplotlib.use('Agg')

names = ['1', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']

x = range(len(names))

Recall = [0.65, 0.76, 0.77, 0.76, 0.77, 0.79, 0.80, 0.76, 0.79, 0.79, 0.78, 0.80, 0.78]
Fvalue = [0.68, 0.77, 0.77, 0.77, 0.78, 0.79, 0.80, 0.78, 0.81, 0.80, 0.78, 0.80, 0.79]
Gmean = [0.69, 0.77, 0.78, 0.77, 0.78, 0.80, 0.80, 0.78, 0.81, 0.80, 0.79, 0.79, 0.79]

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11) # 限定横轴的范围
#pl.ylim(-1, 110) # 限定纵轴的范围
# plt.rcParams['figure.figsize'] = (6.0, 4.0)

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
plt.yticks([0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82])
plt.subplots_adjust(bottom=0.05)

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}
plt.xlabel("Number of base classifiers(The sampling rate of sub-dataset is set to 20%)",font2) #X轴标签
plt.ylabel("Average value of final outputs",font2) #Y轴标签
# plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.xticks(fontproperties = 'Times New Roman', size = 14)

# plt.title("A simple plot") #标题
plt.tight_layout()

# plt.savefig("../result/number2.png")

plt.show()