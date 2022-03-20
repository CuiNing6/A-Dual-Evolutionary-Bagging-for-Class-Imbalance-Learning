import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate
#######################################data##########################################
#sm
sm_16 = [0.5894,0.8657,0.3954,0.4275,0.9929,0.8657,0.4815,0.8657,0.5596,0.0872,0.6989,0.2086,0.6404,
0.8657,0.6404,0.6989,0.8869,0.9519,0.9929,0.2086,0.1287,0.0872,0.8657,0.5596,0.8691,0.8885,0.6192,
0.6404,0.8657,0.0872,0.2655,0.4275,0.9929,0.0872,0.0872,0.6404,0.8657,0.2380,0.2655,0.2655,0.4901,
0.9519,0.2655,0.8657,0.6404,0.2655,0.6989,0.2086,0.8933,0.8691,0.2086,0.9929,0.6989,0.4275,0.2086]
#########################################PLOT#######################################
x = range(len(sm_16))
xnew =np.arange(0,54,1)
sm_16_func = interpolate.interp1d(x,sm_16,kind='cubic')
sm_c_16 = sm_16_func(xnew)
xx = [i for i in range(len(sm_16))]
# sm_17_func = interpolate.interp1d(x,sm_17,kind='cubic')
# sm_c_17 = sm_17_func(xnew)
# sm_18_func = interpolate.interp1d(x,sm_18,kind='cubic')
# sm_c_18 = sm_18_func(xnew)
# sm_19_func = interpolate.interp1d(x,sm_19,kind='cubic')
# sm_c_19 = sm_19_func(xnew)
# sm_20_func = interpolate.interp1d(x,sm_20,kind='cubic')
# sm_c_20 = sm_20_func(xnew)
# Plot

plt.figure()
plt.plot(xnew, sm_c_16,  ms=6, linewidth=2)
# plt.hlines(0.9657, 0, 54, color="red")#横线
plt.scatter(4, 0.9929, color='r', marker='*',linewidths=5)
plt.scatter(18, 0.9929, color='r', marker='*',linewidths=5)
plt.scatter(32, 0.9929, color='r', marker='*',linewidths=5)
plt.scatter(41, 0.9519, color='r', marker='*',linewidths=5)
plt.scatter(51, 0.9929, color='r', marker='*',linewidths=5)
# plt.plot(xnew, sm_c_17,  ms=6, label='Randomly select 10 base classifiers',linestyle='-.',linewidth=2)
# plt.plot(xnew, sm_c_18,  ms=6, label='Randomly select 15 base classifiers',linestyle=':',linewidth=2)
# plt.plot(xnew, sm_c_19,  ms=6, label='Randomly select 20 base classifiers',linestyle='--',linewidth=2)
# plt.plot(xnew, sm_c_20,  ms=6, label='25',linestyle='-',linewidth=2)

plt.legend(loc='best', fontsize=10) # 让图例生效
plt.yticks([0, 0.20, 0.4, 0.6, 0.8, 1, 1.2])
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55])
plt.xlim([0,55])
# plt.ylabel("AUC") #Y轴标签
# plt.xlabel('Number of times')
#number of times
plt.show()