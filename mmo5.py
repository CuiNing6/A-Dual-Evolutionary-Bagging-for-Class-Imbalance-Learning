import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate
#######################################data##########################################
#sm
sm_16 = [0.8095, 0.7891, 0.7823, 0.7891, 0.7959, 0.7823, 0.8027, 0.7823, 0.7823, 0.7891]
sm_17 = [0.8027, 0.7755, 0.8027, 0.7687, 0.8027, 0.7755, 0.7755, 0.7891, 0.8163, 0.7891]
sm_18 = [0.7959, 0.7891, 0.8027, 0.7823, 0.7891, 0.7891, 0.7891, 0.7823, 0.7755, 0.7891]
sm_19 = [0.8027, 0.7755, 0.7687, 0.7755, 0.7959, 0.7823, 0.7687, 0.7823, 0.7959, 0.7823]
# sm_20 = [0.8362, 0.8318, 0.8362, 0.8274, 0.8362, 0.8318, 0.8318, 0.8362, 0.8407, 0.8362]
#########################################PLOT#######################################
x = range(len(sm_16))
xnew =np.arange(0,9,0.1)
sm_16_func = interpolate.interp1d(x,sm_16,kind='cubic')
sm_c_16 = sm_16_func(xnew)
sm_17_func = interpolate.interp1d(x,sm_17,kind='cubic')
sm_c_17 = sm_17_func(xnew)
sm_18_func = interpolate.interp1d(x,sm_18,kind='cubic')
sm_c_18 = sm_18_func(xnew)
sm_19_func = interpolate.interp1d(x,sm_19,kind='cubic')
sm_c_19 = sm_19_func(xnew)
# sm_20_func = interpolate.interp1d(x,sm_20,kind='cubic')
# sm_c_20 = sm_20_func(xnew)
# Plot
plt.figure()
plt.plot(xnew, sm_c_16,  ms=6, label='Randomly select 5 base classifiers',linewidth=2)
plt.plot(xnew, sm_c_17,  ms=6, label='Randomly select 10 base classifiers',linestyle='-.',linewidth=2)
plt.plot(xnew, sm_c_18,  ms=6, label='Randomly select 15 base classifiers',linestyle=':',linewidth=2)
plt.plot(xnew, sm_c_19,  ms=6, label='Randomly select 20 base classifiers',linestyle='--',linewidth=2)
# plt.plot(xnew, sm_c_20,  ms=6, label='25',linestyle='-',linewidth=2)

plt.legend(loc='upper left', fontsize=10) # 让图例生效
plt.yticks([0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83])
plt.ylabel("Accuracy") #Y轴标签
plt.xlabel('Number of times')
#number of times
plt.show()