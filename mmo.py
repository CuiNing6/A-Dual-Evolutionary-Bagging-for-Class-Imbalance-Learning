import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import interpolate
#######################################data##########################################
#sm
sm_16 = [0.8152, 0.8369, 0.8550, 0.8188, 0.8369, 0.8442, 0.8478, 0.8405, 0.8188, 0.8224]
sm_17 = [0.8514, 0.8224, 0.8405, 0.8224, 0.8260, 0.8478, 0.8405, 0.8478, 0.8478, 0.8188]
sm_18 = [0.8405, 0.8405, 0.8442, 0.8550, 0.8115, 0.8478, 0.8478, 0.8514, 0.8478, 0.8478]
sm_19 = [0.8188, 0.8188, 0.8297, 0.8478, 0.8297, 0.8514, 0.8369, 0.8224, 0.8623, 0.8369]
sm_20 = [0.8514, 0.8224, 0.8478, 0.8297, 0.8442, 0.8478, 0.8514, 0.8405, 0.8260, 0.8478]
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
sm_20_func = interpolate.interp1d(x,sm_20,kind='cubic')
sm_c_20 = sm_20_func(xnew)
# Plot
plt.figure()
plt.plot(xnew, sm_c_16,  ms=6, label='Randomly select 5 base classifiers',linewidth=2)
plt.plot(xnew, sm_c_17,  ms=6, label='Randomly select 10 base classifiers',linestyle='-.',linewidth=2)
plt.plot(xnew, sm_c_18,  ms=6, label='Randomly select 15 base classifiers',linestyle=':',linewidth=2)
plt.plot(xnew, sm_c_19,  ms=6, label='Randomly select 20 base classifiers',linestyle='--',linewidth=2)
# plt.plot(xnew, sm_c_20,  ms=6, label='25',linestyle='-',linewidth=2)

plt.legend(loc='best', fontsize=10) # 让图例生效
plt.yticks([0.80, 0.82, 0.84, 0.86, 0.88])
plt.ylabel("Accuracy") #Y轴标签
plt.xlabel('Number of times')
#number of times
plt.show()