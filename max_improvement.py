import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#######################################data##########################################
#AUC,acc,gmean,f1
SVM_NN_MEAN = [0.0024,0.0140,0.0031,0.0356]
SVM_DT_MEAN = [0.0103,0.0200,0.0104,0.0282]
NN_DT_MEAN = [0.0004,0.0081,-0.0041,0.0055]
SVM_DT_NN_MEAN = [0.0103,0.0200,0.0104,0.0559]

A = np.array([0.0024,0.0140,0.0031,0.0356,
              0.0103,0.0200,0.0104,0.0282,
              0.0004,0.0081,-0.0041,0.0055,
              0.0103,0.0200,0.0104,0.0559]).reshape(4,4)
#########################################PLOT#######################################
# Plot
plt.figure()
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(A, yticklabels=['MLP','SVM','DT','SMD'],xticklabels=['AUC','F1-score','Accuracy','G-Mean'],linewidths = 0.05,  cmap='rainbow',center=None, robust=False,annot=True,fmt='.4f')
plt.xlabel('Max Score')

plt.show()