import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#######################################data##########################################
#AUC,acc,gmean,f1
SVM_NN_MEAN = [0.1108,0.0291,0.0382,0.0603]
SVM_DT_MEAN = [0.1098,0.0331,0.0437,0.0772]
NN_DT_MEAN = [0.1089,0.0250,0.0327,0.0451]
SVM_DT_NN_MEAN = [0.1128,0.0341,0.0393,0.0961]

A = np.array([0.1108,0.0291,0.0382,0.0603,
              0.1098,0.0331,0.0437,0.0772,
              0.1089,0.0250,0.0327,0.0451,
              0.1128,0.0341,0.0393,0.0961]).reshape(4,4)
#########################################PLOT#######################################
# Plot
plt.figure()
cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(A, yticklabels=['MLP','SVM','DT','SMD'],xticklabels=['AUC','F1-score','Accuracy','G-Mean'],linewidths = 0.05,  cmap='rainbow',center=None, robust=False,annot=True,fmt='.4f')
plt.xlabel('Mean Score')

plt.show()