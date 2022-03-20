import numpy as np
import pandas as pd
import random
import time
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.metrics import geometric_mean_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
import warnings
warnings.filterwarnings('ignore')
############################import data#########################################
#import data
data = pd.read_csv('abalone9-18.txt',sep=',',header = None)
print('数据集：abalone9-18')
# print(data.head())

data.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y']

data['x1'] = data['x1'].replace("F",0)
data['x1'] = data['x1'].replace("M",1)
data['x1'] = data['x1'].replace("I",2)
data['y'] = data['y'].replace(' negative',0)
data['y'] = data['y'].replace(' positive',1)

X = data.drop(['y'],axis=1)
y = data.y
##################################resampling#######################################
#borderline1
X_resampled_bor1, y_resampled_bor1 = SMOTE(kind='borderline1').fit_sample(X, y)
#borderline2
X_resampled_bor2, y_resampled_bor2 = SMOTE(kind='borderline2').fit_sample(X, y)
#svm
X_resampled_svm, y_resampled_svm = SMOTE(kind='svm').fit_sample(X, y)
#RUS
ros = RandomOverSampler()
X_resampled_ros, y_resampled_ros = ros.fit_sample(X, y)
#smote
X_resampled_smote, y_resampled_smote = SMOTE().fit_sample(X, y)
#################################selective_classifier########################################
def selective_classifier(X_train,y_train,X_test):
    #DT
    pipeline_dt = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy'))])
    parameters_dt = {'clf__max_depth': (4,5,6,7,8,9,10,11,12,13),'clf__min_samples_split': (0.3,0.4,0.5,0.6,0.7,0.8),'clf__min_samples_leaf': (3,4,5,6,7)}
    grid_search_dt = GridSearchCV(pipeline_dt, parameters_dt, cv=10, n_jobs=-1, scoring='f1')
    grid_search_dt.fit(X_train, y_train)
    dt_result = grid_search_dt.best_score_
    # print(dt_result)
    # print('dt最佳效果：%0.3f' % grid_search_dt.best_score_)
    # print('dt最优参数：')
    # best_parameters_dt = grid_search_dt.best_estimator_.get_params()
    # for param_name_dt in sorted(parameters_dt.keys()):
    #     print('\t%s: %r' % (param_name_dt, best_parameters_dt[param_name_dt]))
    #SVM
    parameters_svm = {'kernel':['rbf'], 'gamma':np.logspace(0, 5, num=5, base=2.0),'C':np.logspace(5, 10, num=5, base=2.0)}
    grid_search_svm = GridSearchCV(svm.SVC(probability=True), parameters_svm, cv=10, n_jobs=-1, scoring='f1')
    grid_search_svm.fit(X_train, y_train)
    svm_result = grid_search_svm.best_score_
    # print(svm_result)
    # print('svm最佳效果：%0.3f' % grid_search_svm.best_score_)
    # print('svm最优参数：')
    # best_parameters_svm = grid_search_svm.best_estimator_.get_params()
    # for param_name_svm in sorted(parameters_svm.keys()):
    #     print('\t%s: %r' % (param_name_svm, best_parameters_svm[param_name_svm]))
    #NN
    parameters_nn = {'hidden_layer_sizes':[50,100,200,300,350], 'activation':['relu']}
    grid_search_nn = GridSearchCV(MLPClassifier(), parameters_nn, cv=10, n_jobs=-1, scoring='f1')
    grid_search_nn.fit(X_train, y_train)
    nn_result = grid_search_nn.best_score_
    # print(nn_result)
    # print('nn最佳效果：%0.3f' % grid_search_nn.best_score_)
    # print('nn最优参数：')
    # best_parameters_nn = grid_search_nn.best_estimator_.get_params()
    # for param_name_nn in sorted(parameters_nn.keys()):
    #     print('\t%s: %r' % (param_name_nn, best_parameters_nn[param_name_nn]))

    if svm_result>nn_result and svm_result>dt_result:
        y_pred = grid_search_svm.predict_proba(X_test)
        print('select svm')
    elif nn_result>svm_result and nn_result>dt_result:
        y_pred = grid_search_nn.predict_proba(X_test)
        print('select nn')
    else:
        y_pred = grid_search_dt.predict_proba(X_test)
        print('select dt')

    return y_pred[:,0]
#################################bootstrap##################################################
def bootstrap(X,y,ratio):
    label = np.unique(y)
    x1 = pd.DataFrame(X[y==label[0]])
    x2 = pd.DataFrame(X[y==label[1]])

    sub_num = int(len(x1)*ratio)

    x1_sub = x1.sample(n=sub_num,replace=True)
    y1_sub = np.zeros((1,sub_num))

    x2_sub = x2.sample(n=sub_num,replace=True)
    y2_sub = np.ones((1,sub_num))

    X_sub = x1_sub.append(x2_sub)

    y_sub = np.append(y1_sub,y2_sub)

    return X_sub, y_sub
#############################################GA#############################################
def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)

    return pop[1:]
def calobjValue(pop,result,y_test):
    obj_value = []
    for i in range(len(pop)):
        x = pop[i]
        x1 = np.expand_dims(np.array(x), axis=0)
        x2 = np.mat(x1)*np.mat(result)/result.shape[0]
        vote = np.int64(x2<0.5)
        vote1 = np.ravel(vote)

        obj_value.append(geometric_mean_score(y_test,vote1))
    return obj_value
def calfitValue(obj_value):
    fit_value = []
    c_min = 0
    for i in range(len(obj_value)):
        if(obj_value[i] + c_min > 0):
            temp = c_min + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value
def best(pop, fit_value):
    px = len(pop)
    best_individual = []
    best_fit = fit_value[0]
    for i in range(1, px):
        # print(i)
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total
def cumsum(fit_value):
    for i in range(len(fit_value)-2, -1, -1):
        t = 0
        j = 0
        while(j <= i):
            t += fit_value[j]
            j += 1
            fit_value[i] = t
        fit_value[len(fit_value)-1] = 1
def selection(pop, fit_value):
    newfit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    if total_fit == 0:
        total_fit = 1
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)
    # 计算累计概率
    cumsum(newfit_value)
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    # 转轮盘选择法
    while newin < pop_len:
        if(ms[newin] < newfit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop
def crossover(pop, pc):
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if(random.random() < pc):
            cpoint = random.randint(0,len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i+1][cpoint:len(pop[i])])
            temp2.extend(pop[i+1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i+1] = temp2
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if(random.random() < pm):
            mpoint = random.randint(0, py-1)
            if(pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1
##########################################main################################################
results_bor1_gmean = [];results_bor1_acc = [];results_bor1_auc = [];
results_bor2_gmean = [];results_bor2_acc = [];results_bor2_auc = [];
results_sm_gmean = [];results_sm_acc = [];results_sm_auc = [];
results_ros_gmean = [];results_ros_acc = [];results_ros_auc = [];
results_svm_gmean = [];results_svm_acc = [];results_svm_auc = [];

for k in range(5):
    #split data
    X_train_bor1, X_test_bor1, y_train_bor1, y_test_bor1 = train_test_split(X_resampled_bor1, y_resampled_bor1, test_size=0.2)
    X_train_bor2, X_test_bor2, y_train_bor2, y_test_bor2 = train_test_split(X_resampled_bor2, y_resampled_bor2, test_size=0.2)
    X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_resampled_smote, y_resampled_smote, test_size=0.2)
    X_train_ros, X_test_ros, y_train_ros, y_test_ros = train_test_split(X_resampled_ros, y_resampled_ros, test_size=0.2)
    X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_resampled_svm, y_resampled_svm, test_size=0.2)
    #sampling data and classification
    subdata_num = 10
    ratio = 0.4
    result_bor1 = [];result_bor2 = [];result_sm = [];result_ros = [];result_svm = []
    print("================================================================================================")
    print("子分类器训练开始")
    print("================================================================================================")
    start = time.time()
    for i in range(subdata_num):
        X_sub_bor1, y_sub_bor1 = bootstrap(X_train_bor1,y_train_bor1,ratio)
        X_sub_bor2, y_sub_bor2 = bootstrap(X_train_bor2,y_train_bor2,ratio)
        X_sub_sm, y_sub_sm = bootstrap(X_train_sm,y_train_sm,ratio)
        X_sub_ros, y_sub_ros = bootstrap(X_train_ros,y_train_ros,ratio)
        X_sub_svm, y_sub_svm = bootstrap(X_train_svm,y_train_svm,ratio)

        pre_bor1 = selective_classifier(X_sub_bor1,y_sub_bor1,X_test_bor1)
        pre_bor2 = selective_classifier(X_sub_bor2,y_sub_bor2,X_test_bor2)
        pre_sm = selective_classifier(X_sub_sm,y_sub_sm,X_test_sm)
        pre_ros = selective_classifier(X_sub_ros,y_sub_ros,X_test_ros)
        pre_svm = selective_classifier(X_sub_svm,y_sub_svm,X_test_svm)

        result_bor1.append(pre_bor1)
        result_bor2.append(pre_bor2)
        result_sm.append(pre_sm)
        result_ros.append(pre_ros)
        result_svm.append(pre_svm)
    end = time.time()
    print("训练时间:%.2f秒"%(end-start))
    results_bor1 = np.array(result_bor1)
    results_bor2 = np.array(result_bor2)
    results_sm = np.array(result_sm)
    results_ros = np.array(result_ros)
    results_svm = np.array(result_svm)
    #Genetic algorithm
    pm = 0.08
    pc = 0.8
    bor1_results = [[]];bor2_results = [[]];sm_results = [[]];ros_results = [[]];svm_results = [[]];
    pop_size = 50
    chrom_length = 10
    iters = 100
    pop = geneEncoding(pop_size, chrom_length)
    # print(pop)
    print("================================================================================================")
    print("优化开始")
    print("================================================================================================")
    start1 = time.time()
    for j in range(iters):
        obj_value = calobjValue(pop, results_bor1, y_test_bor1)
        fit_value = calfitValue(obj_value)
        best_individual, best_fit = best(pop, fit_value)
        bor1_results.append([best_fit, best_individual])
        selection(pop, fit_value)
        crossover(pop, pc)
        mutation(pop, pm)
    end1 = time.time()
    print("优化时间:%.2f秒"%(end1-start1))

    start2 = time.time()
    for jj in range(iters):
        obj_value = calobjValue(pop, results_bor2, y_test_bor2)
        fit_value = calfitValue(obj_value)
        best_individual, best_fit = best(pop, fit_value)
        bor2_results.append([best_fit, best_individual])
        selection(pop, fit_value)
        crossover(pop, pc)
        mutation(pop, pm)
    end2 = time.time()
    print("优化时间:%.2f秒"%(end2-start2))

    start3 = time.time()
    for jjj in range(iters):
        obj_value = calobjValue(pop, results_sm, y_test_sm)
        fit_value = calfitValue(obj_value)
        best_individual, best_fit = best(pop, fit_value)
        sm_results.append([best_fit, best_individual])
        selection(pop, fit_value)
        crossover(pop, pc)
        mutation(pop, pm)
    end3 = time.time()
    print("优化时间:%.2f秒"%(end3-start3))

    start4 = time.time()
    for jjjj in range(iters):
        obj_value = calobjValue(pop, results_ros, y_test_ros)
        fit_value = calfitValue(obj_value)
        best_individual, best_fit = best(pop, fit_value)
        ros_results.append([best_fit, best_individual])
        selection(pop, fit_value)
        crossover(pop, pc)
        mutation(pop, pm)
    end4 = time.time()
    print("优化时间:%.2f秒"%(end4-start4))

    start5 = time.time()
    for jjjjj in range(iters):
        obj_value = calobjValue(pop, results_svm, y_test_svm)
        fit_value = calfitValue(obj_value)
        best_individual, best_fit = best(pop, fit_value)
        svm_results.append([best_fit, best_individual])
        selection(pop, fit_value)
        crossover(pop, pc)
        mutation(pop, pm)
    end5 = time.time()
    print("优化时间:%.2f秒"%(end5-start5))

    bor1_results = bor1_results[1:]
    bor2_results = bor2_results[1:]
    sm_results = sm_results[1:]
    ros_results = ros_results[1:]
    svm_results = svm_results[1:]
    bor1_results.sort()
    bor2_results.sort()
    sm_results.sort()
    ros_results.sort()
    svm_results.sort()

    ind_bor1 = bor1_results[-1][1]
    ind_bor2 = bor2_results[-1][1]
    ind_sm = sm_results[-1][1]
    ind_ros = ros_results[-1][1]
    ind_svm = svm_results[-1][1]

    ens_bor1 = np.mat(ind_bor1)*np.mat(results_bor1)/results_bor1.shape[0]
    ens_bor2 = np.mat(ind_bor2)*np.mat(results_bor2)/results_bor2.shape[0]
    ens_sm = np.mat(ind_sm)*np.mat(results_sm)/results_sm.shape[0]
    ens_ros = np.mat(ind_ros)*np.mat(results_ros)/results_ros.shape[0]
    ens_svm = np.mat(ind_svm)*np.mat(results_svm)/results_svm.shape[0]

    vote_bor1 = np.ravel(np.int64(ens_bor1<0.5))
    vote_bor2 = np.ravel(np.int64(ens_bor2<0.5))
    vote_sm = np.ravel(np.int64(ens_sm<0.5))
    vote_ros = np.ravel(np.int64(ens_ros<0.5))
    vote_svm = np.ravel(np.int64(ens_svm<0.5))

    results_bor1_gmean.append(f1_score(y_test_bor1,vote_bor1))
    results_bor1_acc.append(accuracy_score(y_test_bor1,vote_bor1))
    results_bor1_auc.append(roc_auc_score(y_test_bor1, vote_bor1))

    results_bor2_gmean.append(f1_score(y_test_bor2,vote_bor2))
    results_bor2_acc.append(accuracy_score(y_test_bor2,vote_bor2))
    results_bor2_auc.append(roc_auc_score(y_test_bor2, vote_bor2))

    results_sm_gmean.append(f1_score(y_test_sm,vote_sm))
    results_sm_acc.append(accuracy_score(y_test_sm,vote_sm))
    results_sm_auc.append(roc_auc_score(y_test_sm, vote_sm))

    results_ros_gmean.append(f1_score(y_test_ros,vote_ros))
    results_ros_acc.append(accuracy_score(y_test_ros,vote_ros))
    results_ros_auc.append(roc_auc_score(y_test_ros, vote_ros))

    results_svm_gmean.append(f1_score(y_test_svm,vote_svm))
    results_svm_acc.append(accuracy_score(y_test_svm,vote_svm))
    results_svm_auc.append(roc_auc_score(y_test_svm, vote_svm))
######################################评估分类结果#######################################
print("================================================================================================")
print("评估分类结果")
print("================================================================================================")
nnn = 5
print("Bor1 G_Mean_score: %0.4f,%0.4f" % (np.sum(results_bor1_gmean)/nnn,np.max(results_bor1_gmean)-np.sum(results_bor1_gmean)/nnn))
print("Bor1 Accuracy_score: %0.4f,%0.4f" % (np.sum(results_bor1_acc)/nnn,np.max(results_bor1_acc)-np.sum(results_bor1_acc)/nnn))
print("Bor1 Roc_auc_score: %0.4f,%0.4f" % (np.sum(results_bor1_auc)/nnn,np.max(results_bor1_auc)-np.sum(results_bor1_auc)/nnn))
print("================================================================================================")
print("Bor2 G_Mean_score: %0.4f,%0.4f" % (np.sum(results_bor2_gmean)/nnn,np.max(results_bor2_gmean)-np.sum(results_bor2_gmean)/nnn))
print("Bor2 Accuracy_score: %0.4f,%0.4f" % (np.sum(results_bor2_acc)/nnn,np.max(results_bor2_acc)-np.sum(results_bor2_acc)/nnn))
print("Bor2 Roc_auc_score: %0.4f,%0.4f" % (np.sum(results_bor2_auc)/nnn,np.max(results_bor2_auc)-np.sum(results_bor2_auc)/nnn))
print("================================================================================================")
print("sm G_Mean_score: %0.4f,%0.4f" % (np.sum(results_sm_gmean)/nnn,np.max(results_sm_gmean)-np.sum(results_sm_gmean)/nnn))
print("sm Accuracy_score: %0.4f,%0.4f" % (np.sum(results_sm_acc)/nnn,np.max(results_sm_acc)-np.sum(results_sm_acc)/nnn))
print("sm Roc_auc_score: %0.4f,%0.4f" % (np.sum(results_bor1_auc)/nnn,np.max(results_bor1_auc)-np.sum(results_bor1_auc)/nnn))
print("================================================================================================")
print("ros G_Mean_score: %0.4f,%0.4f" % (np.sum(results_ros_gmean)/nnn,np.max(results_ros_gmean)-np.sum(results_ros_gmean)/nnn))
print("ros Accuracy_score: %0.4f,%0.4f" % (np.sum(results_ros_acc)/nnn,np.max(results_ros_acc)-np.sum(results_ros_acc)/nnn))
print("ros Roc_auc_score: %0.4f,%0.4f" % (np.sum(results_ros_auc)/nnn,np.max(results_ros_auc)-np.sum(results_ros_auc)/nnn))
print("================================================================================================")
print("svm G_Mean_score: %0.4f,%0.4f" % (np.sum(results_svm_gmean)/nnn,np.max(results_svm_gmean)-np.sum(results_svm_gmean)/nnn))
print("svm Accuracy_score: %0.4f,%0.4f" % (np.sum(results_svm_acc)/nnn,np.max(results_svm_acc)-np.sum(results_svm_acc)/nnn))
print("svm Roc_auc_score: %0.4f,%0.4f" % (np.sum(results_svm_auc)/nnn,np.max(results_svm_auc)-np.sum(results_svm_auc)/nnn))