# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:35:23 2017

@author: STP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from dateutil.parser import parse
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.stats import spearmanr

def evalerror(pred, df):
    label = df.get_label().copy()
    score = mean_squared_error(label,pred)*0.5
    return ('mse',score)

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')

drop_feature = ['id']
#drop_feature = ['红细胞压积', '红细胞平均体积', '红细胞体积分布宽度', '血小板平均体积', '血小板体积分布宽度', 'id']

data = pd.read_csv('d_train_20180102.csv', engine='python')
data_A = pd.read_csv('d_test_A_20180102.csv', engine='python')
answer_A = pd.read_csv('d_answer_a_20180128.csv', engine='python', header=None)
answer_A.columns = ['血糖']
test = pd.read_csv('d_test_B_20180128.csv', engine='python')

data_A = pd.concat([data_A, answer_A], axis=1)
data = pd.concat([data, data_A], axis=0).reset_index(drop=True)
feature_list = []
feature_list = list(test.columns)
for i in test.columns:
#    if '乙肝' not in i:
#        feature_list.append(i)
    if i in drop_feature:
        feature_list.remove(i)
test = test[feature_list]
feature_list.append('血糖')
data = data[feature_list]

#data = data[data.id != 4228]

high_label = np.percentile(data['血糖'].values, 90)
low_label = np.percentile(data['血糖'], 10)

data['low_high'] = -1
data['low_high'] = data.apply(lambda x:0 if x['血糖'] < low_label else x['low_high'], axis=1)
data['low_high'] = data.apply(lambda x:1 if x['血糖'] > high_label else x['low_high'], axis=1)

#%%
data['性别'] = data['性别'].replace('男', 1).replace('女', 0).replace('??', np.nan)
'''
for i in data.columns.drop('id').drop('体检日期').drop('血糖'):
    each_feature = data[i]
    y = data['血糖']
    #name = './picture/' + str(i) + '.png'
    
    plt.figure()
    #plt.title(u'性别')
    plt.scatter(y, each_feature)
    plt.legend(prop=zhfont1)
    plt.show()
    plt.savefig(i.replace('*','').replace('%',''))
    plt.close()
'''
data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
#del data['体检日期']

data = data.astype(float)
origin_feature = data.columns

k = 0
#print (len(data.columns))
columns_list = data.columns.drop('血糖').drop('性别').drop('年龄').drop('体检日期').drop('low_high')
#columns_list = data.columns.drop('血糖').drop('性别').drop('年龄').drop('low_high')
columns_list_j = columns_list.copy()
for i in columns_list:
    for j in columns_list_j:
        if i != j:
            name1 = str(i) + '-' + str(j)
            name2 = str(i) + '+' + str(j)
            name3 = str(i) + 'x' + str(j)
            name4 = str(i) + '/' + str(j)
            data[name1] = data[i] - data[j] 
            data[name2] = data[i] + data[j] 
            data[name3] = data[i] * data[j] 
            data[name4] = data[i] / data[j]
            #print ('train:', k)
            k += 1
    columns_list_j = columns_list_j.drop(i)

label = data['血糖']
label_origin = label.copy()
#label = np.log(label)
#mean = label.mean()
#var = np.sqrt(label.var())
#label = (label - mean) / var
del data['血糖']

data_median = data.fillna(data.median())
feature_spearman_1 = pd.DataFrame(np.zeros((len(data.columns), 2), dtype=float))
feature_spearman_1.index = data_median.columns
for i in data.columns:
    feature_spearman_1.loc[i] = spearmanr(data_median[i], label)
drop_spearman = (feature_spearman_1[abs(feature_spearman_1[0]) < 0.05]).index
feature_list = data.columns.drop(drop_spearman).drop('low_high')
origin_feature_new = feature_list.copy()

#%%
test['性别'] = test['性别'].replace('男', 1).replace('女', 0).replace('??', np.nan)
test['体检日期'] = (pd.to_datetime(test['体检日期']) - parse('2017-10-09')).dt.days
#del test['体检日期']

test = test.astype(float)

k = 0
#print (len(test.columns))
columns_list_test = test.columns.drop('性别').drop('年龄').drop('体检日期')
#columns_list_test = test.columns.drop('性别').drop('年龄')
columns_list_test_j = columns_list_test.copy()
for i in columns_list_test:
    for j in columns_list_test_j:
        if i != j:
            name1 = str(i) + '-' + str(j)
            name2 = str(i) + '+' + str(j)
            name3 = str(i) + 'x' + str(j)
            name4 = str(i) + '/' + str(j)
            test[name1] = test[i] - test[j] 
            test[name2] = test[i] + test[j] 
            test[name3] = test[i] * test[j] 
            test[name4] = test[i] / test[j]
            #print ('test:', k)
            k += 1
    columns_list_test_j = columns_list_test_j.drop(i)

params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'lambda': 0.5,
        #'gamma': 5,
        'slient': 1,
        #'alpha': 1,
        #'max_depth': 6,
        #'subsample': 0.7,
        #'colsample_bytree': 0.7,
        'eta': 0.01,
        #'seed': 1,
        'n_jobs': -1}
xgb_train = xgb.DMatrix(data[origin_feature.drop(['血糖', 'low_high'])], label=label)
xgb_test = xgb.DMatrix(test[origin_feature.drop(['血糖', 'low_high'])])
xgb_model_pred3 = xgb.train(params, xgb_train, num_boost_round=600, feval=evalerror, verbose_eval=100)
predict_origin = pd.DataFrame(xgb_model_pred3.predict(xgb_test))
predict_origin.to_csv('predict_origin.csv', index=None, header=None, float_format='%.4f')
#%%
'''
outline_data = data[data['low_high'] != -1]
feature_list = origin_feature.drop('血糖').drop('low_high')
#del data['low_high']
params = {
    'objective': 'binary:logistic',
    #'eval_metric': 'rmse',
    'lambda': 1,
    #'gamma': 5,
    'slient': 1,
    #'alpha': 1,
    #'max_depth': 25,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'eta': 0.01,
    #'seed': 1,
    'n_jobs': -1}
test_preds_classify = np.zeros((test.shape[0], 10))
first = 0
for random in [1, 10]:
    kf = KFold(len(outline_data), n_folds = 5, shuffle=True, random_state=random)
    for i, (train_index, test_index) in enumerate(kf):
        train_feat1 = outline_data[feature_list].iloc[train_index]
        train_feat2 = outline_data[feature_list].iloc[test_index]
        xgb_train1 = xgb.DMatrix(train_feat1, outline_data['low_high'].iloc[train_index])
        xgb_train2 = xgb.DMatrix(train_feat2, outline_data['low_high'].iloc[test_index])
        xgb_train4 = xgb.DMatrix(test[feature_list])
        watchlist3 = [(xgb_train1, 'train'), (xgb_train2, 'test')]
        xgb_model3 = xgb.train(params,
                        xgb_train1,
                        num_boost_round=3000,
                        evals=watchlist3,
                        verbose_eval=100,
                        feval=evalerror,
                        early_stopping_rounds=20)
        if first == 1:
            plus = 5
        else:
            plus = 0
        test_preds_classify[:,i+plus] = xgb_model3.predict(xgb_train4)
    first = 1
test_preds_classify = test_preds_classify.mean(axis=1)
test_preds_classify = pd.DataFrame(test_preds_classify).apply(lambda x:0 if x[0]<0.5 else 1, axis=1)
'''
data_all = pd.concat([data[feature_list], test[feature_list]], axis=0)
#del data_all['low_high']
data_all['性别'] = data_all['性别'].replace('男', 1).replace('女', 0).replace('??', np.nan)
data_all['体检日期'] = (pd.to_datetime(data_all['体检日期']) - parse('2017-10-09')).dt.days

data_all_columns = origin_feature.drop(['性别','年龄','体检日期','血糖','low_high'])
for i in data_all_columns:
    if i in data_all.columns:
        name1 = str(i) + '_log'
        data_all[name1] = np.log(1 + data_all[i])
        name2 = str(i) + '_exp'
        data_all[name2] = np.exp(data_all[i] / data_all[i].max())
'''
columns_list = data_all.columns.drop('性别').drop('年龄').drop('体检日期')
#columns_list = data_all.columns.drop('性别').drop('年龄').drop('low_high')
columns_list_j = columns_list.copy()
for i in columns_list:
    for j in columns_list_j:
        if i != j:
            name1 = str(i) + '-' + str(j)
            name2 = str(i) + '+' + str(j)
            name3 = str(i) + 'x' + str(j)
            name4 = str(i) + '/' + str(j)
            data_all[name1] = data_all[i] - data_all[j] 
            data_all[name2] = data_all[i] + data_all[j] 
            data_all[name3] = data_all[i] * data_all[j] 
            data_all[name4] = data_all[i] / data_all[j]
    columns_list_j = columns_list_j.drop(i)
'''
'''
important_feature = ['*r-谷氨酰基转换酶', '*丙氨酸氨基转换酶', '*天门冬氨酸氨基转换酶','甘油三酯']
important_feature_copy = important_feature.copy()
for i in important_feature:
    for j in important_feature_copy:
        if i != j:
            name1 = str(i) + '-' + str(j)
            name2 = str(i) + '+' + str(j)
            name3 = str(i) + 'x' + str(j)
            name4 = str(i) + '/' + str(j)
#            data_all[name1] = data_all[i] - data_all[j] 
#            data_all[name2] = data_all[i] + data_all[j] 
#            data_all[name3] = data_all[i] * data_all[j] 
            data_all[name4] = data_all[i] / data_all[j]
'''
data_all = data_all.fillna(data_all.median())
#data_all = data_all.fillna(0)
data = data_all.iloc[:len(data)]
test = data_all.iloc[len(data):]

#%%
feature_test = data.columns
for i in data.columns:
    if i in origin_feature:
        feature_test = feature_test.drop(i)
data2 = data[feature_test]
test2 = test[feature_test]
feature_score = pd.DataFrame(np.zeros((len(data.columns), 1), dtype=int))
feature_score.columns = ['importance']
feature_score.index = data.columns
for random in [1, 10, 100]:
#for random in [10]:
    train_l, test_l, train_label_l, test_label_l = train_test_split(data2, label, test_size=0.2, random_state=random)
    
    xgb_train_l = xgb.DMatrix(train_l, label=train_label_l)
    xgb_test_l = xgb.DMatrix(test_l, label=test_label_l)
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'lambda': 0.5,
        #'gamma': 5,
        'slient': 1,
        #'alpha': 1,
        #'max_depth': 6,
        #'subsample': 0.7,
        #'colsample_bytree': 0.7,
        'eta': 0.01,
        #'seed': 1,
        'n_jobs': -1}
    watchlist = [(xgb_train_l, 'train'), (xgb_test_l, 'test')]
    #xgb_model_l = xgb.train(params, xgb_train_l, num_boost_round=5000, evals=watchlist, early_stopping_rounds=20)
    #feature_score_l = xgb_model_l.get_score()
    #circle = xgb_model_l.best_iteration
    
    xgb_train = xgb.DMatrix(data2, label=label)
    xgb_test = xgb.DMatrix(test2)
    watchlist2 = [(xgb_train_l, 'train')]
    xgb_model = xgb.train(params, xgb_train_l, num_boost_round=3000, evals=watchlist, feval=evalerror, verbose_eval=100, early_stopping_rounds=50)
    feature_score_xgb = xgb_model.get_score()

    #predict.to_csv('pred.csv', index=None, header=None)

    feature_score_xgb = pd.DataFrame(feature_score_xgb, index=['importance']).T.sort_values('importance', ascending=False)
    feature_score += feature_score_xgb
xgb_model_pred = xgb.train(params, xgb_train, num_boost_round=600, feval=evalerror, verbose_eval=100)
predict_first = pd.DataFrame(xgb_model_pred.predict(xgb_test))
predict_first.to_csv('predict_first.csv', index=None, header=None, float_format='%.4f')

feature_score = feature_score.sort_values('importance', ascending=False)
feature_xgb_new = feature_score[feature_score.importance>40].index
print ('first cross:', len(feature_xgb_new))
imp_feature = feature_score.iloc[:40].index
columns_list_j = imp_feature.copy()
imp_feature_2 = imp_feature.copy()
for i in imp_feature:
    if i in origin_feature:
        imp_feature = imp_feature.drop(i)
for i in imp_feature_2:
    if i not in origin_feature:
        imp_feature_2 = imp_feature_2.drop(i)
data_all = pd.concat([data, test], axis=0)
double_add = []
for i in imp_feature:
    for j in columns_list_j:
        if i != j:
            name1 = str(i) + '-' + str(j)
            name2 = str(i) + '+' + str(j)
            name3 = str(i) + 'x' + str(j)
            name4 = str(i) + '/' + str(j)
            data_all[name1] = data_all[i] - data_all[j] 
            data_all[name2] = data_all[i] + data_all[j] 
            data_all[name3] = data_all[i] * data_all[j] 
            data_all[name4] = data_all[i] / data_all[j]
            double_add.append(name1)
            double_add.append(name2)
            double_add.append(name3)
            double_add.append(name4)
    columns_list_j = columns_list_j.drop(i)
feature_list = list(set(feature_list) | set(imp_feature) | set(double_add))
data = data_all.iloc[:len(data)]
test = data_all.iloc[len(data):]

xgb_train = xgb.DMatrix(data[double_add], label=label)
xgb_test = xgb.DMatrix(test[double_add])
xgb_model_pred2 = xgb.train(params, xgb_train, num_boost_round=600, feval=evalerror, verbose_eval=100)
predict_second = pd.DataFrame(xgb_model_pred2.predict(xgb_test))
predict_second.to_csv('predict_second.csv', index=None, header=None, float_format='%.4f')

data_median = data.fillna(data.median())
feature_spearman_2 = pd.DataFrame(np.zeros((len(double_add), 2), dtype=float))
feature_spearman_2.index = double_add
for i in double_add:
    feature_spearman_2.loc[i] = spearmanr(data_median[i], label)
drop_spearman2 = (feature_spearman_2[abs(feature_spearman_2[0]) < 0.05]).index
feature_list = data.columns.drop(drop_spearman2)

train_l, test_l, train_label_l, test_label_l = train_test_split(data[feature_list], label, test_size=0.2, random_state=10)
xgb_train_l_double = xgb.DMatrix(train_l, label=train_label_l)
xgb_train_double = xgb.DMatrix(data[feature_list], label=label)
xgb_test_double = xgb.DMatrix(test[feature_list])
watchlist2 = [(xgb_train_l_double, 'train')]
xgb_model = xgb.train(params, xgb_train_double, num_boost_round=400, evals=watchlist2, feval=evalerror, verbose_eval=50)
feature_score2 = pd.DataFrame(xgb_model.get_score(), index=['importance']).T.sort_values('importance', ascending=False)
feature_xgb_new2 = feature_score2[feature_score2.importance>10].index
print ('second cross:', len(feature_xgb_new2))
feature_list = list(set(origin_feature_new) | set(feature_xgb_new) | set(feature_xgb_new2))
#add_feature_list = list(feature_score[feature_score['importance'] > 20]['index'])
#for i in origin_feature:
#    try:
#        add_feature_list = add_feature_list.drop(i)
#    except:
#        pass
#feature_list = list(set(add_feature_list) & set(origin_feature))
#feature_list = data.columns
xgb_train2 = xgb.DMatrix(data[feature_list], label=label)
xgb_test2 = xgb.DMatrix(test[feature_list])
watchlist2 = [(xgb_train2, 'train')]
xgb_model2 = xgb.train(params, xgb_train2, num_boost_round=400, evals=watchlist2, feval=evalerror, verbose_eval=50)
predict = pd.DataFrame(xgb_model2.predict(xgb_test2))
#predict = np.exp(predict)
#predict = predict * var + mean
#predict_mean = predict.mean()
#predict = predict.apply(lambda x:x * np.exp(0.08*(x - predict_mean)), axis=1)

#high_label = np.percentile(predict.values, 90)
#low_label = np.percentile(predict, 10)
#predict['low_high'] = -1
#predict['low_high'] = predict.apply(lambda x:0 if x[0] < low_label else x['low_high'], axis=1)
#predict['low_high'] = predict.apply(lambda x:1 if x[0] > high_label else x['low_high'], axis=1)
#predict = pd.concat([predict, test_preds_classify], axis=1)
#predict.columns = ['output', 'true', 'test']
#predict['final'] = predict.apply(lambda x:x['true'] if x['true'] == x['test'] else -1, axis=1)
#predict['final'] = predict['final'].replace(0, 0.95).replace(1, 1.05).replace(-1, 1)
#predict = predict['output'] * predict['final']
#predict.to_csv('pred.csv', index=None, header=None, float_format='%.4f')

LR_data_all = pd.concat([data[feature_list], test[feature_list]], axis=0)
LR_data_all = LR_data_all.fillna(LR_data_all.median())
data = LR_data_all.iloc[:len(data)]
test = LR_data_all.iloc[len(data):]
#LR = Ridge(alpha=0.1, normalize=True, max_iter=3000)
#LR.fit(LR_train, label)
#pred_LR = pd.DataFrame(LR.predict(LR_test))
#pred_LR = pred_LR * var + mean
#pred_LR.to_csv('LR.csv', index=None, header=None, float_format='%.4f')

data = pd.concat([data, label], axis=1)
train_preds = np.zeros(data[feature_list].shape[0])
train_preds_LR = np.zeros(data[feature_list].shape[0])
test_preds = np.zeros((test[feature_list].shape[0], 5))
kf = KFold(len(data[feature_list]), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i+1))
    train_feat1 = data[feature_list].iloc[train_index]
    train_feat2 = data[feature_list].iloc[test_index]
    xgb_train1 = xgb.DMatrix(train_feat1, data['血糖'].iloc[train_index])
    xgb_train2 = xgb.DMatrix(train_feat2, data['血糖'].iloc[test_index])
    xgb_train3 = xgb.DMatrix(train_feat2)
    xgb_train4 = xgb.DMatrix(test[feature_list])
    watchlist3 = [(xgb_train1, 'train'), (xgb_train2, 'test')]
    xgb_model3 = xgb.train(params,
                    xgb_train1,
                    num_boost_round=3000,
                    evals=watchlist3,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=50)
    #feat_imp = pd.Series(xgb_model3.get_score(), index=['importance']).sort_values(ascending=False)
    train_preds[test_index] += xgb_model3.predict(xgb_train3)
    test_preds[:,i] = xgb_model3.predict(xgb_train4)
    
#    LR = Ridge(alpha=0.1, normalize=True, max_iter=3000)
#    LR.fit(train_feat1, data['血糖'].iloc[train_index])
#    train_preds_LR[test_index] += LR.predict(train_feat2)
train_preds = pd.DataFrame(train_preds)
look = pd.concat([label, train_preds], axis=1)
score = mean_squared_error(label,train_preds)*0.5
print ('offline test:', score)
submission = pd.DataFrame({'pred':test_preds.mean(axis=1)})
#submission = submission.apply(lambda x:1.1 * x['pred'] if x['pred'] > 8 else x['pred'], axis=1)
#submission = np.exp(submission)
#submission = submission * var + mean
#sub_mean = submission.mean()
#submission = submission.apply(lambda x:x * np.exp(0.08*(x - sub_mean)), axis=1)
#submission.to_csv('sub.csv', index=None, header=None, float_format='%.4f')
#xgb_cv_data = xgb.DMatrix(data, label=label)
#xgb_cv = xgb.cv(params, xgb_train, num_boost_round=5000, nfold=10, early_stopping_rounds=20)

#RF = RandomForestRegressor(n_estimators=2000, n_jobs=-1, oob_score=True)
#RF.fit(data[feature_list], label)
#predict_RF = RF.predict(test[feature_list])
#feature_imp = RF.feature_importances_
#feature_imp = pd.concat([pd.DataFrame(feature_imp), pd.DataFrame(data.columns)], axis=1)
#feature_imp.columns = [0, 1]
#pd.DataFrame(predict_RF).to_csv('RF.csv', index=None, header=None, float_format='%.4f')

#train_MLP = data[feature_list].fillna(data[feature_list].median())
#test_MLP = test[feature_list].fillna(test[feature_list].median())
#train_MLP = train_MLP.replace(np.inf, 999999999)
#test_MLP = test_MLP.replace('inf', test_MLP.max())
#MLP = MLPRegressor(hidden_layer_sizes=(20,2), solver='sgd', alpha=0.01, learning_rate_init=0.01,
#                   max_iter=1000, verbose=50, early_stopping=False, validation_fraction=0.2)
#MLP.fit(train_MLP, label)
#pred_MLP = MLP.predict(test_MLP)

#test feature
#base score:0.04180336493904
#params = {
#    'objective': 'reg:linear',
#    #'eval_metric': 'rmse',
#    'lambda': 1,
#    #'gamma': 5,
#    'slient': 1,
#    #'alpha': 1,
#    #'max_depth': 25,
#    'subsample': 0.7,
#    'colsample_bytree': 0.7,
#    'eta': 0.01,
#    'nthread': -1}
#add_feature_list = data.columns
#for i in origin_feature:
#    try:
#        add_feature_list = add_feature_list.drop(i)
#    except:
#        pass
#data = pd.concat([data, label], axis=1)
#test_feature_score = pd.DataFrame(np.zeros((5,len(add_feature_list)), dtype=float))
#test_feature_score.columns = add_feature_list
#k = 0
#origin_feature = origin_feature.drop('low_high')
#for each_feature in add_feature_list:
#    feature_list = list(set([each_feature]) | set(origin_feature))
#    for col, random_state in zip(range(5), [1,10,100,500,1000]):
#        test_preds = np.zeros((len(data), ))
#        kf = KFold(len(data[feature_list]), n_folds = 5, shuffle=True, random_state=random_state)
#        for i, (train_index, test_index) in enumerate(kf):
#            #print('第{}次训练...'.format(i+1))
#            train_feat1 = data[feature_list].iloc[train_index]
#            train_feat2 = data[feature_list].iloc[test_index]
#            xgb_train1 = xgb.DMatrix(train_feat1, data['血糖'].iloc[train_index])
#            xgb_train2 = xgb.DMatrix(train_feat2, data['血糖'].iloc[test_index])
#            xgb_train3 = xgb.DMatrix(train_feat2)
#           # xgb_train4 = xgb.DMatrix(test[feature_list])
#            watchlist3 = [(xgb_train1, 'train'), (xgb_train2, 'test')]
#            xgb_model3 = xgb.train(params,
#                            xgb_train1,
#                            num_boost_round=3000,
#                            evals=watchlist3,
#                            verbose_eval=None,
#                            feval=evalerror,
#                            early_stopping_rounds=20)
#            test_preds[test_index] = xgb_model3.predict(xgb_train3)
#        test_score = mean_squared_error(label,test_preds)*0.5
#        print (test_score)
#        #test_feature_score[each_feature][col] = test_score
#    print (each_feature, 'round:', (k+1), 'all:', (len(add_feature_list)))
#    k += 1
