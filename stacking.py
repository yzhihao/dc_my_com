import numpy as np
import pandas as pd
import second_statge.evaluate as evaluate
import lightgbm as lgb
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                                  GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import xgboost


import datetime
operation_trn = pd.read_csv('..\operation_TRAIN_new.csv')
operation_test = pd.read_csv('..\operation_round1_new.csv')
transaction_trn = pd.read_csv('..\\transaction_TRAIN_new.csv')
transaction_test = pd.read_csv('..\\transaction_round1_new.csv')
tag_trn = pd.read_csv('..\\tag_TRAIN_new.csv')

# ===================================处理操作详情=====================================#

operation_trn = pd.merge(operation_trn, tag_trn, how='left', on='UID')
operation_test['Tag'] = -1
df = pd.concat([operation_trn, operation_test])
df['hour'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
del df['time']
label_feature = ['os', 'mode', 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3',
                 'ip1', 'ip2', 'mac1', 'mac2', 'wifi', 'ip1_sub', 'ip2_sub']


for each in label_feature:
    df[each] = pd.factorize(df[each])[0]


def split_version(v, n):
    if pd.isna(v):
        return np.nan
    return int(v.split('.')[n-1])


df['version_1'] = df['version'].apply(lambda v: split_version(v, 1))
df['version_2'] = df['version'].apply(lambda v: split_version(v, 2))
df['version_3'] = df['version'].apply(lambda v: split_version(v, 3))
del df['version']


# df['device2'] 可对型号细分

geo_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
            '9': 9, 'b': 10, 'c':11, 'd':12, 'e':13, 'f':14, 'g':15, 'h':16,  'j':17,
            'k':18, 'm':19, 'n':20, 'p':21, 'q':22, 'r':23, 's':24, 't':25, 'u':26,
            'v':27, 'w':28, 'x':29, 'y':30, 'z':31,}

def split_geo(g, n):
    if pd.isna(g):
        return np.nan
    return geo_dict[g[n-1]]


for i in range(1, 5):
    df['geo_'+str(i)] = df['geo_code'].apply(lambda g: split_geo(g, i))

del df['geo_code']

pd.DataFrame(df).info()

def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mode()[0])#还有众数.mode()；中位数.median()
        else:
            pass
    return data

df=mis_impute(df)

# ===================================训练操作数据=====================================#
xx_auc = []
xx_submit = []
N = 3
#skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)

#train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)
train_x = np.array(df[df.Tag != -1].drop(['Tag', 'UID'], axis=1))
train_y = np.array(df[df.Tag != -1]['Tag'])
test_x = np.array(df[df.Tag == -1].drop(['Tag', 'UID'], axis=1))



class XGBClassifier():
    def __init__(self):
        """set parameters"""
        self.num_rounds = 1000
        self.xgb=xgboost
        self.early_stopping_rounds = 15
        self.params = {
            'objective': 'binary:logitraw',
            #'num_class':2,
            'eta': 0.1,
            'max_depth': 8,
            'tree_method':'gpu_hist',
            #'eval_metric': 'mlogloss',
            'seed': 0,
            'silent': 0
        }

    def fit(self, x_train, y_train,x_val, y_val):
        print('train with xgb model')
        xgbtrain = self.xgb.DMatrix(x_train, y_train)
        xgbval = self.xgb.DMatrix(x_val, y_val)
        watchlist = [(xgbtrain, 'train'), (xgbval, 'val')]
        model=self.xgb.train(self.params,
                          xgbtrain,
                          self.num_rounds,
                           watchlist,
                            early_stopping_rounds = self.early_stopping_rounds)
        return model

    def predict(self,model, x_test,flag='class'):
        print('test with xgb model')
        xgbtest = self.xgb.DMatrix(x_test)
        return model.predict(xgbtest)


import lightgbm
class LGBClassifier():
    def __init__(self,params):
        self.num_boost_round = 2000
        self.lgb=lightgbm
        self.early_stopping_rounds = 15
        self.params=params


    def fit(self, x_train, y_train,x_val,y_val):
        print('train with lgb model')
        lgbtrain = self.lgb.Dataset(x_train, y_train)
        lgbval = self.lgb.Dataset(x_val, y_val)
        model=self.lgb.train(self.params,
                          lgbtrain,
                          valid_sets=lgbval,
                          verbose_eval=self.num_boost_round,
                          num_boost_round=self.num_boost_round,
                         early_stopping_rounds = self.early_stopping_rounds)
        return model
    def predict(self,model, x_test,flag='class'):
        print('test with lgb model')
        if flag== 'class':   #TODO
            return np.argmax(model.predict(x_test, num_iteration=model.best_iteration),axis=1)
        else:
            return model.predict(x_test, num_iteration=model.best_iteration)



def get_stage1(clf, x_train, y_train, x_test, n_folds=5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]
        if clf.__class__.__name__=='XGBClassifier' or clf.__class__.__name__=='LGBClassifier' :
            model=clf.fit(x_tra, y_tra,x_tst, y_tst)
            second_level_train_set[test_index] = clf.predict(model,x_tst,flag='prod')
            #print("eval_stage1:%f" % accuracy_score(y_tst, clf.predict(model,x_tst)))
            test_nfolds_sets[:, i] = clf.predict(model,x_test,flag='prod')
        else:
            clf.fit(x_tra, y_tra)
            second_level_train_set[test_index] = clf.predict_proba(x_tst)[:,1]
            print("eval_stage1:%f" % accuracy_score(y_tst, clf.predict(x_tst)))
            test_nfolds_sets[:, i] = clf.predict_proba(x_test)[:,1]


    #second_level_test_set[:] = np.array(pd.DataFrame(test_nfolds_sets).mode(axis=1).values).T
    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


def get_stage2(stage2_model,train_sets,train_y,test_sets,n_folds=5):
    xx_submit=[]

    meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
    meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

    kf=KFold(n_splits=n_folds)
    #train_num, test_num = meta_train.shape[0], meta_test.shape[0]

    for i,(train_index, test_index) in enumerate(kf.split(meta_train)):
        x_tra, y_tra = meta_train[train_index], train_y[train_index]
        x_tst, y_tst =  meta_train[test_index], train_y[test_index]
        if stage2_model.__class__.__name__=='XGBClassifier' or stage2_model.__class__.__name__=='LGBClassifier' :
            #model=clf.fit(x_tra,y_tra,x_tst,y_tst)
            model = stage2_model.fit(x_tra, y_tra, x_tst, y_tst)
            # xx_auc.append(gbm.best_score['valid_0']['auc'])
            xx_submit.append(stage2_model.predict(model,meta_test))
            print("eval:%f" % evaluate.tpr_weight_funtion(y_tst, stage2_model.predict(model,x_tst)))
        else:
            stage2_model.fit(x_tra, y_tra)
            #xx_auc.append(gbm.best_score['valid_0']['auc'])
            xx_submit.append(stage2_model.predict_proba(meta_test)[:,1])
            print("eval:%f" % evaluate.tpr_weight_funtion(y_tst, stage2_model.predict(x_tst)))

    s = 0
    for each in xx_submit:
        s += each
    return list(s / n_folds)


    '''
    # 使用决策树作为我们的次级分类器
    #from sklearn.tree import DecisionTreeClassifier
    dt_model = stage2_model
    dt_model.fit(meta_train, train_y)
    df_predict = dt_model.predict(meta_test)
    '''

   # return df_predict


if __name__=='__main__':
    #我们这里使用5个分类算法，为了体现stacking的思想，就不加参数了
    params = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'multiclass',
        # 'application':'num_class',
        'metric': 'multi_logloss',
        #'device': 'gpu',
        'num_leaves': 31,
        'learning_rate': 0.02,
        'num_class': 2,
        # 'scale_pos_weight': 1.5,
        'feature_fraction': 0.5,
        'bagging_fraction': 1,
        'bagging_freq': 5,
        'max_bin': 300,
        'is_unbalance': True,
        'lambda_l2': 5.0,
        'verbose': -1
    }

    params_stage2 = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        # 'max_depth': 3,
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.02,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        # 'device': 'gpu',
        'bagging_freq': 5,
        'verbose': 1,
        'is_unbalance': True,
        'lambda_l1': 0.1
    }

    #xgb=XGBClassifier()
    lgb=LGBClassifier(params_stage2)
    #rf_model = RandomForestClassifier()
    #adb_model = AdaBoostClassifier()
    RL_model=LogisticRegression()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()

    svc_model = SVC()
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split




    train_sets = []
    test_sets = []
    lgb_stage2 = LGBClassifier(params)
    for clf in [lgb,gdbc_model,et_model, RL_model]:
        print('now is train: '+str(clf.__class__.__name__))
        train_set, test_set = get_stage1(clf, train_x, train_y, test_x)
        train_sets.append(train_set)
        test_sets.append(test_set)

    RL_model_stage2 = LogisticRegression()

    print('now is fun: get_stage2')
    df_predict=get_stage2(RL_model_stage2,train_sets,train_y,test_sets)
    '''
    print('train_auc:', np.mean(xx_auc))
    s = 0
    for each in xx_submit:
        s += each
    operation_test['Tag'] = list(s / N)
    '''
    operation_test['Tag']=df_predict
    test_index = operation_test.groupby('UID').Tag.mean().index
    Tag = operation_test.groupby('UID').Tag.mean().values
    #print("eval:%f" % evaluate.tpr_weight_funtion(y_test, df_predict))
    test1 = pd.DataFrame(test_index)
    test1['Tag'] = Tag
    test1.columns = ['UID', 'Tag']
    test1['Tag'] = test1['Tag'].apply(lambda x: 1 if x > 1 else x)
    test1['Tag'] = test1['Tag'].apply(lambda x: 0 if x < 0 else x)
    #test1[['UID', 'Tag']].to_csv('result.csv', index=False)


    # ===================================处理交易详情=====================================#
    transaction_trn = pd.merge(transaction_trn, tag_trn, how='left', on='UID')
    transaction_test['Tag'] = -1
    df = pd.concat([transaction_trn, transaction_test])

    label_feature = ['channel', 'amt_src1', 'merchant', 'code1', 'code2', 'trans_type1', 'acc_id1',
                     'device_code1', 'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1',
                     'amt_src2', 'acc_id2', 'acc_id3', 'trans_type2', 'market_code', 'ip1_sub']
    for each in label_feature:
        df[each] = pd.factorize(df[each])[0]
    df['hour'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M:%S').hour)
    del df['time']
    for i in range(1, 5):
        df['geo_' + str(i)] = df['geo_code'].apply(lambda g: split_geo(g, i))
    del df['geo_code']


    def mis_impute(data):
        for i in data.columns:
            if data[i].dtype == "object":
                data[i] = data[i].fillna("other")
            elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
                data[i] = data[i].fillna(data[i].mode()[0])  # 还有众数.mode()；中位数.median()
            else:
                pass
        return data
    df = mis_impute(df)

    X = np.array(df[df.Tag != -1].drop(['Tag', 'UID'], axis=1))
    y = np.array(df[df.Tag != -1]['Tag'])
    test = np.array(df[df.Tag == -1].drop(['Tag', 'UID'], axis=1))


    train_x=X
    train_y=y
    test_x=test
    train_sets = []
    test_sets = []

    #xgb = XGBClassifier()
    lgb = LGBClassifier(params_stage2)
    # rf_model = RandomForestClassifier()
    # adb_model = AdaBoostClassifier()
    RL_model = LogisticRegression()
    gdbc_model = GradientBoostingClassifier()
    et_model = ExtraTreesClassifier()
    lgb_stage2 = LGBClassifier(params)
    RL_model_stage2 = LogisticRegression()
    for clf in [lgb,gdbc_model, et_model,RL_model]:
        print('now is train: ' + str(clf.__class__.__name__))
        train_set, test_set = get_stage1(clf, train_x, train_y, test_x)
        train_sets.append(train_set)
        test_sets.append(test_set)

    print('now is fun: get_stage2')
    df_predict = get_stage2(RL_model_stage2, train_sets, train_y, test_sets)

    transaction_test['Tag'] = df_predict
    test_index = transaction_test.groupby('UID').Tag.mean().index
    Tag = transaction_test.groupby('UID').Tag.mean().values
    # print("eval:%f" % evaluate.tpr_weight_funtion(y_test, df_predict))
    test2 = pd.DataFrame(test_index)
    test2['Tag'] = Tag
    test2.columns = ['UID', 'Tag']
    test2['Tag'] = test2['Tag'].apply(lambda x: 1 if x > 1 else x)
    test2['Tag'] = test2['Tag'].apply(lambda x: 0 if x < 0 else x)

    test1 = pd.concat([test1, test2])
    test_index = test1.groupby('UID').Tag.mean().index
    Tag = test1.groupby('UID').Tag.mean().values
    test1 = pd.DataFrame(test_index)
    test1['Tag'] = Tag
    test1[['UID', 'Tag']].to_csv('result.csv', index=False)
