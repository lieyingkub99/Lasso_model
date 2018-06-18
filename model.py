import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

# ----------------数据处理------------------
"""
划分训练集和测试集8：2
"""
def data_handle():
    data = pd.read_excel("data/text.xlsx", header=None)
    data = data.ix[:, 1:8]
    data.columns = ["radar_1", "radar_2", "radar_3", "radar_4", "radar_5", "radar_6", "radar_7", "radar_8"]
    label = data.pop("radar_8")
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=0)
    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
#-------------------CV-5折验证-------------------
def mse_cv(model, X_train, y):
    mse = -cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)
    return mse
#---------------模型训练---------------------
"""
    采用线性回归
"""
def model_train(feature, train, label,flag,labelname):
    #--归一化处理---
    N = preprocessing.RobustScaler()
    scale_feature = N.fit_transform(feature)
    train_feature = scale_feature[:train.shape[0]]
    test_feature = scale_feature[train.shape[0]:]
    print(train_feature.shape, test_feature.shape)

    #----------------liner_model----------------
    #--cv-5折交叉验证选取最优参数
    alphas = np.logspace(-4, -1, 30)
    cv_lasso = [mse_cv(linear_model.Lasso(alpha), train_feature, label).mean() for alpha in alphas]

    
    # print(alphas)
    # print(cv_lasso)
    index = list(cv_lasso).index(min(cv_lasso))
    print("=best_mse      :", min(cv_lasso))
    print("=best_alphas   :", alphas[index])
    clf = linear_model.Lasso(alphas[index])
    model = clf.fit(train_feature, label)
    res = model.predict(test_feature)
    print("==模型系数：",model.coef_)
    test = pd.read_csv("data/test.csv")
    test["pred"] = res
    test[[labelname, "pred"]].to_csv('data/result_{}.csv'.format(flag), header=None, index=False)

#--------------------主函数---------------------
if __name__ == '__main__':

    #--数据处理--
    data_handle()
    data_train = pd.read_csv("data/train.csv")
    data_test = pd.read_csv("data/test.csv")

    data = pd.concat([data_train, data_test], axis=0)

    #--模型训练--
    for i in range(1,9):
        print(i)
        index_feat = list(data_train.columns)
        label = data_train["radar_{}".format(i)]
        index_feat.remove("radar_{}".format(i))
        feat = data_train[index_feat]
        data_all = data[index_feat]
        model_train(data_all,feat,label,i,"radar_{}".format(i))

