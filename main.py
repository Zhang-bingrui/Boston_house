#多特征线性回顾预测——波士顿房价线性回归预测
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import math
#忽略警告
import warnings
warnings.filterwarnings('ignore')

#特征和标签的划分
def Feature_Division(input):
    Boston=pd.read_csv('house_data.csv')
    X = Boston.drop(['MEDV'],axis=1)
    y = Boston[['MEDV']]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    #规定返回值
    if input=='X_train':
        return X_train
    if input=='X_test':
        return X_test
    if input=='y_train':
        return y_train
    if input=='y_test':
        return y_test

#线性回归各种误差验证
def Error_verification(input):
    if input == 'linear':
        # 数据验证——平均绝对误差
        a = metrics.mean_absolute_error(Feature_Division("y_test"), linear_Model_result_output())
        # 均方误差
        b = metrics.mean_squared_error(Feature_Division("y_test"), linear_Model_result_output())
        # 均方根误差
        c = math.sqrt(b)
        # 均方根对数误差
        d = metrics.mean_squared_log_error(Feature_Division("y_test"), linear_Model_result_output())
        # R平方（可决系数误差）
        e = metrics.r2_score(Feature_Division("y_test"), linear_Model_result_output())
        print("线性回归预测平均绝对误差：", a,
              "均方误差:", b,
              "均方根误差:", c,
              "均方根对数误差:", d,
              "R平方:", e)
    else:
        # 数据验证——平均绝对误差
        a = metrics.mean_absolute_error(Feature_Division("y_test"), tree_Model_result_output())
        # 均方误差
        b = metrics.mean_squared_error(Feature_Division("y_test"), tree_Model_result_output())
        # 均方根误差
        c = math.sqrt(b)
        # 均方根对数误差
        d = metrics.mean_squared_log_error(Feature_Division("y_test"), tree_Model_result_output())
        # R平方（可决系数误差）
        e = metrics.r2_score(Feature_Division("y_test"), tree_Model_result_output())
        print("随机森林回归预测平均绝对误差：", a,
              "均方误差:", b,
              "均方根误差:", c,
              "均方根对数误差:", d,
              "R平方:", e)


#简单线性模型建立，数据拟合
def linear_Model_result_output():
    # 数据拟合
    model = LinearRegression()
    model.fit(Feature_Division("X_train"), Feature_Division("y_train"))
    # 测试集评估
    model.score(Feature_Division("X_test"),Feature_Division("y_test"))
    # 数据预测
    y_pred_class = model.predict(Feature_Division("X_test"))
    return y_pred_class
#随机森林的线性回归模型建立，数据拟合
def tree_Model_result_output():
    param_grid = {
        'n_estimators':[150],
        'max_depth':[7],
        'max_features':[5]
    }
    rf = RandomForestRegressor()
    grid = GridSearchCV(rf,param_grid=param_grid,cv=3)
    grid.fit(Feature_Division("X_train"),Feature_Division("y_train"))
    rf_reg = grid.best_estimator_
    return rf_reg.predict(Feature_Division("X_test"))


def main():
    Error_verification("linear")
    Error_verification("tree")


if __name__ == '__main__':
    main()
















'''

#train_test_split用法

# train_data：所要划分的样本特征集,*arrays：可以是列表、numpy数组、scipy稀疏矩阵或pandas的数据框.

# train_target：所要划分的样本结果

# test_size：样本占比，如果是整数的话就是样本的数量
以为浮点、整数或None，默认为None
①若为浮点时，表示测试集占总样本的百分比
②若为整数时，表示测试样本样本数
③若为None时，test size自动设置成0.25

# random_state：是随机数的种子。
 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
可以为整数、RandomState实例或None，默认为None
①若为None时，每次生成的数据都是随机，可能不一样
②若为整数时，每次生成的数据都相同

#stratify:为了保持split前类的分布。可以为类似数组或None
①若为None时，划分出来的测试集或训练集中，其类标签的比例也是随机的
②若不为None时，划分出来的测试集或训练集中，其类标签的比例同输入的数组中类标签的比例相同，可以用于处理不均衡的数据集
比如有100个数据，80个属于A类，20个属于B类。如果train_test_split(... test_size=0.25, stratify = y_all), 那么split之后数据如下： 
training: 75个数据，其中60个属于A类，15个属于B类。 
testing: 25个数据，其中20个属于A类，5个属于B类。 
用了stratify参数，training集和testing集的类的比例是 A：B= 4：1，等同于split前的比例（80：20）。通常在这种类分布不平衡的情况下会用到stratify。
将stratify=X就是按照X中的比例分配 
将stratify=y就是按照y中的比例分配 

'''
