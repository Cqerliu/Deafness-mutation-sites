import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn import neural_network ,preprocessing
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import pandas as pd
import lightgbm as lgb
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def MAXmin(train_feature,test_feature):
    min_max_scaler = preprocessing.MinMaxScaler().fit(train_feature)
    train_minmax = min_max_scaler.transform(train_feature)
    test_minmax = min_max_scaler.transform(test_feature)
    return train_minmax, test_minmax
def fill_missing_values(data):
    # 检测缺失值
    missing_values = data.isnull().sum()
    if missing_values.sum() == 0:
        print("Excel文件中没有缺失值。")
        return data

    # 计算每列的平均值
    column_means = data.mean()

    # 填补缺失值
    filled_data = data.fillna(column_means)

    print("缺失值已填补。")
    return filled_data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


df20 = pd.read_excel(r'G:\SIFT.xlsx',sheet_name='Sheet1')
df20 = fill_missing_values(df20)
df21 = pd.read_excel(r'G:\SIFT.xlsx',sheet_name='Sheet2')
df21 = fill_missing_values(df21)
df20_data = df20.iloc[:,6:14].values
df20_labels = df20.iloc[:,14].values
df21_data = df21.iloc[:,5:13].values
df21_labels = df21.iloc[:,13].values
x1 = df20_data
y_train = df20_labels
x2 = df21_data
y_test = df21_labels


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x1)
X_test = scaler.fit_transform(x2)
# X_train = x1
# X_test = x2
# X_train,X_test = MAXmin(x1,x2)

classifiers = {
    'lightGBM' : lgb.LGBMClassifier()
    # 'SVM': SVC(probability=True,C=0.1, gamma = 0.1, kernel='linear'),
    # 'Random Forest': RandomForestClassifier(),
    # 'XGBoost': xgb.XGBClassifier(),
    # 'BPNN': MLPClassifier(),
    # 'CatBoost': CatBoostClassifier()
}

auc_scores = {}
acc_scores = {}
sn_scores = {}
sp_scores = {}
mcc_scores = {}

for classifier_name, classifier in classifiers.items():
    auc_scores[classifier_name] = []
    acc_scores[classifier_name] = []
    sn_scores[classifier_name] = []
    sp_scores[classifier_name] = []
    mcc_scores[classifier_name] = []


    for _ in range(10):

        model = classifier
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

   
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, y_pred)
        sn = recall_score(y_test, y_pred)
        sp = precision_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

 
        auc_scores[classifier_name].append(auc)
        acc_scores[classifier_name].append(acc)
        sn_scores[classifier_name].append(sn)
        sp_scores[classifier_name].append(sp)
        mcc_scores[classifier_name].append(mcc)


avg_auc_scores = {classifier_name: np.mean(scores) for classifier_name, scores in auc_scores.items()}
avg_acc_scores = {classifier_name: np.mean(scores) for classifier_name, scores in acc_scores.items()}
avg_sn_scores = {classifier_name: np.mean(scores) for classifier_name, scores in sn_scores.items()}
avg_sp_scores = {classifier_name: np.mean(scores) for classifier_name, scores in sp_scores.items()}
avg_mcc_scores = {classifier_name: np.mean(scores) for classifier_name, scores in mcc_scores.items()}


for classifier_name in classifiers.keys():
    print(f"分类器: {classifier_name}")
    print(f"平均AUC: {avg_auc_scores[classifier_name]:.4f}")
    print(f"平均ACC: {avg_acc_scores[classifier_name]:.4f}")
    print(f"平均SN: {avg_sn_scores[classifier_name]:.4f}")
    print(f"平均SP: {avg_sp_scores[classifier_name]:.4f}")
    print(f"平均MCC: {avg_mcc_scores[classifier_name]:.4f}")
    print()
