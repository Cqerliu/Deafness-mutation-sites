import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
import lightgbm as lgb
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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df1 = pd.read_excel(r'G:\BP.xlsx',sheet_name='Sheet1')
df1 = fill_missing_values(df1)
df11 = pd.read_excel(r'G:\LBLP.xlsx',sheet_name='Sheet1')
df11 = fill_missing_values(df11)
df2 = pd.read_excel(r'G:\BP.xlsx',sheet_name='Sheet2')
df2 = fill_missing_values(df2)
df21 = pd.read_excel(r'G:\LBLP.xlsx',sheet_name='Sheet2')
df21 = fill_missing_values(df21)
df3 = pd.read_excel(r'G:\BP.xlsx',sheet_name='Sheet3')
df3 = fill_missing_values(df3)
df31 = pd.read_excel(r'G:\LBLP.xlsx',sheet_name='Sheet3')
df31 = fill_missing_values(df31)
df4 = pd.read_excel(r'G:\BP.xlsx',sheet_name='Sheet4')
df4 = fill_missing_values(df4)
df41 = pd.read_excel(r'G:\LBLP.xlsx',sheet_name='Sheet4')
df41 = fill_missing_values(df41)
df5 = pd.read_excel(r'G:\BP.xlsx',sheet_name='Sheet5')
df5 = fill_missing_values(df5)
df51 = pd.read_excel(r'G:\LBLP.xlsx',sheet_name='Sheet5')
df51 = fill_missing_values(df51)
df6 = pd.read_excel(r'G:\BP.xlsx',sheet_name='Sheet6')
df6 = fill_missing_values(df6)
df61 = pd.read_excel(r'G:\LBLP.xlsx',sheet_name='Sheet6')
df61 = fill_missing_values(df61)
df7 = pd.read_excel(r'G:\BP.xlsx',sheet_name='Sheet7')
df7 = fill_missing_values(df7)
df71 = pd.read_excel(r'G:\LBLP.xlsx',sheet_name='Sheet7')
df71 = fill_missing_values(df71)


df20 = pd.read_excel(r'G:\SIFT.xlsx',sheet_name='Sheet1')
df20 = fill_missing_values(df20)

df1_data = df1.iloc[:,7:114].values
df1_labels = df1.iloc[:,114].values
df11_data = df11.iloc[:,6:113].values
df11_labels = df11.iloc[:,113].values
x1 = np.vstack((df1_data, df11_data))

df2_data = df2.iloc[:,7:114].values
df21_data = df21.iloc[:,6:113].values
x2 = np.vstack((df2_data, df21_data))

df3_data = df3.iloc[:,7:114].values
df31_data = df31.iloc[:,6:113].values
x3 = np.vstack((df3_data, df31_data))

df4_data = df4.iloc[:,7:114].values
df41_data = df41.iloc[:,6:113].values
x4 = np.vstack((df4_data, df41_data))

df5_data = df5.iloc[:,7:114].values
df51_data = df51.iloc[:,6:113].values
x5 = np.vstack((df5_data, df51_data))

df6_data = df6.iloc[:,7:114].values
df61_data = df61.iloc[:,6:113].values
x6 = np.vstack((df6_data, df61_data))

df7_data = df7.iloc[:,7:114].values
df71_data = df71.iloc[:,6:113].values
x7 = np.vstack((df7_data, df71_data))


df20_data1 = df20.iloc[:,6:7].values
df20_data2 = df20.iloc[:,7:8].values
df20_data3 = df20.iloc[:,8:9].values
df20_data4 = df20.iloc[:,9:10].values
df20_data5 = df20.iloc[:,10:11].values
df20_data6 = df20.iloc[:,11:12].values
df20_data7 = df20.iloc[:,12:13].values
df20_data8 = df20.iloc[:,13:14].values

df20_labels = df20.iloc[:,14].values

X1 = scaler.fit_transform(x1)
X2 = scaler.fit_transform(x2)
X3 = scaler.fit_transform(x3)
X4 = scaler.fit_transform(x4)
X5 = scaler.fit_transform(x5)
X6 = scaler.fit_transform(x6)
X7 = scaler.fit_transform(x7)
X8 = scaler.fit_transform(df20_data1)
X9 = scaler.fit_transform(df20_data2)
X10 = scaler.fit_transform(df20_data3)
X11 = scaler.fit_transform(df20_data4)
X12 = scaler.fit_transform(df20_data5)
X13 = scaler.fit_transform(df20_data6)
X14 = scaler.fit_transform(df20_data7)
X15 = scaler.fit_transform(df20_data8)

y = np.hstack((df1_labels, df11_labels))

X_train_data, X_test_data, y_train_data1, y_test_data1 = train_test_split(X1, y, test_size=0.2, random_state=22)
X_train_data1, X_test_data1, y_train_data2, y_test_data2 = train_test_split(X2, y, test_size=0.2, random_state=22)
X_train_data2, X_test_data2, y_train_data3, y_test_data3 = train_test_split(X3, y, test_size=0.2, random_state=22)
X_train_data3, X_test_data3, y_train_data4, y_test_data4 = train_test_split(X4, y, test_size=0.2, random_state=22)
X_train_data4, X_test_data4, y_train_data5, y_test_data5 = train_test_split(X5, y, test_size=0.2, random_state=22)
X_train_data5, X_test_data5, y_train_data6, y_test_data6 = train_test_split(X6, y, test_size=0.2, random_state=22)
X_train_data6, X_test_data6, y_train_data7, y_test_data7 = train_test_split(X7, y, test_size=0.2, random_state=22)
X_train_data7, X_test_data7, y_train_data8, y_test_data8 = train_test_split(X8, df20_labels, test_size=0.2, random_state=22)
X_train_data8, X_test_data8, y_train_data9, y_test_data9 = train_test_split(X9, df20_labels, test_size=0.2, random_state=22)
X_train_data9, X_test_data9, y_train_data10, y_test_data10 = train_test_split(X10, df20_labels, test_size=0.2, random_state=22)
X_train_data10, X_test_data10, y_train_data11, y_test_data11 = train_test_split(X11, df20_labels, test_size=0.2, random_state=22)
X_train_data11, X_test_data11, y_train_data12, y_test_data12 = train_test_split(X12, df20_labels, test_size=0.2, random_state=22)
X_train_data12, X_test_data12, y_train_data13, y_test_data13 = train_test_split(X13, df20_labels, test_size=0.2, random_state=22)
X_train_data13, X_test_data13, y_train_data14, y_test_data14 = train_test_split(X14, df20_labels, test_size=0.2, random_state=22)
X_train_data14, X_test_data14, y_train_data15, y_test_data15 = train_test_split(X15, df20_labels, test_size=0.2, random_state=22)




datasets = {
    # 'ALT-50bp': (X_train_data, y_train_data1, X_test_data, y_test_data1),
    # 'ALT-100bp': (X_train_data1, y_train_data2, X_test_data1, y_test_data2),
    # 'ALT-150bp': (X_train_data2, y_train_data3, X_test_data2, y_test_data3),
    # 'ALT-200bp': (X_train_data3, y_train_data4, X_test_data3, y_test_data4),
    # 'ALT-300bp': (X_train_data4, y_train_data5, X_test_data4, y_test_data5),
    'ALT-1000bp': (X_train_data6, y_train_data7, X_test_data6, y_test_data7),
    'ALT-500bp': (X_train_data5, y_train_data6, X_test_data5, y_test_data6),
    'CADD': (X_train_data9, y_train_data10, X_test_data9, y_test_data10),
    'GEPR': (X_train_data10, y_train_data11, X_test_data10, y_test_data11),
    'Eigen': (X_train_data11, y_train_data12, X_test_data11, y_test_data12),
    'SIFT': (X_train_data12, y_train_data13, X_test_data12, y_test_data13),
    'M-CAP': (X_train_data14, y_train_data15, X_test_data14, y_test_data15),
    'Revel': (X_train_data7, y_train_data8, X_test_data7, y_test_data8),
    'MutationTaster': (X_train_data13, y_train_data14, X_test_data13, y_test_data14),
    'Provean': (X_train_data8, y_train_data9, X_test_data8, y_test_data9),
}


auc_values = []
data_names = []


for data_name, data in datasets.items():
    X_train, y_train, X_test, y_test = data
    # classifier = lgb.LGBMClassifier()
    classifier = CatBoostClassifier()

    # classifier = xgb.XGBClassifier(colsample_bytree= 0.9,eta= 0.2, gamma= 0.3, max_depth= 6, min_child_weight=8, n_estimators= 500, subsample= 0.8)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)[:, 1]


    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc_value = roc_auc_score(y_test, y_pred)


    data_names.append(data_name)
    auc_values.append(auc_value)

    plt.plot(fpr, tpr, label=f'{data_name} (AUC = {auc_value:.4f})')



plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC')

# plt.grid(True)
plt.legend(loc='lower right')
sns.set(style='whitegrid')
plt.show()

for i in range(len(data_names)):
    print(f'{data_names[i]} AUC: {auc_values[i]:.4f}')

