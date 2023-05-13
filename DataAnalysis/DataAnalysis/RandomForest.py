import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

#file = 'сrop-recommendation.xls'
file = 'Farming\AgriculturalTR.xlsx'
# Загрузить таблицу с именем `file`
#data = pd.read_excel(file, usecols='A:H')
data = pd.read_excel(file, engine='openpyxl')
print(data.sample(5))
# общая информация о столбцах, типах и пропущенных значениях
print(data.info())
# общие статистики
print(data.describe())

# как распределено количество азота в почве
data['tср °C (2021)'].hist(bins=5, figsize=(20,4));
plt.show()
print(data[data['tср °C (2021)']>15]['label'].count())

# список столбцов нашего датасета 
print(data.columns)

col=['tмакс °C (2021)', 'tмин °C (2021)', 'tср °C (2021)', 'скорость ветра (2021)', 'осадки (2021)', 'label']
# код ниже преобразует категорийные данные в переменные 
# и заполняет пропуски наиболее вероятным значением
X=pd.DataFrame()
for i in col:
    if data[i].dtype.name != 'object':
        X[i]=data[i].copy()
        X.loc[X[i].isna(), i]=X[i].median()
    else:
        X[i]=pd.factorize(data[i])[0]
# результат, подготовленные данные
print(X.sample(3))
# Y будет равен нулю если содержание азота 40 и меньше, и единице если больше 40
Y=data['2021'].apply(lambda x: 1 if x>15 else 0).values

#разделим набор на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# создаем и тренируем модель, отдельно можно провести подбор параметров для повышения точности
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

ar_f=[]
for f, idx in enumerate(indices):
    ar_f.append([round(importances[idx],4), col[idx]])
print("Значимость признака:")
ar_f.sort(reverse=True)
print(ar_f)

#удобнее отобразить на столбчатой диаграмме
d_first = len(col)
plt.figure(figsize=(8, 8))
plt.title("Значимость признака")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(col)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first]);
plt.show()

# как выглядит результат предсказания для тестовой выборки
print(model.predict(X_test))

# как выглядядт результаты тестового набора
print(y_test)

# метрика r2
print(r2_score(model.predict(X_test), y_test))

from sklearn import metrics
# метрика, насколько точно мы предсказываем правильные значения как для 0, так и 1
print("Accuracy:",metrics.accuracy_score(y_test, model.predict(X_test)))

# матрица количества правильно и ошибочно угаданных классов
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, model.predict(X_test)))

# так же матрица в процентах и более изящном виде
matrix = confusion_matrix(y_test, model.predict(X_test))
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
class_names = ['<15', '>15']                 # !!!!!! указать названия классов!
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Предсказанные классы')
plt.ylabel('Истинные классы')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()