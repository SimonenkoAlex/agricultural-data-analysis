# загрузим библиотеки
import pandas as pd
import matplotlib
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.cluster import KMeans

# Назначить имя файла электронной таблицы `file`
#file = 'сrop-recommendation.xls'
file = 'Farming\AgriculturalTR.xlsx'
#file = 'Farming\AgriculturalRRR.xlsx'

# Загрузить таблицу
#data = pd.read_excel(file, usecols='A:H')
data=pd.read_excel(file, engine='openpyxl')
# Вывод пяти случайных строк таблицы, 
# таблица не отобразиться полностью
print(data.sample(5))
# Вывод информации о столбцах таблицы и типах переменных
print(data.info())
# статистики по столбцам с количественными переменными
print(data.describe())
# ниже выводит список столцов, удобно для копирования
print(data.columns)
# укажим количественные (int, float) столбцы, 
# по которым выполним кластеризацию
#col=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
#col=['agricultural products', 'crop production', 
#     'livestock products', 'temperature', 'rainfall']
col=['2021', 'tмакс °C (2021)', 'tмин °C (2021)', 'tср °C (2021)', 'скорость ветра (2021)', 'осадки (2021)']
#col=['yield', 'max_temp', 'min_temp', 'mean_temp', 'wind_speed', 'rainfall']


pd.options.mode.chained_assignment = None 
# заменим пропуски данных нулями, в противном случае выдаст ошибку
data[col].fillna(0, inplace=True)
# матрица рассеяния и гистограммы
from pandas.plotting import scatter_matrix
scatter_matrix(data[col], alpha=0.05, figsize=(10, 10));
plt.show()

print(data[col].corr()) # посмотрим на парные корреляции

# загружаем библиотеку препроцесинга данных
# эта библиотека автоматически приведен данные к нормальным значениям
from sklearn import preprocessing
dataNorm = preprocessing.MinMaxScaler().fit_transform(data[col].values)
print(dataNorm[:5])

# Вычислим расстояния между каждым набором данных,
# т.е. строками массива data_for_clust
# Вычисляется евклидово расстояние (по умолчанию)
data_dist = pdist(dataNorm, 'euclidean')

# Главная функция иерархической кластеризии
# Объедение элементов в кластера и сохранение в 
# специальной переменной (используется ниже для визуализации 
# и выделения количества кластеров
data_linkage = linkage(data_dist, method='average')
# Метод локтя. Позволячет оценить оптимальное количество сегментов.
# Показывает сумму внутри групповых вариаций
last = data_linkage[-10:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.show()
k = acceleration_rev.argmax() + 2 
print("Рекомендованное количество кластеров:", k)

'''
#функция построения дендрограмм
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    
    ddata = dendrogram(*args, **kwargs)
    
    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
'''

# !!!!!!!!! укажите, какое количество кластеров будете использовать!
nClust=k

'''
#строим дендрограмму
fancy_dendrogram( 
    data_linkage,
    truncate_mode='lastp',
    p=nClust,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
)
'''

# иерархическая кластеризация
clusters=fcluster(data_linkage, nClust, criterion='maxclust')
print(clusters[:])
print(col)
x=0 # Чтобы построить диаграмму в разных осях, меняйте номера столбцов
y=1 #
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=clusters, cmap='flag')
plt.xlabel(col[x])
plt.ylabel(col[y]);
plt.show()

# к оригинальным данным добавляем номер кластера
data['I']=clusters
res=data.groupby('I')[col].mean()
res['Количество']=data.groupby('I').size().values
#ниже средние цифры по кластерам и количество объектов (Количество)
print(res[:])

col=['subjects_of_the_RF', '2021', 'label', 'tср °C (2021)', 'скорость ветра (2021)', 'осадки (2021)', 'I']
print(data[col][data['I']==1],"\n") # !!!!! меняйте номер кластера
print(data[col][data['I']==2],"\n")
print(data[col][data['I']==3],"\n")
data.to_excel('result_claster1.xlsx', index=False)
col=['2021', 'tмакс °C (2021)', 'tмин °C (2021)', 'tср °C (2021)', 'скорость ветра (2021)', 'осадки (2021)']

nan_claster=np.where(data['I']<3)
print("\nИндексы удаляемых элементов 1-го и 2-го кластера:\n", nan_claster, "\n")
data.drop(nan_claster[0], axis = 0, inplace=True)
dataNorm = preprocessing.MinMaxScaler().fit_transform(data[col].values)

print(data)

# строим кластеризаци методом KMeans
km = KMeans(n_clusters=nClust).fit(dataNorm)
# выведем полученное распределение по кластерам
# так же номер кластера, к котрому относится строка, так как нумерация начинается с нуля, выводим добавляя 1
km.labels_ +1

x=0 # Чтобы построить диаграмму в разных осях, меняйте номера столбцов
y=1 #
centroids = km.cluster_centers_
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=km.labels_, cmap='flag')
plt.scatter(centroids[:, x], centroids[:, y], marker='*', s=300,
            c='r', label='centroid')
plt.xlabel(col[x])
plt.ylabel(col[y]);
plt.show()

# к оригинальным данным добавляем номера кластеров
data['KMeans']=km.labels_+1
res=data.groupby('KMeans')[col].mean()
res['Количество']=data.groupby('KMeans').size().values
print(res[:])

# изменяйте номер кластера, содержание которого хотите просмотреть
col=['subjects_of_the_RF', '2021', 'label', 'tср °C (2021)', 'скорость ветра (2021)', 'осадки (2021)', 'KMeans']
print(data[col][data['KMeans']==1],"\n")
print(data[col][data['KMeans']==2],"\n")
print(data[col][data['KMeans']==3],"\n")
# сохраним результаты в файл
data.to_excel('result_claster2.xlsx', index=False)