# загрузим библиотеки
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#import plotly.express as px

file = 'Farming\AgriculturalTR.xlsx'
data=pd.read_excel(file, engine='openpyxl')
#print(data)

agricultural = pd.DataFrame()
table1 = pd.DataFrame()
table1['subjects_of_the_RF']=data['subjects_of_the_RF']
table1['yield']=data['2021']
table1['year_of_harvest']=2021
table1['label']=data['label']
table1['max_temp']=data['tмакс °C (2021)']
table1['min_temp']=data['tмин °C (2021)']
table1['mean_temp']=data['tср °C (2021)']
table1['wind_speed']=data['скорость ветра (2021)']
table1['rainfall']=data['осадки (2021)']
#print(table1)

table2 = pd.DataFrame()
table2['subjects_of_the_RF']=data['subjects_of_the_RF']
table2['yield']=data['2020']
table2['year_of_harvest']=2020
table2['label']=data['label']
table2['max_temp']=data['tмакс °C (2020)']
table2['min_temp']=data['tмин °C (2020)']
table2['mean_temp']=data['tср °C (2020)']
table2['wind_speed']=data['скорость ветра (2020)']
table2['rainfall']=data['осадки (2020)']
#print(table2)

agricultural = table1.append(table2, ignore_index=True)

table3 = pd.DataFrame()
table3['subjects_of_the_RF']=data['subjects_of_the_RF']
table3['yield']=data['2019']
table3['year_of_harvest']=2019
table3['label']=data['label']
table3['max_temp']=data['tмакс °C (2019)']
table3['min_temp']=data['tмин °C (2019)']
table3['mean_temp']=data['tср °C (2019)']
table3['wind_speed']=data['скорость ветра (2019)']
table3['rainfall']=data['осадки (2019)']
#print(table3)

agricultural = agricultural.append(table3, ignore_index=True)

table4 = pd.DataFrame()
table4['subjects_of_the_RF']=data['subjects_of_the_RF']
table4['yield']=data['2018']
table4['year_of_harvest']=2018
table4['label']=data['label']
table4['max_temp']=data['tмакс °C (2018)']
table4['min_temp']=data['tмин °C (2018)']
table4['mean_temp']=data['tср °C (2018)']
table4['wind_speed']=data['скорость ветра (2018)']
table4['rainfall']=data['осадки (2018)']
#print(table4)

agricultural = agricultural.append(table4, ignore_index=True)
agricultural.to_excel('Farming\AgriculturalRRR.xlsx', index=True);
print(agricultural, "\n")

print("Вывод информации о столбцах таблицы и типах переменных\n")
print(agricultural.info(), "\n")

print("Cписок столбцов для удобного копирования: \n")
print(agricultural.columns, "\n")

pivot_region = agricultural.pivot_table(index = 'subjects_of_the_RF', values = 'yield', aggfunc = 'sum')
top_regions = pivot_region.sort_values(by = 'yield',ascending = False).head(10)
print(top_regions)
top10_list = top_regions.index.tolist()
print(top10_list)

pivot_for_analysis = agricultural.query('subjects_of_the_RF in @top10_list')
popular_regions = pivot_for_analysis.pivot_table(
    index = 'label', columns = 'subjects_of_the_RF', values = 'yield', aggfunc = 'sum')
plot_for_analysis = popular_regions.plot(figsize=(15,10))
#plot_for_analysis = sns.distplot(pivot_for_analysis, x = "year_of_release", hue="platform", kind="kde", fill=True)
plot_for_analysis.set_xlabel('Сельхоз. культуры')
plot_for_analysis.set_ylabel('Кол-во собранного урожая')
plt.title("Распределение урожайности по с.\х. культурам в разных субъектах РФ")
plt.show()

print(popular_regions.describe())

#построим график с накопления для лучшей визуализации
colors = ['maroon','burlywood','darkorange','black','peachpuff','forestgreen','olivedrab', 'orchid', 'indigo',
          'crimson', 'violet', 'cyan'
         ]
plot_of_prediction = pivot_for_analysis.pivot_table(
    index = 'label', columns = 'subjects_of_the_RF', values = 'yield', aggfunc = 'sum').plot.area(
    figsize = (15,10),color = colors, alpha = 0.8
                                                                                                    )
plot_of_prediction.set_xlabel('сельхоз. культуры')
plot_of_prediction.set_ylabel('кол-во собранного урожая')
plt.title('распределение урожайности')
plt.show()

actual_platform = agricultural.query('subjects_of_the_RF == "Тульская область" | subjects_of_the_RF == "Воронежская область" | subjects_of_the_RF == "Тамбовская область"')
sns.set(rc={'figure.figsize':(15, 10)})
ax = sns.boxplot(x = "subjects_of_the_RF", y = 'yield', data = actual_platform)
#plt.ylim(0,3)
plt.title('Диаграммы размаха актуальных платформ')
ax.set_xlabel('платформа')
ax.set_ylabel('глобальные продажы')
plt.show()

def correlation(product):
    sales = agricultural.query('label == @product')[['mean_temp', 'wind_speed', 'rainfall','yield']]
    print(sales.corr())
    pd.plotting.scatter_matrix(sales)
    plt.show()

correlation('горох')
#correlation('вики')
#correlation('фасоль')
#correlation('люпин')
#correlation('сорго')
#correlation('нут')
#correlation('чечевица')

products = agricultural.pivot_table(index = 'label', values = 'yield').sort_values(by = 'yield', ascending = False).head()
print(products)
colors = ['burlywood','gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'crimson', 'violet', 'cyan']
sizes = products['yield']
plt.pie(sizes , labels = products.index,colors=colors,autopct='%1.1f%%', startangle=140)
plt.title('Распределение урожайности')
plt.show()

from sklearn import linear_model, model_selection # импортируем линейную модель для обучения и библиотеку для разделения нашей выборки
X = agricultural.query('label == \'горох\'')['yield'].values.reshape(-1,1) # создаем вектор признаков, вектора так как у нас один признак 
y = agricultural.query('label == \'горох\'')['mean_temp'].values.reshape(-1,1) # создаем вектор ответом  
plt.scatter(X, y) # рисуем график точек
plt.xlabel('Урожай, ц. с га.') # добавляем описание для оси x
plt.ylabel('Спедняя темпкратура')# добавляем описание для оси y
plt.show()

# Делим созданную нами выборку на тестовую и обучающую
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y) 
regr = linear_model.LinearRegression() # создаем линейную регрессию 
#Обучаем модель
regr.fit(X_train, y_train)
linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# Посмотрим какие коэффициенты установила модель
print('Коэфициент: \n', regr.coef_)
# Средний квадрат ошибки
print("Средний квадрат ошибки: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Оценка дисперсии: 1 - идеальное предсказание. Качество предсказания.
print('Оценка дисперсии:: %.2f' % regr.score(X_test, y_test))
# Посмотрим на получившуюся функцию 
print ("y = {:.2f}*x + {:.2f}".format(regr.coef_[0][0], regr.intercept_[0]))
# Посмотрим, как предскажет наша модель тестовые данные.
plt.scatter(X_test, y_test, color='black')# рисуем график точек
plt.plot(X_test, regr.predict(X_test), color='blue') # рисуем график линейной регрессии 
plt.show() # Покажем график 

print(regr.predict([[15]])) 