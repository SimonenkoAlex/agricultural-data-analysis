import pandas as pd
import matplotlib
import os
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.special import erfc
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import seaborn as sns

climate2021 = ['Climate\Climate-2021-1.xlsx', 'Climate\Climate-2021-2.xlsx', 'Climate\Climate-2021-3.xlsx']
climate2020 = ['Climate\Climate-2020-1.xlsx', 'Climate\Climate-2020-2.xlsx', 'Climate\Climate-2020-3.xlsx']
climate2019 = ['Climate\Climate-2019-1.xlsx', 'Climate\Climate-2019-2.xlsx', 'Climate\Climate-2019-3.xlsx']
climate2018 = ['Climate\Climate-2018-1.xlsx', 'Climate\Climate-2018-2.xlsx', 'Climate\Climate-2018-3.xlsx']

def main():
    resp2021 = urlopen('https://rosstat.gov.ru/storage/mediabank/Val1.zip')
    resp2020 = urlopen('https://rosstat.gov.ru/storage/mediabank/Val1_2020.zip')
    resp2019 = urlopen('https://rosstat.gov.ru/storage/mediabank/Z2bDPBEl/Val1-19.zip')

    zip2021 = ZipFile(BytesIO(resp2021.read()))
    zip2020 = ZipFile(BytesIO(resp2020.read()))
    zip2019 = ZipFile(BytesIO(resp2019.read()))

    print("2021 -\n", zip2021.namelist(), "\n")
    print("2020 -\n", zip2020.namelist(), "\n")
    print("2019 -\n", zip2019.namelist(), "\n")

    zip2021.extract('tab123.xls')
    os.rename("tab123.xls", "agricultural2021.xls")
    zip2020.extract('tab123.xls')
    os.rename("tab123.xls", "agricultural2020.xls")
    zip2019.extract('tab123.xls')
    os.rename("tab123.xls", "agricultural2019.xls")
    
    peas2021 = pd.read_excel('agricultural2021.xls', sheet_name='T_1123', header=4)
    peas2020 = pd.read_excel('agricultural2020.xls', sheet_name='T_1123', header=4)
    peas2019 = pd.read_excel('agricultural2019.xls', sheet_name='T_1123', header=4)

    beans2021 = pd.read_excel('agricultural2021.xls', sheet_name='T_1124', header=4)
    beans2020 = pd.read_excel('agricultural2020.xls', sheet_name='T_1124', header=4)
    beans2019 = pd.read_excel('agricultural2019.xls', sheet_name='T_1124', header=4)
    #lentils (чечевица)
    lentils2021 = pd.read_excel('agricultural2021.xls', sheet_name='T_1125', header=4)
    lentils2020 = pd.read_excel('agricultural2020.xls', sheet_name='T_1125', header=4)
    lentils2019 = pd.read_excel('agricultural2019.xls', sheet_name='T_1125', header=4)
    #chickpeas (нут)
    chickpeas2021 = pd.read_excel('agricultural2021.xls', sheet_name='T_1129', header=4)
    chickpeas2020 = pd.read_excel('agricultural2020.xls', sheet_name='T_1129', header=4)
    chickpeas2019 = pd.read_excel('agricultural2019.xls', sheet_name='T_1129', header=4)
    #vetch (вики и смесей виковых)
    vetch2021 = pd.read_excel('agricultural2021.xls', sheet_name='T_1127', header=4)
    vetch2020 = pd.read_excel('agricultural2020.xls', sheet_name='T_1127', header=4)
    vetch2019 = pd.read_excel('agricultural2019.xls', sheet_name='T_1127', header=4)
    #lupine (люпин)
    lupine2021 = pd.read_excel('agricultural2021.xls', sheet_name='T_1128', header=4)
    lupine2020 = pd.read_excel('agricultural2020.xls', sheet_name='T_1128', header=4)
    lupine2019 = pd.read_excel('agricultural2019.xls', sheet_name='T_1128', header=4)
    #sorgo (сорго)
    sorgo2021 = pd.read_excel('agricultural2021.xls', sheet_name='T_1135', header=4)
    sorgo2020 = pd.read_excel('agricultural2020.xls', sheet_name='T_1135', header=4)
    sorgo2019 = pd.read_excel('agricultural2019.xls', sheet_name='T_1135', header=4)

    df = pd.DataFrame()
    df['subjects_of_the_RF']=peas2021['Unnamed: 0']
    df.set_index('subjects_of_the_RF', inplace=True)
    delete_col = ['РОССИЙСКАЯ ФЕДЕРАЦИЯ', 'ЦЕНТРАЛЬНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-ЗАПАДНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 
                   'ЮЖНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-КАВКАЗСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 
                   'УРАЛЬСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ДАЛЬНЕВОСТОЧНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ',
                   'г. Москва', 'г. Севастополь']
    df.drop(delete_col, axis = 0,  inplace=True)
    df.reset_index(inplace=True)

    climate2021df = read_climate_data(climate2021)
    climate2021df['subjects_of_the_RF'] = df['subjects_of_the_RF']
    #print(climate2021df, "\n")
    climate2020df = read_climate_data(climate2020)
    climate2020df['subjects_of_the_RF'] = df['subjects_of_the_RF']
    #print(climate2020df, "\n")
    climate2019df = read_climate_data(climate2019)
    climate2019df['subjects_of_the_RF'] = df['subjects_of_the_RF']
    #print(climate2019df, "\n")
    climate2018df = read_climate_data(climate2018)
    climate2018df['subjects_of_the_RF'] = df['subjects_of_the_RF']
    #print(climate2018df, "\n")

    agricultural = pd.DataFrame()
    peas = create_dataframe(peas2021, peas2020, peas2019)
    #peas.to_excel('Farming\SborValPeas.xlsx', index=True)
    peas['label']="горох"
    datap = peas.merge(climate2021df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datap.rename({'tempMax': 'tмакс °C (2021)', 'tempMin': 'tмин °C (2021)', 'tempMean': 'tср °C (2021)', 'speed': 'скорость ветра (2021)', 'fallout': 'осадки (2021)'}, axis=1, inplace=True)
    datap = datap.merge(climate2020df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datap.rename({'tempMax': 'tмакс °C (2020)', 'tempMin': 'tмин °C (2020)', 'tempMean': 'tср °C (2020)', 'speed': 'скорость ветра (2020)', 'fallout': 'осадки (2020)'}, axis=1, inplace=True)
    datap = datap.merge(climate2019df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datap.rename({'tempMax': 'tмакс °C (2019)', 'tempMin': 'tмин °C (2019)', 'tempMean': 'tср °C (2019)', 'speed': 'скорость ветра (2019)', 'fallout': 'осадки (2019)'}, axis=1, inplace=True)
    datap = datap.merge(climate2018df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datap.rename({'tempMax': 'tмакс °C (2018)', 'tempMin': 'tмин °C (2018)', 'tempMean': 'tср °C (2018)', 'speed': 'скорость ветра (2018)', 'fallout': 'осадки (2018)'}, axis=1, inplace=True)
    datap.to_excel('Farming\SborValPeasT.xlsx', index=True)

    beans = create_dataframe(beans2021, beans2020, beans2019)
    #beans.to_excel('Farming\SborValBeans.xlsx', index=True)
    beans['label']="фасоль"
    datab = beans.merge(climate2021df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datab.rename({'tempMax': 'tмакс °C (2021)', 'tempMin': 'tмин °C (2021)', 'tempMean': 'tср °C (2021)', 'speed': 'скорость ветра (2021)', 'fallout': 'осадки (2021)'}, axis=1, inplace=True)
    datab = datab.merge(climate2020df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datab.rename({'tempMax': 'tмакс °C (2020)', 'tempMin': 'tмин °C (2020)', 'tempMean': 'tср °C (2020)', 'speed': 'скорость ветра (2020)', 'fallout': 'осадки (2020)'}, axis=1, inplace=True)
    datab = datab.merge(climate2019df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datab.rename({'tempMax': 'tмакс °C (2019)', 'tempMin': 'tмин °C (2019)', 'tempMean': 'tср °C (2019)', 'speed': 'скорость ветра (2019)', 'fallout': 'осадки (2019)'}, axis=1, inplace=True)
    datab = datab.merge(climate2018df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datab.rename({'tempMax': 'tмакс °C (2018)', 'tempMin': 'tмин °C (2018)', 'tempMean': 'tср °C (2018)', 'speed': 'скорость ветра (2018)', 'fallout': 'осадки (2018)'}, axis=1, inplace=True)
    datab.to_excel('Farming\SborValBeansT.xlsx', index=True)

    #agricultural = peas.append(beans, ignore_index=True)
    agricultural = datap.append(datab, ignore_index=True)

    lentils = create_dataframe(lentils2021, lentils2020, lentils2019)
    #lentils.to_excel('Farming\SborValLentils.xlsx', index=True)
    lentils['label']="чечевица"
    datal = lentils.merge(climate2021df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datal.rename({'tempMax': 'tмакс °C (2021)', 'tempMin': 'tмин °C (2021)', 'tempMean': 'tср °C (2021)', 'speed': 'скорость ветра (2021)', 'fallout': 'осадки (2021)'}, axis=1, inplace=True)
    datal = datal.merge(climate2020df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datal.rename({'tempMax': 'tмакс °C (2020)', 'tempMin': 'tмин °C (2020)', 'tempMean': 'tср °C (2020)', 'speed': 'скорость ветра (2020)', 'fallout': 'осадки (2020)'}, axis=1, inplace=True)
    datal = datal.merge(climate2019df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datal.rename({'tempMax': 'tмакс °C (2019)', 'tempMin': 'tмин °C (2019)', 'tempMean': 'tср °C (2019)', 'speed': 'скорость ветра (2019)', 'fallout': 'осадки (2019)'}, axis=1, inplace=True)
    datal = datal.merge(climate2018df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datal.rename({'tempMax': 'tмакс °C (2018)', 'tempMin': 'tмин °C (2018)', 'tempMean': 'tср °C (2018)', 'speed': 'скорость ветра (2018)', 'fallout': 'осадки (2018)'}, axis=1, inplace=True)
    datal.to_excel('Farming\SborValLentilsT.xlsx', index=True)

    #agricultural = agricultural.append(lentils, ignore_index=True)
    agricultural = agricultural.append(datal, ignore_index=True)

    chickpeas = create_dataframe(chickpeas2021, chickpeas2020, chickpeas2019)
    #chickpeas.to_excel('Farming\SborValChickpeas.xlsx', index=True)
    chickpeas['label']="нут"
    datac = chickpeas.merge(climate2021df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datac.rename({'tempMax': 'tмакс °C (2021)', 'tempMin': 'tмин °C (2021)', 'tempMean': 'tср °C (2021)', 'speed': 'скорость ветра (2021)', 'fallout': 'осадки (2021)'}, axis=1, inplace=True)
    datac = datac.merge(climate2020df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datac.rename({'tempMax': 'tмакс °C (2020)', 'tempMin': 'tмин °C (2020)', 'tempMean': 'tср °C (2020)', 'speed': 'скорость ветра (2020)', 'fallout': 'осадки (2020)'}, axis=1, inplace=True)
    datac = datac.merge(climate2019df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datac.rename({'tempMax': 'tмакс °C (2019)', 'tempMin': 'tмин °C (2019)', 'tempMean': 'tср °C (2019)', 'speed': 'скорость ветра (2019)', 'fallout': 'осадки (2019)'}, axis=1, inplace=True)
    datac = datac.merge(climate2018df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datac.rename({'tempMax': 'tмакс °C (2018)', 'tempMin': 'tмин °C (2018)', 'tempMean': 'tср °C (2018)', 'speed': 'скорость ветра (2018)', 'fallout': 'осадки (2018)'}, axis=1, inplace=True)
    datac.to_excel('Farming\SborValChickpeasT.xlsx', index=True)

    #agricultural = agricultural.append(chickpeas, ignore_index=True)
    agricultural = agricultural.append(datac, ignore_index=True)

    vetch = create_dataframe(vetch2021, vetch2020, vetch2019)
    #vetch.to_excel('Farming\SborValVetch.xlsx', index=True)
    vetch['label']="вики"
    datav = vetch.merge(climate2021df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datav.rename({'tempMax': 'tмакс °C (2021)', 'tempMin': 'tмин °C (2021)', 'tempMean': 'tср °C (2021)', 'speed': 'скорость ветра (2021)', 'fallout': 'осадки (2021)'}, axis=1, inplace=True)
    datav = datav.merge(climate2020df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datav.rename({'tempMax': 'tмакс °C (2020)', 'tempMin': 'tмин °C (2020)', 'tempMean': 'tср °C (2020)', 'speed': 'скорость ветра (2020)', 'fallout': 'осадки (2020)'}, axis=1, inplace=True)
    datav = datav.merge(climate2019df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datav.rename({'tempMax': 'tмакс °C (2019)', 'tempMin': 'tмин °C (2019)', 'tempMean': 'tср °C (2019)', 'speed': 'скорость ветра (2019)', 'fallout': 'осадки (2019)'}, axis=1, inplace=True)
    datav = datav.merge(climate2018df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datav.rename({'tempMax': 'tмакс °C (2018)', 'tempMin': 'tмин °C (2018)', 'tempMean': 'tср °C (2018)', 'speed': 'скорость ветра (2018)', 'fallout': 'осадки (2018)'}, axis=1, inplace=True)
    datav.to_excel('Farming\SborValVetchT.xlsx', index=True)

    #agricultural = agricultural.append(vetch, ignore_index=True)
    agricultural = agricultural.append(datav, ignore_index=True)

    lupine = create_dataframe(lupine2021, lupine2020, lupine2019)
    #lupine.to_excel('Farming\SborValLupine.xlsx', index=True)
    lupine['label']="люпин"
    datalu = lupine.merge(climate2021df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datalu.rename({'tempMax': 'tмакс °C (2021)', 'tempMin': 'tмин °C (2021)', 'tempMean': 'tср °C (2021)', 'speed': 'скорость ветра (2021)', 'fallout': 'осадки (2021)'}, axis=1, inplace=True)
    datalu = datalu.merge(climate2020df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datalu.rename({'tempMax': 'tмакс °C (2020)', 'tempMin': 'tмин °C (2020)', 'tempMean': 'tср °C (2020)', 'speed': 'скорость ветра (2020)', 'fallout': 'осадки (2020)'}, axis=1, inplace=True)
    datalu = datalu.merge(climate2019df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datalu.rename({'tempMax': 'tмакс °C (2019)', 'tempMin': 'tмин °C (2019)', 'tempMean': 'tср °C (2019)', 'speed': 'скорость ветра (2019)', 'fallout': 'осадки (2019)'}, axis=1, inplace=True)
    datalu = datalu.merge(climate2018df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datalu.rename({'tempMax': 'tмакс °C (2018)', 'tempMin': 'tмин °C (2018)', 'tempMean': 'tср °C (2018)', 'speed': 'скорость ветра (2018)', 'fallout': 'осадки (2018)'}, axis=1, inplace=True)
    datalu.to_excel('Farming\SborValLupineT.xlsx', index=True)

    #agricultural = agricultural.append(lupine, ignore_index=True)
    agricultural = agricultural.append(datalu, ignore_index=True)

    sorgo = create_dataframe(sorgo2021, sorgo2020, sorgo2019)
    #sorgo.to_excel('Farming\SborValSorgo.xlsx', index=True)
    sorgo['label']="сорго"
    datas = sorgo.merge(climate2021df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datas.rename({'tempMax': 'tмакс °C (2021)', 'tempMin': 'tмин °C (2021)', 'tempMean': 'tср °C (2021)', 'speed': 'скорость ветра (2021)', 'fallout': 'осадки (2021)'}, axis=1, inplace=True)
    datas = datas.merge(climate2020df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datas.rename({'tempMax': 'tмакс °C (2020)', 'tempMin': 'tмин °C (2020)', 'tempMean': 'tср °C (2020)', 'speed': 'скорость ветра (2020)', 'fallout': 'осадки (2020)'}, axis=1, inplace=True)
    datas = datas.merge(climate2019df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datas.rename({'tempMax': 'tмакс °C (2019)', 'tempMin': 'tмин °C (2019)', 'tempMean': 'tср °C (2019)', 'speed': 'скорость ветра (2019)', 'fallout': 'осадки (2019)'}, axis=1, inplace=True)
    datas = datas.merge(climate2018df[['subjects_of_the_RF', 'tempMax', 'tempMin', 'tempMean', 'speed', 'fallout']])
    datas.rename({'tempMax': 'tмакс °C (2018)', 'tempMin': 'tмин °C (2018)', 'tempMean': 'tср °C (2018)', 'speed': 'скорость ветра (2018)', 'fallout': 'осадки (2018)'}, axis=1, inplace=True)
    datas.to_excel('Farming\SborValSorgoT.xlsx', index=True)

    #agricultural = agricultural.append(sorgo, ignore_index=True)
    agricultural = agricultural.append(datas, ignore_index=True)
    #agricultural.set_index('label', inplace=True)
    agricultural.rename({'subjects_of_the_RF': 'субъект РФ'})
    agricultural.to_excel('Farming\AgriculturalT.xlsx', index=True);

    os.remove("agricultural2021.xls")
    os.remove("agricultural2020.xls")
    os.remove("agricultural2019.xls")

def outlier_removal(dataframe):
    columns=['2021', '2020', '2019', '2018']
    for col in columns:
        # Box Plot
        #sns.boxplot(dataframe[col])
        #plt.show()
        # IQR
        Q1 = np.percentile(dataframe[col], 25, interpolation = 'midpoint')
        Q3 = np.percentile(dataframe[col], 75, interpolation = 'midpoint')
        IQR = Q3 - Q1
        #print("Old Shape: ", dataframe.shape)
        # Above Upper bound
        upper=Q3+1.5*IQR
        upper_array=np.where(dataframe[col]>=upper)
        #Below Lower bound
        lower=Q1-1.5*IQR
        lower_array=np.where(dataframe[col]<=lower)
        # Removing the outliers
        dataframe.drop(upper_array[0], axis = 0, inplace=True)
        dataframe.drop(lower_array[0], axis = 0, inplace=True)
        #print("New Shape: ", dataframe.shape)
        dataframe.reset_index(drop=True, inplace=True)
        #print(dataframe[col], "\n")
        # Box Plot
        #sns.boxplot(dataframe[col])
        #plt.show()
    return dataframe

def create_dataframe(year1, year2, year3):
    peas = pd.DataFrame()
    peas['subjects_of_the_RF']=year1['Unnamed: 0']
    peas['2021']=year1[2021]
    peas['2020']=year2[2020]
    peas['2019']=year3[2019]
    peas['2018']=year3[2018]
    peas.set_index('subjects_of_the_RF', inplace=True)
    delete_col1 = ['РОССИЙСКАЯ ФЕДЕРАЦИЯ', 'ЦЕНТРАЛЬНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-ЗАПАДНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ЮЖНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-КАВКАЗСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'УРАЛЬСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ДАЛЬНЕВОСТОЧНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'г. Москва', 'г. Севастополь']
    delete_col2 = ['РОССИЙСКАЯ ФЕДЕРАЦИЯ', 'ЦЕНТРАЛЬНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-ЗАПАДНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ЮЖНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-КАВКАЗСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ДАЛЬНЕВОСТОЧНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'г. Севастополь']
    delete_col3 = ['РОССИЙСКАЯ ФЕДЕРАЦИЯ', 'ЦЕНТРАЛЬНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ЮЖНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-КАВКАЗСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'УРАЛЬСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'г. Севастополь']
    delete_col4 = ['РОССИЙСКАЯ ФЕДЕРАЦИЯ', 'ЦЕНТРАЛЬНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-ЗАПАДНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ЮЖНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'УРАЛЬСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ']
    delete_col5 = ['РОССИЙСКАЯ ФЕДЕРАЦИЯ', 'ЦЕНТРАЛЬНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ЮЖНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СЕВЕРО-КАВКАЗСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ПРИВОЛЖСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'СИБИРСКИЙ ФЕДЕРАЛЬНЫЙ ОКРУГ', 'ДАЛЬНЕВОСТОЧНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ']
    try:
        peas.drop(delete_col1, axis = 0,  inplace=True) 
    except:
        print("Преобразование прошло неудачно")
    try:
        peas.drop(delete_col2, axis = 0,  inplace=True)
    except:
        print("Преобразование прошло неудачно")
    try:
        peas.drop(delete_col3, axis = 0,  inplace=True)
    except:
        print("Преобразование прошло неудачно")
    try:
        peas.drop(delete_col4, axis = 0,  inplace=True)
    except:
        print("Преобразование прошло неудачно")
    try:
        peas.drop(delete_col5, axis = 0,  inplace=True)
    except:
        print("Преобразование прошло неудачно")
    peas.reset_index(inplace=True)
    #print(peas, "\n");
    peas.fillna(0, inplace=True)
    outlier_removal(peas)
    #peas.fillna(peas.mean(), inplace=True)
    #print(peas.info(), "\n")
    print(peas, "\n")
    return peas

def read_climate_data(files):
    tempMax = []; tempMin = []; tempMean = []; speed = []; precipitation = []
    for file in files:
        excel_sheets = pd.ExcelFile(file, engine='openpyxl').sheet_names
        for sheet in excel_sheets:
            table = pd.read_excel(file, sheet_name=sheet, header=1, engine='openpyxl')
            tempMax.append(round(table['Максимальная'].mean(axis=0), 2))
            tempMin.append(round(table['Минимальная'].mean(axis=0), 2))
            tempMean.append(round(table['Средняя'].mean(axis=0), 2))
            speed.append(round(table['Скорость'].mean(axis=0), 2))
            precipitation.append(round(table['Осадки'].mean(axis=0), 2))
    df = pd.DataFrame()
    df['tempMax'] = tempMax
    df['tempMin'] = tempMin
    df['tempMean'] = tempMean
    df['speed'] = speed
    df['fallout'] = precipitation
    return df

main()




#print("Вывод информации о столбцах таблицы и типах переменных\n")
#print(Belgorod2021.info(), "\n")
#print("статистики по столбцам с количественными переменными\n")
#print(Belgorod2021.describe(), "\n")
#print("ниже выводит список столцов, удобно для копирования\n")
#print(Belgorod2021.columns, "\n")

#tempMax = round(Belgorod2021['Максимальная'].mean(axis=0), 2)
#print('Максимальная температура = ', tempMax, "\n")
#tempMin = round(Belgorod2021['Минимальная'].mean(axis=0), 2)
#print('Минимальная температура = ', tempMin, "\n")
#tempMean = round(Belgorod2021['Средняя'].mean(axis=0), 2)
#print('Средняя температура = ', tempMean, "\n")

#pressure = round(Belgorod2021['Атмосферное'].mean(axis=0), 2)
#print('Атмосферное давление = ', pressure, " мм.рт.ст.\n")
#speed = round(Belgorod2021['Скорость'].mean(axis=0), 2)
#print('Скорость ветра = ', speed, "\n")
#precipitation = round(Belgorod2021['Осадки'].mean(axis=0), 2)
#print('Осадки = ', precipitation, " мм.\n")