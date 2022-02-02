#!/usr/bin/env python
# coding: utf-8

# # SUMMARY
# 
# ## [0 - IMPORT LIBRARIES AND FILES](#ch0)
# 
# ## [I - CORRECTION TEMPERATURE](#ch1)
# 
# ## [II - REMOVE SEASONALITY](#ch2)
# 
# #### [1) Autocorrelation](#ch2-1)
# #### [2) Seasonality](#ch2-2)
# #### [3) Stationarity](#ch2-3)
# #### [4) Partial autocorrelation](#ch2-4)
# 
# ## [III - FORECAST CONSUMPTION](#ch3)
#    #### [1) Holt-Winters](#ch3-1)
#    #### [2) SARIMA](#ch3-2)
#    #### [3) Validation](#ch3-3)

# <a id="ch0"></a>
# # 0 - IMPORT LIBRARIES AND FILES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from math import sqrt

# Linear Regression
import statsmodels.api as sm
from sklearn import preprocessing

# Seasonality
from statsmodels.tsa.seasonal import seasonal_decompose

# Autocorrelation
from statsmodels.graphics import tsaplots

# Stationarity
from statsmodels.tsa.stattools import adfuller

# Holt-Winters
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

# SARIMA
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data = pd.read_csv('data_nettoye')
data.date = pd.to_datetime(data.date)


# <a id="ch1"></a>
# # I - CORRECTION TEMPERATURE

# In[2]:


# Linear Regression

X = data[["dju"]]
y = data["consumption"]

X_assign = X.assign(intercept = [1]*X.shape[0])

lr = sm.OLS(y, X_assign).fit()
print(lr.summary2())


# In[3]:


# Parameters
a, b = lr.params["dju"], lr.params["intercept"]
a, b


# In[4]:


# Plot LR
X = X_assign["dju"]

plt.plot(X,y,"o")
plt.plot(np.arange(min(X), max(X)), [a*x+b for x in np.arange(min(X), max(X))])
plt.title("Diagramme de dispersion consommation, dju", fontsize=14)
plt.xlabel("Dju", fontsize=12)
plt.ylabel("Consommation",  fontsize=12)
plt.savefig("images_soutenance/lr.jpeg", format="jpeg")


# Il y a une corrélation très forte entre les DJU et la consommation.

# In[5]:


# New column with standardized consumption
data["cons_norm"] = (data['consumption']-(data["dju"]*a))
data


# In[6]:


# Plot
plt.figure(figsize=(20,5))
sns.lineplot(data = data, x="date", y="consumption")
sns.lineplot(data = data, x="date", y="cons_norm")
plt.title("Evolution de la consommation selon la date", fontsize="25")
plt.xlabel('Temps (années)', fontsize='19')
plt.ylabel('Consommation (MwH)', fontsize='19')
plt.legend(["Consommation réelle", "Consommation normalisée"])
plt.savefig("images_soutenance/correction_effet_temp.jpeg", format="jpeg")


# <a id="ch2"></a>
# # II - REMOVE SEASONALITY

# In[7]:


desaisonnalisation = data[["date", "cons_norm"]].set_index("date")
desaisonnalisation.head(3)


# <a id="ch2-1"></a>
# ##### Autocorrelation

# In[8]:


fig = tsaplots.plot_acf(desaisonnalisation, lags=30)
plt.savefig("images_soutenance/acf.jpeg", format="jpeg")
plt.show()


# J'observe une saisonnalité claire, prévisible, de 12 mois (c'est quand x=12 que le graphe est le plus autocorrélé).  
# Il semble y avoir une saisonnalité à 3 et 6 mois aussi.
# Les pics à 12 mois étant très hauts, on les analyse tout de même en premier lieu.  

# <a id="ch2-2"></a>
# ##### Seasonality

# In[9]:


seasonal_decompose(desaisonnalisation, model='additive', period=3).plot()
seasonal_decompose(desaisonnalisation, model='additive', period=6).plot()
seasonal_decompose(desaisonnalisation, model='additive', period=12).plot()
plt.savefig("images_soutenance/seasonal_decompose.jpeg", format="jpeg")
plt.show()


# La saisonnalité est parfaite à 3 mois.  
# Cependant, il est logique que la saisonnalité suive le cycle des saisons, sur toute l'année. Je désaisonnalise donc, dans un premier temps, sur 12 mois.

# In[10]:


# Trend with seasonal_decompose

trend = pd.DataFrame(seasonal_decompose(desaisonnalisation, model='additive', 
                                        period=12).trend)

# Trend with rolling

trend['pandas_SMA_12'] = desaisonnalisation.iloc[:,0].rolling(window=12).mean()

trend.head(20)


# In[11]:


trend.pandas_SMA_12.plot()


# <a id="ch2-3"></a>
# ##### Stationarity

# In[12]:


# Dicky-Fuller

result = adfuller(trend['pandas_SMA_12'].dropna())

# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# La pvalue est très supérieure à 5%, la série n'est donc pas stationnaire.

# In[13]:


# Make the series stationnary
data_stationnarise = trend['pandas_SMA_12'].diff().diff(12).dropna()

# Plot the time series
fig, ax = plt.subplots()
data_stationnarise.plot(ax=ax)
plt.show()


# In[14]:


result = adfuller(data_stationnarise)
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# Après transformation, la p-value est très inférieure à 5% : la série est stationnaire.  
# d = 1 et D = 1

# <a id="ch2-4"></a>
# ##### Partial autocorrelation

# In[15]:


# PACF

fig = tsaplots.plot_pacf(desaisonnalisation, lags=30)
plt.savefig("images_soutenance/pacf.jpeg", format="jpeg")
plt.show()


# Pics au niveau de la zone bleutée à 4, 8, 9, 12, 18, 20, 24, 26 et surtout 27 et 29 mois.  
# On observe bien une saisonnalité à 12 mois.  
# Le coefficient p = 0.

# <a id="ch3"></a>
# # III - FORECAST CONSUMPTION

# <a id="ch3-1"></a>
# ##### Holt-Winters

# In[16]:


# HWES needs no seasonality

model_HWES = HWES(data_stationnarise, seasonal_periods=12, trend='add', seasonal='add') 
fitted_HWES = model_HWES.fit()
forecast_HWES = fitted_HWES.forecast(steps=12)


# In[17]:


df_prediction_HWES = pd.DataFrame(forecast_HWES)
df_prediction_HWES.rename(columns={0:"consommation"}, inplace=True)
df_prediction_HWES.index.names = ["date"]


# In[18]:


# Plot HWES

fig = plt.figure()
plt.figure(figsize=(15,8))
plt.xlabel('Date', fontsize=17)
plt.ylabel('Consommation (MwH)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.suptitle('Prédiction de la consommation - modèle HWES')
conso_reelle = plt.plot(data_stationnarise, label='Consommation réelle')
charge_predite_hwes = plt.plot(forecast_HWES, label='Prédiction HWES')
plt.title("Prévision de la consommation avec HWES",fontsize=24)
plt.legend()
plt.savefig("images_soutenance/prediction_hwes.jpeg", format="jpeg")
plt.show()


# <a id="ch3-2"></a>
# ##### SARIMA
# 
# SARIMA tient compte de la tendance et de la saisonnalité : on reprend les données avant stationnarisation et désaisonnalisation, mais avec l'effet température corrigé.

# In[19]:


data_sarima = data[["date", "cons_norm"]].set_index("date")

# Forecast 1 year

results_sarima_auto_1_an = pm.auto_arima(data_sarima, seasonal=True, m=12,d=1, D=1,
                                         start_p=0,
                                         start_q=0, max_p=2, max_q=2, max_P=2, max_Q=2,
                                         trace=True,
                                            error_action='ignore',suppress_warnings=True)

liste_forecast_sarima_auto_1_an = results_sarima_auto_1_an.predict(n_periods=12).tolist()

dernier_mois = data_sarima[-1:].index
dates_forecast = pd.date_range(dernier_mois[0]+relativedelta(months=1), freq='MS', periods=12).tolist()

forecast_sarima_auto_1_an = pd.DataFrame(index = dates_forecast)
forecast_sarima_auto_1_an["prediction"] = liste_forecast_sarima_auto_1_an


# In[20]:


# Plot
fig = plt.figure()
plt.figure(figsize=(15,8))
plt.xlabel('Date', fontsize=17)
plt.ylabel('Consommation (MwH)', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
fig.suptitle('Prédiction de la consommation - modèle SARIMA')
conso_reelle = plt.plot(data_sarima, label='Consommation réelle')
charge_predite_auto_sarima = plt.plot(forecast_sarima_auto_1_an, 
                                                      label='Prédiction SARIMA')
plt.title("Prévision de la consommation avec Auto SARIMA",fontsize=24)
plt.legend()
plt.savefig("images_soutenance/prediction_sarima.jpeg", format="jpeg")
plt.show()


# In[21]:


results_sarima_auto_1_an.plot_diagnostics()
plt.show()


# Résidus standradisés : équilibrés entre le négatif et le positif.
# Bonne adéquation à la droite de Henry.
# Corrélogramme : tous les points sont dans la zone bleutée : good. Les résidus ne sont pas autocorrélés : l'erreur faite au temps t n'est pas corrélée au temps t+1

# <a id="ch3-3"></a>
# ###### Validation

# In[22]:


# Make model HWES and SARIMA and check performance 

nb_mois_test = 12

# Dataset HWES
nb_lignes_hwes = len(data_stationnarise)
data_hwes_train = data_stationnarise.iloc[:nb_lignes_hwes-nb_mois_test]
data_hwes_test = data_stationnarise.iloc[nb_lignes_hwes-nb_mois_test:]

# Model HWES
model_HWES = HWES(data_hwes_train, seasonal_periods=12, trend='add', seasonal='add') 
fitted_HWES = model_HWES.fit()
forecast_HWES = fitted_HWES.forecast(steps=12)

df_prediction_HWES = pd.DataFrame(forecast_HWES)
df_prediction_HWES.rename(columns={0:"consommation"}, inplace=True)
df_prediction_HWES.index.names = ["date"]

# Dataset SARIMA
nb_lignes_sarima = len(data_sarima)
data_sarima_train = data_sarima.iloc[:nb_lignes_sarima-nb_mois_test]
data_sarima_test = data_sarima.iloc[nb_lignes_sarima-nb_mois_test:]

# Model SARIMA
results_sarima_auto_1_an = pm.auto_arima(data_sarima_train, seasonal=True, m=12,d=1, D=0,start_p=0,
                              start_q=0, max_p=2, max_q=2, max_P=2, max_Q=2,trace=True,
                                         error_action='ignore',suppress_warnings=True)

liste_forecast_sarima_auto_1_an = results_sarima_auto_1_an.predict(n_periods=12).tolist()

dernier_mois = data_sarima_train[-1:].index
dates_forecast = pd.date_range(dernier_mois[0]+relativedelta(months=1), freq='MS', periods=12).tolist()

forecast_sarima_auto_1_an = pd.DataFrame(index = dates_forecast)
forecast_sarima_auto_1_an["prediction"] = liste_forecast_sarima_auto_1_an


# In[23]:


# RMSE
rmse_sarima = sqrt(mean_squared_error(data_sarima_test, forecast_sarima_auto_1_an))
rmse_hwes = sqrt(mean_squared_error(data_hwes_test, df_prediction_HWES))

print("Score RMSE SARIMA : {:.2f}.".format(rmse_sarima))
print("Score RMSE Holt-Winters {:.2f}.".format(rmse_hwes))

# MAPE

mape_sarima = mean_absolute_percentage_error(data_sarima_test, 
                                                 forecast_sarima_auto_1_an)
mape_hwes = mean_absolute_percentage_error(data_hwes_test, 
                                                           df_prediction_HWES)

print("Score MAPE SARIMA : {:.2f}.".format(mape_sarima))
print("Score MAPE Holt-Winters {:.2f}.".format(mape_hwes))


# Le score MAPE est plus intéressant car les deux datasets utilisés ne sont pas les mêmes : celui de Holt-Winters étant stationnarisé, les valeurs sont bien plus faibles.  
# On conclut que la modélisation la plus pertinente est SARIMA.

# In[ ]:




