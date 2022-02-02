#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data")
df_best_countries = pd.read_csv("best_countries")
gini_transposed = pd.read_csv("gini_transposed")


# In[2]:


n_clust=7


# In[3]:


# Choix d'un pays par cluster
best_countries =[]
for i in range(0,n_clust):
    best_countries.extend(df_best_countries.loc[df_best_countries.cluster==i].country.sample().tolist())

best_countries


# In[4]:


# Montrez la diversité des pays en termes de distribution de revenus

data_best_countries = df_best_countries[df_best_countries.country.isin(best_countries)]
data_best_countries


# In[5]:


data_best_countries["income_ln"] = np.log(data_best_countries.income)
data_best_countries


# In[6]:


sns.scatterplot(data=data_best_countries, x="quantile", y="income_ln", hue="country")
plt.savefig("images_soutenance/mission2_diversite_distribution_pays.jpeg".format(best_countries[i]), format="jpeg")


# In[20]:


# Représentez la courbe de Lorenz pour chacun des pays choisis

# Codage d'une courbe de Lorenz : la répartition du CA en fonction des clients

plt.figure(figsize=(12,12))
for i in range(0,n_clust):
    data_lorenz = data_best_countries.loc[data_best_countries.country==best_countries[i]].income.values
    n = len(data_lorenz)
    lorenz = np.cumsum(np.sort(data_lorenz)) / data_lorenz.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0
    # ajouter un plt.plot, ou qch de tout simple comme ça
    # Modélisation
    xaxis = np.linspace(0-1/n,1+1/n,n+1) # Suppr
    plt.plot(xaxis,lorenz) # Suppr
plt.title("Répartition de l'income en fonction des quantiles", fontsize='x-large')
plt.xlabel("Quantiles")
plt.ylabel("Income")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(best_countries,borderaxespad=0, fontsize = "small")
plt.savefig("images_soutenance/mission2_lorenz.jpeg".format(best_countries[i]), format="jpeg")


# In[8]:


gini_transposed.head(2)


# In[9]:


test = gini_transposed.drop(index=0)
test.head(2)


# In[10]:


test= test[best_countries]
test


# In[11]:


test = test.astype(float)


# In[12]:


# Représentez l'évolution de l'indice de Gini au fil des ans
sns.set_context("talk", font_scale=1.1)
plt.figure(figsize=(12,12))
sns.lineplot(data = test, palette="tab10")
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.title("Evolution de l'indice de Gini 2004-2011")
plt.xlabel("Années")
plt.ylabel("Indice de Gini")
plt.tight_layout()
plt.savefig("images_soutenance/mission2_evolution_gini.jpeg".format(best_countries[i]), format="jpeg")


# In[13]:


# Classez les pays par indice de Gini
countries_gini = data.groupby("gini").agg({"country":max})
countries_gini = pd.DataFrame(countries_gini)
countries_gini.reset_index(inplace=True)

print("La moyenne des indices de Gini pour tous les pays est {}.".format(countries_gini.mean()))
print("Les cinq pays avec l'indice de Gini le plus fort :")
print(countries_gini.sort_values(by=['gini'], ascending = False).head(5))
print("Les cinq pays avec l'indice de Gini le plus faible :")
print(countries_gini.sort_values(by=['gini'], ascending = True).head(5))
print("La France se trouve à la position :")
print(countries_gini.loc[countries_gini.country=="FRA"])


# In[ ]:




