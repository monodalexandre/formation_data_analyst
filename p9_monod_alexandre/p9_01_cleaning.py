#!/usr/bin/env python
# coding: utf-8

# # SUMMARY
# 
# ## 0 - [Import libraries and files](#ch0)
# 
# ## 1 - [Data cleaning](#ch1)

# <a id="ch0"></a>
# ## 0 - Import libraries and files

# In[1]:


import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("energie_mensuel.csv", sep=';')
pd.set_option('display.max_columns', None)
data = data.replace('È','é', regex=True) # Les "é" sont écrits comme des "È" dans le 
                                         # fichier initial
data.head(3)


# <a id="ch1"></a>
# ## 1 - Data cleaning

# In[3]:


# Keep "France" rows
data = data.loc[data.Territoire=="France"]
data = data[["Mois", "Consommation totale"]]
data.columns = ["date", "consumption"]

# Datetime format
data.date = pd.to_datetime(data.date)
data


# In[4]:


# France data
plt.figure(figsize=(20,5))
sns.lineplot(data = data, x="date", y="consumption")
plt.legend(bbox_to_anchor=(1, 1), fontsize=13)
plt.tight_layout()
plt.title("Evolution de la consommation selon la date", fontsize="25")
plt.xlabel('Temps (années)', fontsize='19')
plt.ylabel('Consommation (GwH)', fontsize='19')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# L'effet température est nettement visible : ce sont les oscillations des courbes.

# In[5]:


# Chosen DJU : Tours 

# Dataframe from the csv

inputFilename = 'tours.csv'
tempFilename = 'temp.csv'
    
# read the csv file
file = open(inputFilename, 'r')
lines = [line.strip() for line in file]
file.close()

# save a temporary file with the only data we are interested in
file = open(tempFilename, 'w+')
file.write("\n".join(lines[11:]))
file.close()

dju = pd.read_csv(tempFilename, sep=';')

# fix first column title
cols = ['ANNÉE']
cols.extend(dju.columns[1:])
dju.columns = cols

dju.drop(columns=["Total"],inplace=True)
dju


# In[6]:


# Merging the dataframes : changing shape of the second dataframe

# Dictionary month-number of the month

# Months list
liste_mois = dju.columns[1:].tolist()

# Numbers list
liste_indices = []
liste_indices.extend(i+1 for i,j in enumerate(liste_mois))

# Dictionary
dictionnaire_mois = dict(zip(liste_mois, liste_indices))
dictionnaire_mois


# In[7]:


# Turn strings into ints

def conversion_numerique(elt):
    elt = list(elt.split(','))
    elt = ".".join(elt)
    return elt

for i in liste_mois:
    x = map(conversion_numerique, dju[i])
    dju[i]= list(x)
    
dju[liste_mois] = dju[liste_mois].astype(float)


# In[8]:


# Shift months from row to column

dju_reset = pd.melt(dju, id_vars = ["ANNÉE"] , value_vars = liste_mois)
dju_reset.head()


# In[9]:


# Date split (year, month, day)
dju_reset.rename(columns={"ANNÉE":"year", "variable":"month", "value":"dju"}, 
                 inplace=True)
dju_reset['month'] = dju_reset['month'].apply(lambda x: dictionnaire_mois[x])
dju_reset["day"]=1

# Datetime format
dju_reset["date"] = pd.to_datetime(dju_reset[["year", "month", "day"]])
dju_reset.drop(columns = ["year", "month", "day"], inplace=True)
dju_reset.head(3)


# In[10]:


# Merge
data_final = pd.merge(dju_reset, data)

# Date column as first column
data_final = data_final[["date", "dju", "consumption"]]
data_final.sort_values(by="date", inplace=True)

# Nulls deleted
data_final = data_final.loc[(data_final["dju"]!=0) & (data_final["consumption"]!=0)]
data_final


# In[11]:


# Export df
data_final.to_csv("data_nettoye", index=False)


# In[ ]:




