#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


pop = pd.read_csv("pop2007_2017.csv")
ani = pd.read_csv("animal_2017.csv")
veg = pd.read_csv("vegetal_2017.csv")
pib = pd.read_csv("pib.csv")


# In[3]:


# Création d'une colonne variation_population

# Nettoyage du df
pop = pop[["Area Code", "Area", "Year", "Value"]]
pop.columns = ['country_code', "country", "year", "population"]
pop.population = pop.population*1000   # L'unité était en milliers de personnes

# Création d'un pivot_table pour avoir les deux années en colonnes
pop = pop.pivot_table(
                       index = ['country_code', 'country'],
                       columns = ["year"], values=["population"], aggfunc=sum)

# Suppression de l'échelon supérieur, inutile, et conversion en df exploitable
pop.columns = pop.columns.droplevel(0)
pop = pd.DataFrame(pop)
pop.reset_index(inplace=True)

# Création de la variable qui nous intéresse
pop['variation_pop'] = (pop[2017]-pop[2007])/pop[2007]*100
pop.head()


# In[4]:


# Concaténation des bilans animaux et végétaux

# Ajout de la variable origin
ani["origin"] = "animal"
veg["origin"] = "vegetal"

# On regroupe veg et ani en un unique dataframe, via une union
temp = ani.append(veg)

# Suppression de ani et veg
del ani, veg

temp.head()


# In[5]:


# Création du df principal

temp.columns =["xx", "xx1", "country_code", 'country', "xx2", "xx3", 'xx4', "xx5", 'year', "xx6", "element",
              "value", "xx6", "xx7", "origin"]

temp = pd.merge(temp, pop[['country_code', "variation_pop"]], on="country_code", how="left")

data = temp.pivot_table(                       index = ['country_code', 'country', "year", 'variation_pop'],
                       columns = ["element"], values=["value"], aggfunc =sum)

data.columns = ["food_supply_Kcal_cap_day","protein_supply_g_cap_day"]

data.reset_index(inplace=True)
data.head(10)


# In[6]:


# Création du df secondaire, dont on extraira la colonne concernant les protéines animales
# 1er pivot-table
temp = temp.pivot_table(                       index = ['country_code', 'country', "year", 'variation_pop', 'origin'],
                       columns = ["element"], values=["value"], aggfunc=sum)

temp.columns = temp.columns.droplevel(0)
temp.reset_index(inplace=True)
temp.head()


# In[7]:


# Deuxième pivot-table pour créer la colonne concernant les protéines animales par personne et par jour
temp = temp.pivot_table(                       index = ['country_code', 'country', "variation_pop", 'year'],
                       columns = ["origin"], values=["g/personne/jour"], aggfunc=sum)

temp.columns = temp.columns.droplevel(0)
temp.reset_index(inplace=True)
temp.head()


# In[8]:


# Merge et création de la colonne qui nous intéresse (le ratio du total des protéines animales sur le total des
# protéines)
data = pd.merge(data, temp[['country_code', "animal"]], on="country_code")

data.animal = data.animal/data["protein_supply_g_cap_day"]*100 
data.rename(columns={"animal":"ani_protein_ratio"}, inplace=True)
data.head()


# In[9]:


# Ajout du PIB par habitant de chaque pays

data= pd.merge(data, pib[["Code zone", "Valeur"]], left_on="country_code", right_on="Code zone",               how="left")
data.drop(["Code zone"], axis=1, inplace=True)
data.rename(columns={"Valeur":"PIB_hab"}, inplace=True)
data.dropna(inplace=True)


# In[10]:


# Comme au P3, je supprime la ligne correspondant au code 351
data = data.loc[data.country_code!=351]


# In[11]:


# Le df est satisfaisant
data.to_csv("data", index=False)

