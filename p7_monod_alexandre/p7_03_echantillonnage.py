#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from collections import Counter
import json


dataset_bm = pd.read_csv("data_coeff_elasticite/GDIMMay2018.csv")
data = pd.read_csv("data")


# # SOMMAIRE
# 
# ## [I Préparation des données pour la Mission 3](ch1)
# 
# ## [II Mission 3](ch2)
# 
# ### [Réponses aux questions 1 à 6](ch2)
# 
# ### [Réponses aux questions 7 à 11](ch3)

# <a id="ch1"></a>
# ## Préparation des données pour la Mission 3

# In[2]:


raw_data = {"base_cas" : [0.2, 0.4, 0.4, 0.5, 0.66],
            "optimistic_high_mobility":[0.15, 0.3, 0.3, 0.4, 0.5],
            "pessimistic_low_mobility":[0.3, 0.5, 0.5, 0.6, 0.9]}
dataset_continents = pd.DataFrame(raw_data, index = ["nordic_eur_and_canada", "rest_eur", "aus_nzl_usa", "asia",
                                           "latin_am_africa"])


# In[3]:


dataset_continents


# In[4]:


pd.set_option('display.max_columns', None)
dataset_bm.head(2)


# In[5]:


pd.set_option('display.max_columns', None)
dataset_bm = dataset_bm[["iso3","IGEincome", "region"]]
print(dataset_bm.iso3.nunique())
dataset_bm


# In[6]:


# Rajout du Kosovo et de la Syrie, manquants dans le fichier de la BM

line_XKX = {"iso3":"XKX", "region":"Europe & Central Asia"}
line_SYR = {"iso3":"SYR", "region":"Middle East & North Africa"}
dataset_bm = dataset_bm.append(line_XKX, ignore_index=True)
dataset_bm = dataset_bm.append(line_SYR, ignore_index=True)


# Il y a 150 pays dans ce df : il faudra le compléter avec l'autre dataframe.

# In[7]:


dataset_bm[dataset_bm.isna().any(axis=1)].iso3.count()


# In[8]:


# Création d'une liste avec chaque région présente dans le df de la BM

list_regions = dataset_bm.region.unique()
print(list_regions)

# Les régions ne correspondent pas exactement aux catégories de dataset_continents
# Je dois vérifier certaines données

print(dataset_bm.loc[dataset_bm.region == "Europe & Central Asia"].iso3.unique())
dataset_bm.loc[dataset_bm.region == "Middle East & North Africa"].iso3.unique()


# On ne retrouve pas l'Europe occidentale, plus riche et plus homogène quant au coefficient d'élasticité. On pourra donc attribuer à "Europe & Central Asia" le coefficient de "rest_eur".  
# Le groupe de pays dont "Middle East & North Africa" est le plus proche au niveau économique est "rest_eur" : on lui attribuera donc le même coefficient.

# In[9]:


# Liste des coefficients atttribués à chaque groupe
list_coeffs = [0.5, 0.66, 0.4, 0.66, 0.2, 0.5, 0.4]

# Création d'un objet zip à partir des deux listes
dict_prep = zip(list_regions, list_coeffs)

# Création d'un dictionnaire à partir de l'objet zip
dict_countries_coeffs = dict(dict_prep)
dict_countries_coeffs


# In[10]:


dataset_bm.region = dataset_bm.region.map(dict_countries_coeffs)
dataset_bm.rename(columns={"region":"coeff"}, inplace=True)


# In[11]:


dataset_bm.sort_values(by=["iso3", "IGEincome"], ascending=[True, False], inplace=True)
dataset_bm.drop_duplicates(subset="iso3", keep='first', inplace=True)
dataset_bm.rename(columns={"iso3":"country"}, inplace=True)
dataset_bm


# In[12]:


data


# In[13]:


data = pd.merge(dataset_bm, data, how="right")
data


# In[14]:


# Je remplace les NaN par la colonne des coefficients
data["IGEincome"] = data["IGEincome"].fillna(data["coeff"])
data


# In[15]:


data[data["IGEincome"].isna()]


# In[16]:


# Suppression des colonnes inutiles
data.drop(["coeff", "year_survey", "nb_quantiles", "count_val","cluster"], axis=1, inplace=True)

# Je renomme la colonne quantile en c_child
data.rename(columns={"quantile":"c_child"}, inplace=True)

# Observations sur les données

plt.figure(figsize=(10, 3))
sns.boxplot(data=data.IGEincome, showmeans=True, orient="h")
data.describe()


# Le coeff d'élasticité doit-il être compris entre 0 et 1 ?

# <a id="ch2"></a>
# ## MISSION 3
# ### Réponses aux questions 1 à 6

# In[17]:


from mes_fonctions.fonctions_p7 import *
import matplotlib.pyplot as plt


# In[18]:


# Estimations pour un pays aléatoire j

pj = 0.9                 # coefficient d'élasticité du pays j
nb_quantiles = 10       # nombre de quantiles (nombre de classes de revenu)
n  = 1000*nb_quantiles   # taille de l'échantillon

y_child, y_parents = generate_incomes(n, pj)
sample = compute_quantiles(y_child, y_parents, nb_quantiles)
cd = conditional_distributions(sample, nb_quantiles)
#plot_conditional_distributions(pj, cd, nb_quantiles) # Cette instruction prendra du temps si nb_quantiles > 10
print(cd)

c_i_child = 5 
c_i_parent = 8
p = proba_cond(c_i_parent, c_i_child, cd)
print("\nP(c_i_parent = {} | c_i_child = {}, pj = {}) = {}".format(c_i_parent, c_i_child, pj, p))

plot_conditional_distributions(pj, cd, nb_quantiles)


# In[19]:


# Suppression des individus générés
del pj, nb_quantiles, n, y_child, y_parents, sample, cd, c_i_child, c_i_parent, p


# <a id="ch3"></a>
# ### Réponses aux questions 7 à 11

# In[20]:


# CREATION D'UN DICTIONNAIRE ET EXPORT EN JSON : LE CODE MET 30MIN À S'EXÉCUTER
# DONC JE L'AI PASSÉ EN COMMENTAIRE : ON UTILISERA LE FICHIER JSON QUE J'AI CRÉÉ EN
# LA FOIS OÙ J'AI EU LA PATIENCE D'EXÉCUTER CE CODE

# Création d'un dictionnaire avec les coefficients de chaque quantile pour chaque pays

## Etape préliminaire 1 : création d'un dataframe synthétique, avec le pays et les coefficients d'élasticité
#data_pays_et_coeff= data[["country","IGEincome"]].groupby("country").min()
#data_pays_et_coeff = pd.DataFrame(data_pays_et_coeff)
#data_pays_et_coeff.reset_index(inplace=True)
#data_pays_et_coeff

## Etape préliminaire 2 : création d'une liste contenant les listes des ces quantiles
#pj=0
#nb_quantiles = 100      # nombre de quantiles (nombre de classes de revenu)
#n  = 1000*nb_quantiles   # taille de l'échantillon
#liste_cd = []

#for i in data_pays_et_coeff.IGEincome.astype(float):
    #pj=i
    #y_child, y_parents = generate_incomes(n, pj)
    #sample = compute_quantiles(y_child, y_parents, nb_quantiles)
    #cd = conditional_distributions(sample, nb_quantiles)
    #liste_cd.append(cd.tolist())


## Etape préliminaire 3 : création d'une liste contenant les pays correspondants
#liste_pays = list(data_pays_et_coeff.country)
#liste_pays

## Création du dictionnaire
#dictionnaire_pays = dict(zip(liste_pays, liste_cd))

## Exportation du dictionnaire en json
#with open("dictionnary.json", "w") as json_files:
    #json.dump(dictionnaire_pays, json_files)


# In[21]:


# Ouverture du dictionnaire json

with open("dictionnary.json") as dictionnaire_pays:
    dictionnaire_pays = dict(json.loads(dictionnaire_pays.read()))


# In[22]:


# MÊME CHOSE QUE PLUS HAUT : CE CODE MET 4H À S'EXÉCUTER, DONC JE L'AI PASSÉ EN
# COMMENTAIRE : ON UTILISE LE FICHIER QUE J'AI OBTENU QUAND J'AI EU LA PATIENCE DE
# L'EXÉCUTER

# Multiplication des lignes du df
#n_lignes = 1000
#data_cd = pd.concat([data]*n_lignes, ignore_index=True)

# Initialisation d'une colonne parent à 0
#data_cd["c_i_parent"] = 0

## Assimilation à chaque ligne de la classe parent correspondante
## ATTENTION cet algorithme met 4h à s'exécuter !
#for country, cd in dictionnaire_pays.items():
    #for i, row in enumerate(cd):
        #c_parent = []
        #for j, column in enumerate(row):
            #nb_iterations = n_lignes*column
            #c_parent+=[j+1]*int(nb_iterations)
        #data_cd.loc[(data_cd.country==country) & (data_cd.c_child==i+1), "c_i_parent"]=c_parent
        
# Exportation de data en csv

#data_cd.to_csv("data_mission_4.csv", index=False)


# In[23]:


data_cd = pd.read_csv("data_mission_4.csv")

# Vérification : les valeurs de la classe parent se sont-elles placées au bon endroit dans le dataframe,
# au bon endroit selon le pays et le quantile ?

# Codage d'une fonction de vérification
def verification_distribution_conditionnelle(pays, quantile):
    # Liste des valeurs attendues théoriquement
    liste_verif_cd = dictionnaire_pays[pays]
    distribution_conditionnelle_théorique = []
    for i, j in enumerate(liste_verif_cd[quantile-1]):
        nb_iterations = int(1000*j)
        distribution_conditionnelle_théorique+=[i+1]*nb_iterations
        
    # Liste des valeurs du dataframe
    distribution_conditionnelle_pratique = data_cd.loc[(data_cd.country==pays)
                                        & (data_cd.c_child==quantile)].c_i_parent.tolist()

    plt.figure(figsize=(12,8))
    plt.hist(distribution_conditionnelle_théorique, alpha = 0.5, bins = 100, label='Théorique')
    plt.hist(distribution_conditionnelle_pratique, alpha = 0.5, bins=100, label='Pratique')
    plt.legend(loc=1)
    plt.xlabel("Quantiles", fontsize=13)
    plt.ylabel("Nb personnes", fontsize=13)
    plt.title("Vérification théorique/pratique pour {} et le quantile {}".format(pays,quantile), fontsize = 16)

verification_distribution_conditionnelle("USA",3)
plt.savefig("images_soutenance/mission3_verification.jpeg", format="jpeg")


# Superposition parfaite des deux distributions : les valeurs se sont placées au bon endroit, et la mission 3 est validée.

# In[24]:


data_cd.head()


# In[ ]:




