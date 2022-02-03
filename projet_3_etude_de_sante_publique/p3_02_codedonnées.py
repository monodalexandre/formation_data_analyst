#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


# In[2]:


df_population_2013 = pd.read_csv("fichiers_csv/fr_population.csv")
df_population_2011 = pd.read_csv("fichiers_csv/population_2011.csv")

df_population = pd.concat([df_population_2013, df_population_2011])


# In[3]:


df_population.head()


# In[4]:


# Nettoyage de la table
df_population = df_population.drop(columns=["Code Domaine", "Domaine", "Code Élément", "Élément",                                            "Code Produit", "Produit", "Code année", "Unité", "Symbole",                                            "Description du Symbole"])

df_population["Valeur"]=df_population["Valeur"]*1000
df_population.rename(columns={"Valeur":"population"}, inplace=True)
df_population.rename(columns={"Code zone":"country_code", "Zone":"country", "Année":"year"}, inplace=True)

df_population


# In[5]:


df_population.to_csv('df_population.csv')


# 
# ## Nettoyage DF bilan alimentaire
#    
#      

# In[6]:


# Import des librairies
import numpy as np
import pandas as pd

# Import des données des bilans alimentaires
veg = pd.read_csv("fichiers_csv/fr_vegetaux.csv")
ani = pd.read_csv("fichiers_csv/fr_animaux.csv")

veg.head()


# In[7]:




# Concaténation des bilans animaux et végétaux

# Ajout de la variable origin
ani["origin"] = "animal"
veg["origin"] = "vegetal"

# On regroupe veg et ani en un unique dataframe, via une union
temp = ani.append(veg)

# Suppression de ani et veg
del ani, veg

# On renomme les colonnes de temp
temp.columns = ["xx","xx2","country_code","country",'xx3','element'
    ,'item_code','item','xx4',"year","unit","value",'xx5','xx6'
    ,'origin']

# Transformation de temp en table pivot
data_aniveg_2013 = temp.pivot_table(
   index=["country_code","country","item_code","item","year","origin"],
   columns = ["element"], values=["value"], aggfunc=sum)


# In[8]:


# Je réitère avec l'année 2011

# Import des données des bilans alimentaires
veg = pd.read_csv("fichiers_csv/vegetaux_2011.csv")
ani = pd.read_csv("fichiers_csv/animaux_2011.csv")

# Concaténation des bilans animaux et végétaux

# Ajout de la variable origin
ani["origin"] = "animal"
veg["origin"] = "vegetal"

# On regroupe veg et ani en un unique dataframe, via une union
temp = ani.append(veg)

# Suppression de ani et veg
del ani, veg

# On renomme les colonnes de temp
temp.columns = ["xx","xx2","country_code","country",'xx3','element'
    ,'item_code','item','xx4',"year","unit","value",'xx5','xx6'
    ,'origin']

# Transformation de temp en table pivot
data_aniveg_2011 = temp.pivot_table(
   index=["country_code","country","item_code","item","year","origin"],
   columns = ["element"], values=["value"], aggfunc=sum)

df_aniveg = pd.concat([data_aniveg_2013, data_aniveg_2011])

df_aniveg.head()


# In[9]:


# On renomme les colonnes
df_aniveg.columns = ['feed', 'other_uses', 'food_supply_kcal_capita_day',
                'food_supply_quantity_kg_capita_yr','fat_supply_quantity_g_capita_day',
                'protein_supply_quantity_g_capita_day', 'domestic_supply_quantity',
                'export_quantity', 'import_quantity', 'food', 'waste', 'population', 'production',
                'seed', 'processing', 'stock_variation']
df_aniveg=df_aniveg.drop(columns=['population'])
df_aniveg.head()


# In[10]:


df_aniveg.to_csv('df_aniveg.csv')


# # Fichier sous-alimentation

# In[11]:


df_sousalim_2013 = pd.read_csv("fichiers_csv/fr_sousalimentation.csv")
df_sousalim_2011 = pd.read_csv("fichiers_csv/sousalim_2011.csv")

df_sousalim = pd.concat([df_sousalim_2013, df_sousalim_2011])

df_sousalim.head()


# In[12]:


# Je vérifie la redondance de certaines données avant de nettoyer le df

print(df_sousalim['Code Produit'].unique())
print(df_sousalim['Code Élément'].unique())
print(df_sousalim['Code Domaine'].unique())


# In[13]:


# Nettoyage

df_sousalim = df_sousalim.drop(columns=["Code Domaine", "Domaine", "Code Élément", "Élément",                                        "Code Produit", "Produit", "Année", "Unité", "Symbole",                                        "Description du Symbole", "Note"])

# Suppression du doublon Chine

df_sousalim=df_sousalim.loc[df_sousalim['Code zone'] != 351]

df_sousalim.rename(columns={"Code zone":"country_code", "Zone":"country", "Code année":"year",                            "Valeur":"pop_starvation"}, inplace=True)

# Je remplace les valeurs inconnues par des NaN que je vais examiner par la suite
        
df_sousalim.replace("<0.1","NaN",inplace=True)
df_sousalim["pop_starvation"]=pd.to_numeric(df_sousalim["pop_starvation"], errors='coerce')

# Multiplication de la colonne population par 10**6, pour avoir la population réelle
df_sousalim["pop_starvation"]=df_sousalim["pop_starvation"]*(10**6)


# In[14]:


# Je sélectionne les années 2011 et 2013 uniquement

annee=["20122014", "20102012"]

df_sousalim=df_sousalim[df_sousalim.year.isin(annee)]

# Puis je renomme les valeurs des années pour avoir un df plus clair

df_sousalim["year"]=df_sousalim["year"].astype('str')
df_sousalim["year"]=df_sousalim["year"].replace({"20102012":"2011", "20122014":"2013"})


df_sousalim


# ##### Je remarque qu'il y a un certain nombre de NaN, assez nombreux, qui correspondent aux pays dans lesquels il y a moins de 100 000 humains en sous-alimentation. Concernant les pays riches, cela semble indiquer qu'il y a très peu de sous-alimentation, mais concernant certains pays pauvres, africains surtout, cela peut indiquer un chiffre seulement légèrement en-dessous de 100 000. Je vérifie donc par moi-même ces informations sur qqes pays.
# 
# ##### Azerbaïdjan, Burundi, Cuba, Erythrée, Guinée équatoriale, Koweït, Palestine, RDC, Sao-Tomé, Somalie, Tonga
# 
# ##### Afin de vérifier puis corriger les données concernant ces pays je décide de calculer moi-même par un autre moyen la population qui y est, approximativement, en sous-nutrition.
# ##### En pratique, je télécharge les données de "prévalence" concernant tous les pays et je les multiplie par la population. Puis je compare.

# In[15]:


df_sousalim_prevalence = pd.read_csv('fichiers_csv/sousalim_2011_2013_prevalence.csv')

df_sousalim_prevalence = df_sousalim_prevalence.drop(columns=["Code Domaine",                                "Domaine", "Code Élément", "Élément", "Code Produit", "Produit", "Année", "Unité",                                                              "Symbole", "Description du Symbole", "Note"])

df_sousalim_prevalence.rename(columns={"Code zone":"country_code", "Zone":"country", "Code année":"year",                                       "Valeur":"prevalence"}, inplace=True)


df_sousalim_prevalence["year"]=df_sousalim_prevalence["year"].astype('str')
df_sousalim_prevalence["year"]=df_sousalim_prevalence["year"].replace({"20102012":"2011", "20122014":"2013"})

df_sousalim_prevalence.head()


# In[16]:


# Je fais un ".merge" pour avoir les données de la population
    # Pour que le merge se fasse sur l'année aussi, il faut que l'anne du df_population soit du même type que
    # l'année du df prevalence
    
df_population["year"]=df_population["year"].astype('str')

    # Ensuite je peux faire le merge

df_prevalence_pop = pd.merge(df_population, df_sousalim_prevalence, on=["country_code","country","year"])

df_prevalence_pop.head()


# In[17]:


# Je nettoie les valeurs non numériques et je transforme la colonne, qui est en object, en float

df_prevalence_pop.replace("<2.5",'NaN',inplace=True)

df_prevalence_pop["prevalence"]=pd.to_numeric(df_prevalence_pop["prevalence"], errors='coerce')

# Et je multiplie les deux colonnes pour avoir les chiffres 

df_prevalence_pop["pop_starvation"]=df_prevalence_pop["population"]*df_prevalence_pop["prevalence"]/100

# Je crée un df ne contenant que l'année 2013

df_prevalence_pop_2013=df_prevalence_pop.loc[df_prevalence_pop["year"]=="2013"]

pd.set_option('display.max_row', None)
df_prevalence_pop_2013


# ##### Je constate qu'à part Sao Tomé (23 353 personnes), les données sont soit égales à 0, ce qui ne pose pas de pb, soit me donnent des NAN
# 

# In[18]:


# Je vérifie si ces données étaient disponibles au départ

pd.set_option('display.max_row', 1000)
df_sousalim_prevalence


# ###### Je me rends compte que ces données étaient absentes : je ne peux donc corriger que les données de Sao Tomé, avant d'exporter le fichier en csv. Mais il me faut trouver un autre moyen d'obtenir les données des pays qui m'ont donné des NaN.

# In[19]:


df_sousalim['pop_starvation'].loc[df_sousalim.country_code==193]=23353
df_sousalim.to_csv("df_sousalim.csv")


# In[ ]:





# # Je nettoie le fichier cereales

# In[ ]:





# In[20]:


df_cer_2013 = pd.read_csv("fichiers_csv/fr_cereales.csv")
df_cer_2011 = pd.read_csv('fichiers_csv/cereales_2011.csv')

df_cer = pd.concat([df_cer_2013, df_cer_2011])

df_cer.head()


# In[21]:


# Je nettoie

df_cer = df_cer.drop(columns=["Code Domaine", "Domaine", "Code Élément", "Élément",                              "Code année","Unité", "Symbole", "Description du Symbole"])

# Je supprime la Chine

df_cer=df_cer.loc[df_cer['Code zone'] != 351]

# Je veux le poids des céréales en kg et pas en tonnes

df_cer["Valeur"]=df_cer["Valeur"]*10**6 

# Je change les noms des colonnes

df_cer.rename(columns={"Code zone":"country_code", "Zone":"country", "Année":"year",                       "Valeur":"poids_kg", "Produit":"item", "Code Produit":"code_item"}, inplace=True)

df_cer.head()


# In[22]:


# Le fichier cereales me convient, je peux l'exporter

df_cer.to_csv('df_cer.csv')

