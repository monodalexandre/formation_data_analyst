#!/usr/bin/env python
# coding: utf-8

# # Sommaire
# 
# ## [I Vue générale des csv et merge](#ch1)
# 
# ## [II Gestion des -1.0 (colonne price)](#ch2)
# 
# ## [III Vérifications de base](#ch3)
# 
# ## [IV Création d'un df commandes](#ch4)
# 
# ## [V Nettoyage de la colonne "date"](#ch5)
# 

# In[ ]:





# # Importation des 3 fichiers du p4

# In[1]:


import pandas as pd
customers=pd.read_csv("dataset/customers.csv")
products=pd.read_csv("dataset/products.csv")
transactions=pd.read_csv("dataset/transactions.csv")


# <a id="ch1"></a>
# # I Vue generale des csv et merge

# 1) Je m'intéresse d'abord au fichier "products"

# In[2]:


products.head(3)


# In[3]:


products.info()


# In[4]:


products.describe(include='all')


# A priori le -1.0 est une erreur ; mais c'est peut-être aussi un remboursement.

# Je vérifie que chaque produit correspond bien à chaque catégorie, c-à-d que l'indice de chaque produit correspond bien à sa catégorie (que tous les produis de la catégorie 0 commencent par 0, tous ceux de la catégorie 1 commencent par 1, etc...)

# In[5]:


temp0 =products[products.id_prod.str.startswith('0')].loc[products.categ!=0]
temp1 =products[products.id_prod.str.startswith('1')].loc[products.categ!=1]
temp2 =products[products.id_prod.str.startswith('2')].loc[products.categ!=2]

print(temp0, temp1, temp2)


# Les trois df temporaires sont vides : il n'y pas d'erreur de ce côté-là.

# In[6]:


del temp0, temp1, temp2


# 2) je m'intéresse au fichier "customers"

# In[7]:


customers.head(3)


# In[8]:


print(customers.info())


# In[9]:


print(customers.describe(include='all'))
print(customers.sex.unique())


# Il n'y a bien que deux sexes : pas de souci de ce côté-là

# 3) Je m'intéresse au dernier fichier, "transactions"

# In[10]:


transactions.head(3)


# In[11]:


transactions.info()


# In[12]:


transactions.describe(include='all')


# 4) Je fais un merge avec les trois fichiers

# In[13]:


data_temp = pd.merge(transactions, customers)
data = pd.merge(data_temp, products)
data.head()


# In[14]:


data.describe()


# In[15]:


data_temp.count()-data.count()


# Il y a 103 transactions de moins après le merge qu'avant : il y a un pb
# J'ai fait deux merge, pour joindre les 3 csv : je cherche au moment de quel merge des lignes ont été effacées.

# In[16]:


# Suppression de l'actuel dataframe

del data_temp, data

# Je refais le premier merge

data_temp = pd.merge(transactions, customers)
data_temp.describe()


# J'obtiens le même nombre de lignes que pour le csv "transactions" : le pb s'est donc fait au merge suivant, avec "products".

# In[17]:


data = pd.merge(data_temp, products, how="left")
data.describe()


# In[18]:


# Je transforme la colonne birth en colonne age, plus pratique et fonctionnelle

birth_to_age_function = lambda x : 2022-x

data.birth=birth_to_age_function(data.birth)

data=data.rename(columns={"birth":"age"})


# J'ai bien 337 016 lignes, le merge a fonctionné.

# <a id="ch2"></a>
# # II Gestion des -1.0 (colonne 'price')

# Je cherche des informations sur le "-1.0", valeur trouvée dans le describe de "products" et qui est aberrante à première vue. Avec le merge j'ai probablement des informations en ce qui concerne les transactions qui ont "-1.0" comme prix.

# In[19]:


data.loc[data.price== -1.0]


# L'intitulé de la colonne "date" dit qu'il s'agit d'une procédure de test. Je vérifie qu'il n'y a pas d'autres éléments qui ne soient pas des tests dans cette colonne.

# In[20]:


# Sélection des lignes dans lesquelles le mot "test" n'apparaît pas (suppression des lignes contenant "test")
data=data[~data.date.str.contains("test")]

# Vérification de la présence de valeurs négatives qui ne soient pas des tests
data.loc[data.price== -1.0]


# Toutes les valeurs négatives étaient des tests, et sont désormais supprimées.

# <a id="ch3"></a>
# # III Verifications de base
# 
# ### 1) Vérification des doublons

# In[21]:


data[data.duplicated()]


# Il n'y a pas de doublons

# ### 2) Vérification des NaN

# In[22]:


print(pd.isna(data).sum())


# 103 NaN, peut-être sur les mêmes lignes. On retrouve les 103 transactions effacées dans le premier merge.
# Je cherche davantage d'informations sur ces transactions.

# In[23]:


pd.set_option('display.max_rows', 103)
data[data.isna().any(axis=1)]


# In[24]:


pd.reset_option('display.max_rows')


# Il y a un produit spécifique, dont le code est "0_2245", qui semble faire échouer la collecte de données. Je vérifie s'il y a des données néanmoins complètes avec ce produit.

# In[25]:


data.loc[data.id_prod=="0_2245"].count()


# Je retrouve 103 lignes, donc toutes les transactions liées à ce produit sont incomplètes en ce qui concerne la collecte des données. Il s'agit de MAR, soit Missing At Random, des données manquantes qui sont liées à une variable autre que la variable sur laquelle elles manquent : cette variable c'est le code produit ici.  
# Que faire pour compléter ces données ? Faut-il seulement les compléter ?  
# Je remarque d'abord que l'id_prod commence par 0 : ces produits sont donc de la catégorie 0.  
# Ces données me donnent de nombreuses informations utiles (date, heure, client_id, session_id, ...), je vais donc chercher à compléter la colonne manquante (le prix) au lieu de le laisser en NaN. 

# In[26]:


# Calcul du mode et de la médiane du prix de la catégorie 0

print("Mode categ 0.0 : " + str(data.loc[data['categ']==0.0].price.mode()))
print("Mediane categ 0.0 : " + str(data.loc[data['categ']==0.0].price.median()))


# Le mode est trop éloigné de la médiane et de la moyenne et l'écart-type est de 5 euros, ce qui, rapporté à la moyenne ou à la médiane (10€ et 10,65€) est assez important et m'oblige à chercher une autre méthode.  
# La méthode des KNN ne marchera pas, car le tableau contient des colonnes non numériques, qui seraient utiles pour mieux imputer le prix du produit.

# In[27]:


# Je remplace la catégorie NaN par 0

data.categ[data.isna().any(axis=1)]=0


# In[28]:


# Imputation par la médiane :

data.fillna(data.loc[data['categ']==0.0].price.median(), inplace=True)


# In[29]:


data.price.loc[data.id_prod=='0_2245']


# <a id="ch4"></a>
# # IV Création d'un df "commandes"

#   

# In[30]:


# Groupby selon le numéro de transaction
data_commandes = data.groupby(['session_id', 'client_id', "age"]).agg({'date':'max',                                                                       "session_id":"count", 'price':'sum'})
data_commandes.rename(columns={'session_id':"number_items"}, inplace=True)
data_commandes = data_commandes.reset_index()
data_commandes.head(3)


# In[ ]:





# <a id="ch5"></a>
# # V Nettoyage des colonnes 'date'

# In[31]:


# Séparation de "date" en deux colonnes : "date" et "time"
data[["date", "hour"]] = data.date.str.split(" ", expand=True)
data_commandes[["date", "hour"]] = data_commandes.date.str.split(" ", expand=True)

# Suppression des secondes et millièmes de secondes dans "time"
data.hour = pd.to_datetime(data.hour)
data_commandes.hour = pd.to_datetime(data_commandes.hour)
data.hour = data.hour.round("H")
data.hour=data.hour.dt.hour
data_commandes.hour=data_commandes.hour.dt.hour

# Réorganisation de l'ordre des colonnes, pour plus de clarté
data=data[['id_prod', 'categ', 'price', 'client_id', 'sex', 'age', 'session_id', 'date', 'hour']]
data_commandes=data_commandes[['session_id', 'client_id', 'age','price', 'number_items', 'date', 'hour']]
data


# In[32]:


# Passage de la colonne "date" en catégorie date, avec l'import du module datetime
import datetime as dt
data['date']=pd.to_datetime(data['date'])

# Création d'une colonne "day_week" pour étudier la consommation selon le jour de la semaine
data['day_week']=data['date'].dt.dayofweek
days = {0:'Lun',1:'Mar',2:'Merc',3:'Jeu',4:'Ven',5:'Sam',6:'Dim'}
data['day_week'] = data['day_week'].apply(lambda x: days[x])

# Je fais de même pour le df data_commandes

data_commandes['date']=pd.to_datetime(data_commandes['date'])
data_commandes['day_week']=data_commandes['date'].dt.dayofweek
data_commandes['day_week'] = data_commandes['day_week'].apply(lambda x: days[x])

data


# Je transforme aussi le type de la colonne "categ" en category, plus approprié à cette colonne

# In[33]:


data.categ=data.categ.astype("category")


# ### J"exporte data pour l'analyse

# In[34]:


data.to_csv("data", index=False)
data_commandes.to_csv('data_commandes', index=False)


# In[35]:


data.head(3)


# In[36]:


data_commandes.head(3)


# In[ ]:




