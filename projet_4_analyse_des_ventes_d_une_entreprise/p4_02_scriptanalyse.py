#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Analyse de données

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt

customers=pd.read_csv("dataset/customers.csv")
products=pd.read_csv("dataset/products.csv")
transactions=pd.read_csv("dataset/transactions.csv")

data=pd.read_csv("data")
data_commandes=pd.read_csv("data_commandes")


# In[2]:


# Je fais un describe général

data.describe()


# In[3]:


# Informations temporelles sur mes données

data.date = pd.to_datetime(data.date)
data_commandes.date = pd.to_datetime(data_commandes.date)

print("Les données sont collectées à partir du",data.date.min(),", jusqu'au", data.date.max())
print("Ce qui fait un total de", data.date.max()-data.date.min(), "jours.")


# # MISSION 2
# 
# ## [I Etude des clients](#ch1)
# 
# 1) Répartition en fonction de l'âge et du sexe
# 
# 2) Mes meilleurs clients
# 
# 3) Le CA en fonction des clients
# 
# 4) Taux d'attrition   
#    
#    
#    
# ## [II Etude des produits](#ch2)
# 
# 1) Répartition du CA selon les produits
# 
# 2) Produits les mieux vendus / les moins vendus
# 
# 3) Répartition des prix en fonction de la catégorie
#    
#    
#    
# ## [III Etude des ventes](#ch3)
# 
# 1) CA annuel
# 
# 2) CA mensuel en fonction des catégories
# 
# 3) CA et fréquence d'achat en fonction des catégories
# 
# 4) CA en fonction de la catégorie et du jour de la semaine
# 
# 5) CA en fonction de la catégorie et de l'heure
# 
# 
# # [MISSION 3](#ch4)
# 
# I Corrélation sexe des clients / catégorie de produits achetés
# 
# II Corrélation âge des clients et ...
# 
#     1) montant total des achats 
#     
#     2) fréquence d’achat (ie. nombre d'achats par mois par exemple)
#     
#     3) taille du panier moyen (en nombre d’articles)
#     
#     4) catégories de produits achetés

# In[ ]:





# <a id="ch1"></a>
# # I Etude des clients

# 1) Répartition en fonction de l'âge et du sexe

# In[4]:


# Création d'un df restreint

clients_age = data[["client_id", "age", "sex"]].groupby("client_id").max()
clients_age = pd.DataFrame(data=clients_age, columns=['age', 'sex'])
clients_age.head(3)


# In[5]:


sns.set_theme(style="darkgrid")

plt.figure(figsize=(15,8))
plt.title("Distribution des clients en fonction de l'âge", fontsize='xx-large')
plt.xlabel('Âge (années)', fontsize='large')
plt.ylabel('Clients (nb)', fontsize='large')
sns.histplot(data=clients_age, x="age", hue="sex", multiple="stack", bins=75)
plt.legend(["Femme", "Homme"], title="Sexe")

plt.savefig("images_soutenance/I1.jpeg", format="jpeg")


# Il y a une répartition identique des clients hommes et des clients femmes en fonction de leur âge.
# Les clients sont donc des hommes et des femmes dont la moitié a plus de 40 ans et l'autre moitié moins de 40 ans, avec la moitié des clients entre 50 et 33 ans environ.  
# La tranche 18 ans est un outlier ; elle peut comprendre tous ceux qui sont trop jeunes pour commander et se sont créés un compte 18 ans, et des gens qui ne souhaitent pas donner leur âge.

# 2) Mes meilleurs clients

# In[6]:


# Je vais déterminer qui sont mes meilleurs clients via le calcul du score RFM

# Création d'un df restreint pour le calcul du RFM
data_rfm = data_commandes[["client_id", "price", "date"]]

# Création d'une date repère à partir de laquelle calculer la Récence et la Fréquence
date_repere = max(data_rfm['date']) + dt.timedelta(days=1)
print(date_repere)


# In[ ]:





# In[7]:


# Groupby pour obtenir la Récence, la Fréquence et le Montant
data_process = data_rfm.groupby(["client_id"]).agg({                                            "date": lambda x : (date_repere-x.max()).days,
                                            "client_id" : 'count',
                                            "price" : "sum"})

data_process.rename(columns={"date": "Recency",
                            "client_id" : "Frequency",
                            "price" : "MonetaryValue"}, inplace=True)

data_process.head(3)


# In[8]:


# Création de listes pour R, F et M, qui leur attribuent 4 valeurs potentielles (calcul d'un RFM qui va
# de 1 à 4)
r_labels = range(4, 0, -1); f_labels = m_labels = range(1, 5)

# Découpage de chaque colonne en quartiles, et attribution à chaque quartile d'une valeur de 1 à 4
r_groups = pd.qcut(data_process['Recency'], q=4, labels=r_labels)
f_groups = pd.qcut(data_process['Frequency'], q=4, labels=f_labels)
m_groups = pd.qcut(data_process['MonetaryValue'], q=4, labels=m_labels)

# Création des nouvelles colonnes R, F et M 
data_process = data_process.assign(R = r_groups, F = f_groups, M = m_groups)
data_rfm = pd.DataFrame(data=data_process)
data_rfm.head()


# In[9]:


# Calcul du score RFM
data_rfm['RFM_Score'] = data_rfm[['R','F','M']].sum(axis=1)
data_rfm['RFM_Score'].head()


# In[10]:


# Modélisation
plt.figure(figsize=(7,5))
sns.histplot(data_rfm['RFM_Score'])
plt.title('Répartition des clients en fonction du score RFM', fontsize='x-large')
plt.xlabel('Score RFM', fontsize='large')
plt.ylabel('Clients (nb)', fontsize='large')

plt.savefig("images_soutenance/I2.jpeg", format="jpeg")


# In[11]:


# Création de deux fichiers meilleurs_clients avec mes dix meilleurs clients : le premier selon le montant total
# et le second selon la fréquence d'achat


# 1 - Les 10 meilleurs clients selon le montant total


# Je classe les clients en fonction du Montant total des achats (et je vérifie que leur score RFM est normal, ce
# que j'estime ici comme étant supérieur à 8)

meilleurs_clients = data_process.sort_values(by=['MonetaryValue'], ascending = False)
meilleurs_clients = meilleurs_clients.head(10)

# Merge pour obtenir les informations sur ces clients
meilleurs_clients = pd.merge(meilleurs_clients, customers, left_on="client_id", right_on="client_id")

# Transformation de la colonne birth en colonne age
birth_to_age_function = lambda x : 2022-x
meilleurs_clients.birth=birth_to_age_function(meilleurs_clients.birth)
meilleurs_clients=meilleurs_clients.rename(columns={"birth":"age"})

meilleurs_clients


# Leur score RFM est bien supérieur à 8 dans chaque cas.

# In[12]:


# 2 - Les 10 meilleurs clients selon la fréquence d'achat

meilleurs_clients_freq = data_process.sort_values(by=['Frequency'], ascending = False)
meilleurs_clients_freq = meilleurs_clients_freq.head(10)
meilleurs_clients_freq

# Merge pour obtenir les informations sur ces clients
meilleurs_clients_freq = pd.merge(meilleurs_clients_freq, customers, left_on="client_id", right_on="client_id")

# Transformation de la colonne birth en colonne age
meilleurs_clients_freq.birth=birth_to_age_function(meilleurs_clients_freq.birth)
meilleurs_clients_freq=meilleurs_clients_freq.rename(columns={"birth":"age"})

meilleurs_clients_freq


# Je constate que mes quatre premiers clients ont fait plus de trois achats par jour en moyenne, ce qui me laisse supposer qu'il ne s'agit pas de clients classiques, mais probablement d'entreprises, peut-être de bibliothèques.  
# Je cherche des informations complémentaires sur ces clients : je vérifie à la main dans un premier temps.

# In[13]:


# Création d'une variable avec les quatre meilleurs clients qui sont dans outliers
clients_verif = meilleurs_clients.head(4).client_id.to_list() 

# Je sélectionne 100 lignes au hasard parmi les achats de ces quatre clients
data[data.client_id.isin(clients_verif)].sample(n=100)


# Je ne constate rien d'anormal. Je garde donc ces clients tels quels pour l'analyse de données.  
# Je les supprimerai pour l'analyse des corrélations car il s'agit d'outliers.

# 3) Le CA en fonction des clients

# In[14]:


# Codage d'une courbe de Lorenz : la répartition du CA en fonction des clients

clients_temp = data_process.MonetaryValue.values
n = len(clients_temp)
lorenz = np.cumsum(np.sort(clients_temp)) / clients_temp.sum()
lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

# Modélisation

fig, ax = plt.subplots(1, figsize=(8, 6))
xaxis = np.linspace(0-1/n,1+1/n,n+1)
ax.plot(xaxis,lorenz,drawstyle='steps-post')
plt.title("Répartition du CA en fonction des clients ", fontsize='x-large')
plt.xlabel("Clients (%)")
plt.ylabel("CA total (%)")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig("images_soutenance/I3.jpeg", format="jpeg")


# In[15]:


# Calcul de l'indice de Gini

AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
S = 0.5 - AUC # surface entre la première bissectrice et la courbe de Lorenz
gini = 2*S
gini


# L'indice de Gini est moyen, légèrement inférieur à 0,50. La répartition du CA en fonction des clients n'est pas préoccupante.  
# J'observe que la loi de Pareto n'est pas respectée : 50% des clients font 20% du CA, et 50% font 80% du CA.  
# J'observe aussi que la fin de la courbe est verticale : cela correspond aux 4 meilleurs clients, qui sont des outliers.

# 4) Taux d'attrition

# In[16]:


# Création d'un df restreint avec les clients qui ont acheté l'an passé
# 1er groupby
clients_recents=data[['client_id', 'sex']].groupby("client_id").max()
sexe_clients_recents=pd.DataFrame(data=clients_recents, columns=['sex'])
# 2e groupby
sexe_clients_recents=sexe_clients_recents.groupby('sex').agg({'sex': "count"})
sexe_clients_recents.head(3)

# Création d'un df restreint avec l'ensemble des clients de la base de données
sexe_clients=customers.groupby('sex').agg({'sex': "count"})

sexe_clients_perdus=sexe_clients-sexe_clients_recents
print("Le taux d'attrition est", sexe_clients_perdus.sum()/sexe_clients.sum()*100, "%")

plt.figure(figsize=(8,8))
plt.pie(
    sexe_clients_perdus,
   labels=sexe_clients_perdus.sex,
    shadow=False,
    startangle=90,
    autopct='%1.1f%%'
    )
plt.title("Les anciens clients en fonction du sexe", fontsize = 'x-large')
plt.legend(["Homme", "Femme"], title="Sexe")

plt.savefig("images_soutenance/I4.jpeg", format="jpeg")


# Le taux d'attrition de cette année n'est pas lié au sexe.

# In[17]:


# J'observe le taux d'attrition en fonction de l'âge

age_clients = customers[["client_id", "birth"]].groupby("client_id").max()
age_clients=pd.DataFrame(data=age_clients, columns=['birth'])

age_clients_recents = data[["client_id", "age"]].groupby("client_id").max()
age_clients_recents=pd.DataFrame(data=age_clients_recents, columns = ['age'])

# Comme lors du nettoyage je transforme la colonne birth en colonne age

age_clients.birth=birth_to_age_function(age_clients.birth)
age_clients=age_clients.rename(columns={"birth":"age"})

age_clients_perdus=age_clients[~age_clients.index.isin(age_clients_recents.index)]

# Je supprime les valeurs qui correspondent à des tests
age_clients_perdus.drop(["ct_1", "ct_0"], inplace=True)

age_clients_perdus.hist()
plt.title("Les anciens clients en fonction de l'âge", fontsize = 'x-large')
plt.xlabel('Âge (années)', fontsize='large')
plt.ylabel('Clients (nb)', fontsize='large')

plt.savefig("images_soutenance/I4-2.jpeg", format="jpeg")


# Le taux d'attrition de cette année n'est pas lié à l'âge.

# <a id="ch2"></a>
# # II Etude des produits
# 
# 1) Répartition du CA selon les produits
# 
# 2) Produits les mieux vendus / les moins vendus
# 
# 3) Répartition des prix en fonction de la catégorie
# 

#   

# 1) Répartition du CA selon les produits

# In[18]:


# Création d'un df restreint
ca_produits = data.groupby(["id_prod", "price"]).agg({                                "price" : "sum"})

ca_produits = ca_produits.rename(columns={"price":"sum_price"})
ca_produits = pd.DataFrame(ca_produits)

ca_produits.head()


# In[19]:


# Codage de la courbe de Lorenz : l'équilibre des ventes selon les produits

ventes_prod = ca_produits['sum_price'].values
n = len(ventes_prod)
lorenz = np.cumsum(np.sort(ventes_prod)) / ventes_prod.sum()
lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0

# Modélisation

fig, ax = plt.subplots(1, figsize=(8, 6))
xaxis = np.linspace(0-1/n,1+1/n,n+1)
ax.plot(xaxis,lorenz,drawstyle='steps-post')
plt.title("Répartition du CA selon les produits", fontsize='x-large')
plt.xlabel("CA total (%)")
plt.ylabel("Produits (%)")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

plt.savefig("images_soutenance/II1.jpeg", format="jpeg")


# In[20]:


# Calcul de l'indice de Gini

AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
S = 0.5 - AUC # surface entre la première bissectrice et la courbe de Lorenz
gini = 2*S
gini


# L'indice de Gini est important, la répartition du CA selon les produits est très inégalitaire : peu de produits rapportent beaucoup, et beaucoup de produits rapportent peu.    
# La loi de Pareto est d'ailleurs respectée : 20% des produits font 80% du CA, et 80% des produits font 20% du CA.

# 2) Produits les mieux vendus / les moins vendus

# In[21]:


# Groupby sur les produits, j'affiche le prix et le nombre d'achats

data_prod = data.groupby('id_prod').agg({"id_prod":"count", 'price':'sum'})
data_prod.rename(columns={"id_prod":"number_purchases", "price":"monetary_value"}, inplace=True)

# Création de variables : les meilleurs et les pires produits en terme de CA et de nb de ventes

meilleurs_prod_montant = data_prod.sort_values(by=['monetary_value'], ascending = False).head(10)
meilleurs_prod_nb_achats = data_prod.sort_values(by=['number_purchases'], ascending = False).head(10)
pire_prod_montant = data_prod.sort_values(by=['monetary_value'], ascending = True).head(10)
pires_prod_nb_achats = data_prod.sort_values(by=['number_purchases'], ascending = True).head(10)

# Affichage de ces variables

meilleurs_prod_montant, meilleurs_prod_nb_achats, pire_prod_montant, pires_prod_nb_achats


# In[22]:


pire_prod_montant


# 3) Répartition des prix en fonction de la catégorie

# In[23]:


# Je change la catégorie du df "products" en "category", pour avoir trois boxplots différents sur ma modélisation,
# sur l'axe des Y

products.categ=products.categ.astype("category")

# Représentation graphique de la moyenne
meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick', "markersize":'15'}

plt.figure(figsize=(40,15))
sns.boxplot(y="categ", x="price",data=products, showmeans=True, meanprops=meanprops)
plt.title('Répartition des prix en fonction de la catégorie', fontsize='43')
plt.xlabel('Prix', fontsize='30')
plt.ylabel("Catégories", fontsize='30')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.savefig("images_soutenance/II3.jpeg", format="jpeg")


# <a id="ch3"></a>
# # III Etude des ventes
# 
# 1) CA annuel
# 
# 2) CA mensuel en fonction des catégories
# 
# 3) Les ventes des différentes catégories en fonction du prix
# 
# 4) CA et fréquence d'achat en fonction des catégories
# 
# 5) CA en fonction de la catégorie et du jour de la semaine
# 
# 6) CA en fonction de la catégorie et de l'heure

#   

# 1) CA annuel

# In[24]:


CA_2021_2022 = data.price.sum()
print("Le CA de mars 2021 à mars 2022 est de", CA_2021_2022, "€")


# 2) CA mensuel en fonction des catégories

# In[25]:


# Création d'une colonne 'mois' dans data
data["month"]=data.date.dt.month

# Affichage du CA mensuel en fonction des catégories
data_CA_cat = data.groupby(["month", "categ"]).agg({"price" : "sum"})
data_CA_cat = pd.DataFrame(data_CA_cat)
data_CA_cat.reset_index(inplace=True)

# Affichage des mois dans l'ordre
months = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Avr', 5:'Mai', 6:'Juin', 7:'Juil', 8:'Aout',
         9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

data_CA_cat['month'] = data_CA_cat['month'].apply(lambda x: months[x])

# Création du seuil entre la cétégorie 1 et 2, pour l'affichage du graphique
seuil_categ2=data_CA_cat.price.loc[data_CA_cat.categ==0.0].values    +data_CA_cat.price.loc[data_CA_cat.categ==1.0].values

# Modélisation
plt.figure(figsize=(15,8))

plt.bar(data_CA_cat.month.loc[data_CA_cat.categ==0.0],        data_CA_cat.price.loc[data_CA_cat.categ==0.0],         label="0")
plt.bar(data_CA_cat.month.loc[data_CA_cat.categ==1.0],        data_CA_cat.price.loc[data_CA_cat.categ==1.0],        bottom=data_CA_cat.price.loc[data_CA_cat.categ==0.0],        label="1")
plt.bar(data_CA_cat.month.loc[data_CA_cat.categ==2.0],        data_CA_cat.price.loc[data_CA_cat.categ==2.0],        bottom= seuil_categ2,        label="2")

plt.legend(title="Catégorie")
plt.title("CA mensuel en fonction de la catégorie", fontsize = 'xx-large')
plt.xlabel("Mois")
plt.ylabel('CA (€)')

plt.savefig("images_soutenance/III2.jpeg", format="jpeg")


# In[26]:


# Restriction à un df contenant uniquement la catégorie 1, pour les mois de septembre à novembre
# Puis étude des achats sur cette période (modélisation)

data_CA_cat = data.loc[(data.month.isin([9, 10, 11])) & (data.categ==1.0)]
data_CA_cat = data_CA_cat.groupby("date").agg({"price":"sum"})
data_CA_cat = pd.DataFrame(data_CA_cat)
data_CA_cat.reset_index(inplace=True)

plt.bar(data_CA_cat.date,        data_CA_cat.price,         )
plt.xticks(fontsize=7)
plt.title("CA de la catégorie 1 de septembre à novembre", fontsize = "large")
plt.xlabel("Date")
plt.ylabel("CA")


# In[27]:


# Recherche des dates exactes de la période problématique

date_début = data_CA_cat.loc[data_CA_cat.date<"2021-10-15"].max().date
date_fin = data_CA_cat.loc[data_CA_cat.date>"2021-10-15"].min().date
print("Les achats de produits de catégorie 1 s'arrêtent le {} et reprennent le {}.".format(date_début, date_fin))


# Il y a un arrêt total des achats de catégorie 1 entre le 1er et le 28 octobre ; soit 27 jours vides.  
# On verra plus loin que tous les âges, et les deux sexes, achètent dans cette catégorie 1. Ce n'est donc pas un pb client.  
# Deux hypothèses s'imposent : un arrêt des ventes de catégorie 1 à ce moment ; ou plus probablement un bug informatique qui a empêché d'obtenir les données de ventes qui ont bien eu lieu ; d'autant plus probable que les ventes de catégorie 1 reprennent au niveau où elles s'étaient arrêtées 27 jours plus tôt.
# 
# Si cette situation arrivait en pratique, il faudrait envoyer un mail à celui ou celle qui nous a transmis les données pour avoir des informations complémentaires, avant d'éventuellement imputer par la médiane s'il y a bien eu bug informatique et perte de données. Dans le cas présent, n'ayant pas cette confirmation, on n'impute pas.

# 3) CA et fréquence d'achat en fonction des catégories

# In[28]:


# Création d'un sous-df avec un groupby sur la colonne "categ"

data_categ = data.groupby(["categ"]).agg({"client_id" : 'count',
                                            "price" : "sum"})

data_categ.rename(columns={"client_id" : "total_purchases",
                            "price" : "monetary_value"}, inplace=True)

data_categ=pd.DataFrame(data_categ)
data_categ.reset_index(inplace=True)

# Modélisation
sns.set()
plt.figure(figsize=(8,8))
plt.pie(
    data_categ.monetary_value,
    labels=data_categ.categ,
    shadow=False,
    startangle=90,
    autopct='%1.1f%%'
    )
plt.title("CA en fonction des catégories", fontsize = 'x-large')
plt.savefig("images_soutenance/III4-1.jpeg", format="jpeg")
plt.show()

plt.figure(figsize=(8,8))
plt.pie(
    data_categ.total_purchases,
    labels=data_categ.categ,
    shadow=False,
    startangle=90,
    autopct='%1.1f%%'
    )
plt.title("Fréquence d'achat selon les catégories", fontsize = 'x-large')
plt.savefig("images_soutenance/III4-2.jpeg", format="jpeg")


# 4) CA en fonction de la catégorie et du jour de la semaine

# In[29]:


# Calcul du CA en fonction du jour de la semaine, selon les différentes catégories 

# Création d'un sous-df 
data_jour=data.groupby(['day_week', "categ"]).agg({'price':'sum'})
data_jour=pd.DataFrame(data_jour)
data_jour.reset_index(inplace=True)

# Tri des jours dans l'ordre de la semaine (sinon ils s'affichent par ordre alphabétique)
from pandas.api.types import CategoricalDtype
cat_days_order = CategoricalDtype(                                 ['Lun', 'Mar', 'Merc', 'Jeu', 'Ven', 'Sam', 'Dim'],
                                 ordered = True)
data_jour.day_week = data_jour.day_week.astype(cat_days_order)
data_jour.sort_values('day_week', inplace=True)

# Création du seuil entre la cétégorie 1 et 2, pour l'affichage du graphique
seuil_categ2=data_jour.price.loc[data_jour.categ==0.0].values    +data_jour.price.loc[data_jour.categ==1.0].values

# Modélisation
plt.figure(figsize=(15,8))

plt.bar(data_jour.day_week.loc[data_jour.categ==0.0],        data_jour.price.loc[data_jour.categ==0.0],         label="0")
plt.bar(data_jour.day_week.loc[data_jour.categ==1.0],        data_jour.price.loc[data_jour.categ==1.0],        bottom=data_jour.price.loc[data_jour.categ==0.0],        label="1")
plt.bar(data_jour.day_week.loc[data_jour.categ==2.0],        data_jour.price.loc[data_jour.categ==2.0],        bottom= seuil_categ2,        label="2")

plt.legend(title="Catégorie")
plt.title("CA en fonction du jour de la semaine selon les catégories", fontsize = 'xx-large')
plt.xlabel("Jours", fontsize='15')
plt.ylabel('CA (€)', fontsize='15')
plt.legend(title="Catégorie", loc='best')

plt.savefig("images_soutenance/III5.jpeg", format="jpeg")


# 5) CA en fonction de la catégorie et de l'heure

# In[30]:


# Je fais de même pour avoir le CA en fonction de l'heure d'achat

data_heure=data.groupby(['hour', "categ"]).agg({'price':'sum'})
data_heure=pd.DataFrame(data_heure)
data_heure.reset_index(inplace=True)
seuil_categ2=data_heure.price.loc[data_heure.categ==0.0].values    +data_heure.price.loc[data_heure.categ==1.0].values

# Création du seuil entre la cétégorie 1 et 2, pour l'affichage du graphique
seuil_categ2=data_heure.price.loc[data_heure.categ==0.0].values    +data_heure.price.loc[data_heure.categ==1.0].values


# Modélisation
plt.figure(figsize=(15,8))

plt.bar(data_heure.hour.loc[data_heure.categ==0.0],        data_heure.price.loc[data_heure.categ==0.0],         label="0")
plt.bar(data_heure.hour.loc[data_heure.categ==1.0],        data_heure.price.loc[data_heure.categ==1.0],        bottom=data_heure.price.loc[data_heure.categ==0.0],        label="1")
plt.bar(data_heure.hour.loc[data_heure.categ==2.0],        data_heure.price.loc[data_heure.categ==2.0],        bottom= seuil_categ2,        label="2")

plt.legend(title="Catégorie")
plt.title("CA en fonction de l'heure selon les catégories", fontsize = 'xx-large')
plt.xlabel("Heure")
plt.ylabel('CA (€)')
plt.xlabel("Heures", fontsize='15')
plt.ylabel('CA (€)', fontsize='15')
plt.legend(title="Catégorie", loc='best')

plt.savefig("images_soutenance/III6.jpeg", format="jpeg")


# Le prix des commandes est très également réparti selon les jours de la semaine...

# In[31]:


del data_jour, data_heure


# <a id="ch4"></a>
# # MISSION 3
# 
# I Corrélation sexe des clients / catégorie de produits achetés
# 
# II Corrélation âge des clients et ...
# 
#     1) montant total des achats 
#     
#     2) fréquence d’achat (ie. nombre d'achats par mois par exemple)
#     
#     3) taille du panier moyen (en nombre d’articles)
#     
#     4) catégories de produits achetés

# In[32]:


# Je masque les quatre outliers rencontrés en I-2, et les valeurs qui ont pour âge "18"
data=data.mask(data.client_id.isin(clients_verif))
data=data.mask(data.age==18)

data_commandes=data_commandes.mask(data_commandes.client_id.isin(clients_verif))
data_commandes=data_commandes.mask(data_commandes.age==18)


# I Corrélation sexe des clients / catégorie de produits achetés

# In[33]:


# Réalisation d'une heatmap

# Tableau de contingence réel
data.categ=data.categ.astype("float64")

X="categ"
Y="sex"
d=data[[X,Y]].pivot_table(index=X, columns=Y, aggfunc=len)
d.head(3)


# In[34]:


cont=d.copy()
tx=data[X].value_counts()
ty=data[Y].value_counts()

cont.loc[:,"Total"]=tx
cont.loc["total",:]=ty
cont.loc["total","Total"]=len(data)
cont


# In[35]:


# Tableau de contingence théorique

tx = pd.DataFrame(tx)
ty = pd.DataFrame(ty)
tx.columns= ['foo']
ty.columns= ['foo']
n = len(data)
indep = tx.dot(ty.T)/n

indep.sort_index(axis=1, inplace=True)
indep.sort_index(inplace=True)
indep


# In[36]:


mesure = (d-indep)**2/indep
xi_n = mesure.sum().sum()

plt.figure(figsize=(8,8))
sns.heatmap(mesure/xi_n, annot=indep - d)
plt.title("Corrélation entre la catégorie et le sexe", fontsize = 'x-large')
plt.xlabel("Sexe")
plt.ylabel("Catégorie")

plt.savefig("images_soutenance/IV1.jpeg", format="jpeg")


# Il y a une incidence moyenne du sexe sur la catégorie concernant la catégorie 0, une incidence faible concernant la catégorie 1, et une incidence très faible concernant la catégorie 2.

# II Corrélation âge des clients et ...
# 
#     1) montant total des achats 

# Nous verrons plus bas que les trois variables étudiées en fonction de l'âge ont un coefficient de Pearson très bas ; de plus les modélisations montrent une répartition non linéaire. La régression linéaire n'est donc pas adaptée.  
# Le graphe "Age, panier moyen" fait distinctement apparaître trois catégories d'âge qui ont des comportements différents. Il faut séparer les âges en ces trois catégories, et réétudier les trois paramètres (montant total, fréquence d'achat, panier moyen) avec une ANOVA.  
# Pour une meilleure lisbilité, les ANOVA seront faites à la suite de chaque régression linéaire.

# In[37]:


# Groupby sur l'id_client pour avoir le montant total de leurs achats
data_montant_clients = data.groupby(["client_id", "age"]).agg({"price": "sum"})
data_montant_clients = data_montant_clients
data_montant_clients = pd.DataFrame(data_montant_clients)
data_montant_clients.reset_index(inplace=True)

# Diagramme de dispersion
X = data_montant_clients["age"]
y = data_montant_clients["price"]

plt.figure(figsize=(15,8))
plt.scatter (X,y)

plt.title("Diagramme de dispersion age, montant total des achats", fontsize="xx-large")
plt.xlabel("Âge (années)")
plt.ylabel("Montant total des achats (€)")


# In[38]:


import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[39]:


# Calcul du coefficient de Pearson
r, p_value = st.pearsonr(X, y)
print("Le coefficient de corrélation r est de : {}".format(r))
print("Sa p-value est de : {}".format(p_value))


# Les deux variables sont faiblement corrélées (coefficient de corrélation << 0.50)

# In[40]:


# Calcul de la régression linéaire

X = data_montant_clients[["age"]]
y = data_montant_clients["price"]

X = X.assign(intercept = [1]*X.shape[0])

lr = sm.OLS(y, X).fit()
print(lr.summary2())

# Affichage de a et b, coefficients de ma droite de régression linéaire
print(lr.params)


# In[41]:


# Stockage de a et b dans les variables correspondantes
a,b = lr.params['age'], lr.params['intercept']

# Affichage du nuage de points avec la droite de régression linéaire
X = data_montant_clients['age']
y = data_montant_clients['price']
plt.figure(figsize=(15,8))
plt.plot(X, y, "o")
plt.plot(np.arange(min(X), max(X)), [a*x+b for x in np.arange(min(X), max(X))])
plt.title("Diagramme de dispersion age, montant total des achats",  fontsize = 'xx-large')
plt.xlabel("Âge (années)")
plt.ylabel("Montant total des achats (€)")

plt.savefig("images_soutenance/IV2-1.jpeg", format="jpeg")


# Comme dit plus haut, la régression linéaire n'est pas adaptée : on crée trois catégories d'âge (d'après les observations de la troisième régression linéaire) et on fait une ANOVA.

# In[42]:


# Création de trois catégories d'âge dans les deux df principaux

categ_age = []

for row in data.age:
    if row > 50:
        categ_age.append(">50")
    elif row > 30:
        categ_age.append("30-50")
    elif row > 18:
        categ_age.append("19-29")
    else:
        categ_age.append(np.NaN)

data['categ_age'] = categ_age

categ_age_commandes = []

for row in data_commandes.age:
    if row > 50:
        categ_age_commandes.append(">50")
    elif row > 30:
        categ_age_commandes.append("30-50")
    elif row > 18:
        categ_age_commandes.append("19-29")
    else:
        categ_age_commandes.append(np.NaN)

data_commandes['categ_age'] = categ_age_commandes

# Tri des âges par ordre croissant (sinon ils ils s'affichent dans n'importe quel ordre)

categ_age_order = CategoricalDtype(["19-29", "30-50", ">50"], ordered = True)
data.categ_age = data.categ_age.astype(categ_age_order)
data.sort_values("categ_age", inplace=True)
data_commandes.categ_age = data_commandes.categ_age.astype(categ_age_order)
data_commandes.sort_values("categ_age", inplace=True)


# In[43]:


# Groupby sur l'id_client pour avoir le montant total de leurs achats
data_montant_clients = data.groupby(["client_id", "categ_age"], observed = True).agg({"price": "sum"})
data_montant_clients = data_montant_clients
data_montant_clients = pd.DataFrame(data_montant_clients)
data_montant_clients.reset_index(inplace=True)

# Modélisation des boxplots

X = "price"
Y = "categ_age" 

meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}

plt.figure(figsize=(20, 6))
sns.boxplot(x=data_montant_clients[X], y=data_montant_clients[Y], showmeans=True, meanprops=meanprops, orient="h")
plt.title("Montant total en fonction des catégories d'âge", fontsize = '25')
plt.xlabel("Montant")
plt.ylabel("Catégorie d'âge")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig("images_soutenance/IV3-1.jpeg", format="jpeg")


# In[44]:


# Calcul de l'ANOVA

from statsmodels.formula.api import ols

linear_model = ols('price ~ C(categ_age)', data=data_montant_clients).fit()
table = sm.stats.anova_lm(linear_model, typ=2)
print(table)


# In[45]:


pvalue = linear_model.f_pvalue
print('Ici la p-value est égale à {}'.format(pvalue))

import math
rsquared = math.sqrt(linear_model.rsquared)
print('Ici le coefficient de corrélation est égal à {}'.format(rsquared))


# In[46]:


# Affichage de la régression linéaire correspondante
print(linear_model.summary2())


# In[47]:


# P-value : choix d'un seuil de probabilité standard, de 5%
seuil = 0.05

# Coeff de corrélation : A partir du tableau d'interprétation de Cohen (https://en.wikipedia.org/wiki/Effect_size),
# se dégagent trois seuils selon l'importance de la corrélation
correl_petit = 0.1
correl_moyen = 0.3
correl_grand = 0.5

#Conclusion
if pvalue < seuil:
    if rsquared > correl_grand:
        print("On rejette HO : il y a une corrélation forte entre le montant total des achats et la catégorie d'âge.")
    elif rsquared > correl_moyen:
        print("On rejette HO : il y a une corrélation moyenne entre le montant total des achats et la catégorie d'âge.")
    elif rsquared > correl_petit:
        print("On rejette HO : il y a une petite corrélation entre le montant total des achats et la catégorie d'âge.")
elif pvalue > seuil : 
    print("On ne peut pas rejeter HO: il n'y a pas de corrélation entre le montant total des achats et la catégorie d'âge")


#     2) fréquence d’achat (ie. nombre d'achats par mois par exemple)

# In[48]:


# Groupby sur l'id_client pour avoir le montant total de leurs achats

data_freq = data_commandes.groupby(["client_id","age"]).agg({"session_id": "count"})
data_freq = pd.DataFrame(data_freq)
data_freq = data_freq.rename(columns={"session_id":"nb_achats"})

nb_mois_annee = 12
data_freq.nb_achats = data_freq.nb_achats/nb_mois_annee
data_freq.reset_index(inplace=True)


# Diagramme de dispersion
X = data_freq["age"]
y = data_freq["nb_achats"]

plt.figure(figsize=(15,8))
plt.scatter (X,y)

plt.title("Diagramme de dispersion age, fréquence des achats", fontsize = 'xx-large')
plt.xlabel("Âge (années)")
plt.ylabel("Fréquence des achats (par an)")


# In[49]:


# Calcul du coefficient de Pearson

r, p_value = st.pearsonr(X, y)
print("Le coefficient de corrélation r est de : {}".format(r))
print("Sa p-value est de : {}".format(p_value))


# Les deux variables sont faiblement corrélées (coefficient de corrélation << 0.50)

# In[50]:


# Calcul de la régression linéaire
X = data_freq[["age"]]
y = data_freq["nb_achats"]

X = X.assign(intercept = [1]*X.shape[0])

lr = sm.OLS(y, X).fit()
print(lr.summary2())

# Affichage de a et b, coefficients de ma droite de régression linéaire
print(lr.params)


# In[51]:


# Stockage de a et b dans les variables correspondantes
a,b = lr.params['age'], lr.params['intercept']

# Affichage du nuage de points avec la droite de régression linéaire
X = data_freq['age']
y = data_freq['nb_achats']
plt.figure(figsize=(15,8))
plt.plot(X, y, "o")
plt.plot(np.arange(min(X), max(X)), [a*x+b for x in np.arange(min(X), max(X))])
plt.title("Diagramme de dispersion age, fréquence des achats", fontsize = 'xx-large')
plt.xlabel("Âge")
plt.ylabel("Fréquence des achats")

plt.savefig("images_soutenance/IV2-2.jpeg", format="jpeg")


# La régression linéaire n'est pas adaptée, il faut faire une ANOVA.

# In[52]:


# Groupby sur l'id_client pour avoir la fréquence d'achat

data_freq = data_commandes.groupby(["client_id","categ_age"], observed = True).agg({"session_id": "count"})
data_freq = pd.DataFrame(data_freq)
data_freq = data_freq.rename(columns={"session_id":"nb_achats"})

nb_mois_annee = 12
data_freq.nb_achats = data_freq.nb_achats/nb_mois_annee
data_freq.reset_index(inplace=True)


# Modélisation des boxplots

X = "nb_achats" 
Y = "categ_age"

meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}

plt.figure(figsize=(20, 6))
sns.boxplot(x=data_freq[X], y=data_freq[Y], showmeans=True, meanprops=meanprops, orient="h")
plt.title("Fréquence des achats, par mois, en fonction des catégories d'âge", fontsize = '25')
plt.xlabel("Nombre d'achats par mois")
plt.ylabel("Catégorie d'âge")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig("images_soutenance/IV3-2.jpeg", format="jpeg")


# In[53]:


# Calcul de l'ANOVA

linear_model = ols('nb_achats ~ C(categ_age)', data=data_freq).fit()
table = sm.stats.anova_lm(linear_model, typ=2)
print(table)


# In[54]:


pvalue = linear_model.f_pvalue
print('Ici la p-value est égale à {}'.format(pvalue))

rsquared = math.sqrt(linear_model.rsquared)
print('Ici le coefficient de corrélation est égal à {}'.format(rsquared))


# In[55]:


# Affichage de la régression linéaire correspondante
print(linear_model.summary2())


# In[56]:


seuil = 0.05

#Conclusion
if pvalue < seuil:
    if rsquared > correl_grand:
        print("On rejette HO : il y a une corrélation forte entre la fréquence des achats et la catégorie d'âge.")
    elif rsquared > correl_moyen:
        print("On rejette HO : il y a une corrélation moyenne entre la fréquence des achats et la catégorie d'âge.")
    elif rsquared > correl_petit:
        print("On rejette HO : il y a une petite corrélation entre la fréquence des achatset la catégorie d'âge.")
elif pvalue > seuil : 
    print("On ne peut pas rejeter HO: il n'y a pas de corrélation entre la fréquence des achats et la catégorie d'âge")


#     3) taille du panier moyen (en nombre d’articles)

# In[57]:


# Groupby sur l'age pour avoir le nombre d'achats moyen par panier

data_panier_moyen = data_commandes.groupby(['client_id',"age"]).agg({"number_items": "mean"})
data_panier_moyen = pd.DataFrame(data_panier_moyen)
data_panier_moyen = data_panier_moyen.rename(columns={"number_items": "panier_moyen"})

data_panier_moyen.reset_index(inplace=True)


# Diagramme de dispersion
X = data_panier_moyen["age"]
y = data_panier_moyen["panier_moyen"]

plt.figure(figsize=(15,8))
plt.scatter (X,y)

plt.title("Diagramme de dispersion Age, Panier moyen", fontsize = 'xx-large')
plt.xlabel("Âge (années)")
plt.ylabel("Panier moyen (articles)")


# In[58]:


# Calcul du coefficient de Pearson

r, p_value = st.pearsonr(X, y)
print("Le coefficient de corrélation r est de : {}".format(r))
print("Sa p-value est de : {}".format(p_value))


# Les deux variables sont faiblement corrélées (coefficient de corrélation << 0.50)

# In[59]:


# Calcul de la régression linéaire
X = data_panier_moyen[["age"]]
y = data_panier_moyen["panier_moyen"]

X = X.assign(intercept = [1]*X.shape[0])

lr = sm.OLS(y, X).fit()
print(lr.summary2())

# Affichage de a et b, coefficients de ma droite de régression linéaire
print(lr.params)


# In[60]:


# Stockage de a et b dans les variables correspondantes
a,b = lr.params['age'], lr.params['intercept']

# Affichage du nuage de points avec la droite de régression linéaire
X = data_panier_moyen['age']
y = data_panier_moyen['panier_moyen']
plt.figure(figsize=(15,8))
plt.plot(X, y, "o")
plt.plot(np.arange(min(X), max(X)), [a*x+b for x in np.arange(min(X), max(X))])
plt.title("Diagramme de dispersion Age, Panier moyen", fontsize = 'xx-large')
plt.xlabel("Âge (années)")
plt.ylabel("Panier moyen (articles)")

plt.savefig("images_soutenance/IV2-3.jpeg", format="jpeg")


# On distingue les trois catégories d'âge utilisées pour les ANOVA. On fait une ANOVA aussi.

# In[61]:


# Groupby sur l'id_client pour avoir le panier moyen

data_panier_moyen = data_commandes.groupby(['client_id',"categ_age"], observed = True).agg({"number_items": "mean"})
data_panier_moyen = pd.DataFrame(data_panier_moyen)
data_panier_moyen = data_panier_moyen.rename(columns={"number_items": "panier_moyen"})

data_panier_moyen.reset_index(inplace=True)

# Modélisation des boxplots

X = "panier_moyen" 
Y = "categ_age"

meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}

plt.figure(figsize=(20, 6))
sns.boxplot(x=data_panier_moyen[X], y=data_panier_moyen[Y], showmeans=True, meanprops=meanprops, orient="h")
plt.title("Taille moyenne du panier en fonction des catégories d'âge", fontsize = '25')
plt.xlabel("Taille moyenne du panier")
plt.ylabel("Catégorie d'âge")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.savefig("images_soutenance/IV3-3.jpeg", format="jpeg")


# In[62]:


# Calcul de l'ANOVA
data_panier_moyen.panier_moyen = data_panier_moyen.panier_moyen.astype("float64")

linear_model = ols('panier_moyen ~ C(categ_age)', data=data_panier_moyen).fit()
table = sm.stats.anova_lm(linear_model, typ=2)
print(table)


# In[63]:


pvalue = linear_model.f_pvalue
print('Ici la p-value est égale à {}'.format(pvalue))

rsquared = math.sqrt(linear_model.rsquared)
print('Ici le coefficient de corrélation est égal à {}'.format(rsquared))


# In[64]:


# Affichage de la régression linéaire correspondante
print(linear_model.summary2())


# In[65]:


seuil = 0.05

#Conclusion
if pvalue < seuil:
    if rsquared > correl_grand:
        print("On rejette HO : il y a une corrélation forte entre la taille du panier moyen et la catégorie d'âge.")
    elif rsquared > correl_moyen:
        print("On rejette HO : il y a une corrélation moyenne entre la taille du panier moyen et la catégorie d'âge.")
    elif rsquared > correl_petit:
        print("On rejette HO : il y a une petite corrélation entre la taille du panier moyen et la catégorie d'âge.")
elif pvalue > seuil : 
    print("On ne peut pas rejeter HO: il n'y a pas de corrélation entre la taille du panier moyen et la catégorie d'âge")


#      4) catégories de produits achetés

# In[66]:


X = "age" # quantitative
Y = "categ" # qualitative

meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}

plt.figure(figsize=(20, 6))
plt.title("Répartition des âges en fonction des catégories", fontsize = '25')
sns.boxplot(x=data[X], y=data[Y], showmeans=True, meanprops=meanprops, orient="h")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Âge (années)")
plt.ylabel("Catégorie")

plt.savefig("images_soutenance/IV2-4.jpeg", format="jpeg")


# In[67]:


# Je calcule une ANOVA

linear_model = ols('age ~ C(categ)', data=data).fit()
table = sm.stats.anova_lm(linear_model, typ=2)
print(table)


# In[68]:


pvalue = linear_model.f_pvalue
print('Ici la p-value est égale à {}'.format(pvalue))


# In[69]:


# Affichage de la régression linéaire correspondante
print(linear_model.summary2())


# In[70]:


pvalue = linear_model.f_pvalue
print('Ici la p-value est égale à {}'.format(pvalue))

rsquared = math.sqrt(linear_model.rsquared)
print('Ici le coefficient de corrélation est égal à {}'.format(rsquared))


# In[71]:


seuil = 0.05

#Conclusion
if pvalue < seuil:
    if rsquared > correl_grand:
        print("On rejette HO : il y a une corrélation forte entre la catégorie de produits et l'âge.")
    elif rsquared > correl_moyen:
        print("On rejette HO : il y a une corrélation moyenne entre la catégorie de produits et l'âge.")
    elif rsquared > correl_petit:
        print("On rejette HO : il y a une petite corrélation entre la catégorie de produits et l'âge.")
elif pvalue > seuil : 
    print("On ne peut pas rejeter HO: il n'y a pas de corrélation entre la catégorie de produits et l'âge.")


# Comme on l'a fait plus haut, on complète cette étude selon l'âge par une étude selon les catégories d'âge, plus pertinente.  
# On se retrouve avec deux variables qualitatives : il faut faire une heatmap.

# In[72]:


# Réalisation d'une heatmap

# Tableau de contingence réel
data.categ_age=data.categ_age.astype("str")

X="categ"
Y="categ_age"
d=data[[X,Y]].pivot_table(index=X, columns=Y, aggfunc=len)
d.head(3)


# In[73]:


cont=d.copy()
tx=data[X].value_counts()
ty=data[Y].value_counts()

cont.loc[:,"Total"]=tx
cont.loc["total",:]=ty
cont.loc["total","Total"]=len(data)

cont


# In[74]:


# Tableau de contingence théorique

tx = pd.DataFrame(tx)
ty = pd.DataFrame(ty)
tx.columns= ['foo']
ty.columns= ['foo']
n = len(data)
indep = tx.dot(ty.T)/n

indep.sort_index(axis=1, inplace=True)
indep.sort_index(inplace=True)
indep.drop(columns="nan", inplace=True)


# In[75]:


mesure = (d-indep)**2/indep
xi_n = mesure.sum().sum()

plt.figure(figsize=(8,8))
sns.heatmap(mesure/xi_n, annot=indep - d)
plt.title("Corrélation entre la catégorie d'âge et la catégorie.", fontsize = 'x-large')
plt.xlabel("Catégorie d'âge")
plt.ylabel("Catégorie de produit")

plt.savefig("images_soutenance/IV2.jpeg", format="jpeg")


# La Catégorie d'âge est corrélée à la catégorie de produits car les jeunes achètent la catégorie 2 de manière significative.
# La catégorie 2 est corrélée à la tranche 19-29.  

# In[ ]:




