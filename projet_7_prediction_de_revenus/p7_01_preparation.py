#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn import decomposition, preprocessing


data = pd.read_csv("data-projet7.csv", decimal=",")
pop = pd.read_csv("population2004_2011.csv")


# ## [I Nettoyage des Données](#ch1)
# 
# ## [II Préparation des données pour la MISSION 2](#ch2)

# <a id="ch1"></a>
# ## I Nettoyage des Données

# In[2]:


data


# In[3]:


data_miss = data[data.isna().any(axis=1)]
print(data_miss.country.unique())
data_miss.info()


# Il manque 200 valeurs pour le gdpppp, ce qui correspond à deux pays :
# XKX correspond au Kosovo et PSE correspond à la Palestine.

# In[4]:


# Imputation des gdpppp des deux pays
gdpppp_XKX_2008 = 7.367 # source : https://www.imf.org/external/datamapper/PPPPC@WEO/OEMDC/ADVEC/WEOWORLD/TUN
gdpppp_PSE_2008 = 3.324

data.loc[data["country"] == "PSE", ["gdpppp"]] = gdpppp_PSE_2008
data.loc[data["country"] == "XKX", ["gdpppp"]] = gdpppp_XKX_2008


# In[5]:


print(data.year_survey.unique())
print(data.year_survey.count())
print(data.country.value_counts())


# Le fichier csv contient les années de 2004 à 2011, sauf, étonnamment, 2005.  
# Il y a 116 pays différents dans le jeu de données.  
# PROBLEME : j'ai 11 599 lignes pour 116 pays : il doit manquer des centiles pour certains pays.

# In[6]:


# Je vérifie que chaque pays contient bien 100 centiles
nb_centiles = data.groupby("country").count()
nb_centiles = pd.DataFrame(nb_centiles)
nb_centiles.reset_index(inplace=True)

nb_centiles.loc[nb_centiles["quantile"] != 100]


# Seule la Lituanie n'a pas ses 100 quantiles. Il en manque un.

# In[7]:


del nb_centiles


# In[8]:


pd.set_option('display.max_rows', 100)

data.loc[data["country"]=="LTU"]


# Il manque le quantile 41 à la Lituanie.  
# Je l'impute à partir des quantiles 40 et 42.

# In[9]:


# Imputation de la valeur manquante pour LTU

bornes_quantile_manquant = [6239, 6240]
income_quantile_41 = (data.loc[bornes_quantile_manquant[0], "income"] + data.loc[bornes_quantile_manquant[1],
                                                                                 "income"])/2
income_quantile_41


# In[10]:


imputed_quantile = data.loc[bornes_quantile_manquant[0]]
missing_quantile = 41
imputed_quantile = dict(imputed_quantile)
imputed_quantile["quantile"] = missing_quantile
imputed_quantile["income"] = income_quantile_41
imputed_quantile

data = data.append(imputed_quantile, ignore_index=True)


# In[11]:


# Affichage du nombre de pays en fonction de l'année

countries_per_year = data.groupby("year_survey").count()
countries_per_year = pd.DataFrame(countries_per_year)
countries_per_year.reset_index(inplace=True)
countries_per_year["country"] = countries_per_year["country"]/100
countries_per_year[["year_survey","country"]]


# 2008 est à la fois l'année qui correspond au plus grand nombre de données, et l'année la plus centrale de l'échantillon.

# In[12]:


# Préparation du merge

pop = pop[["Code zone", "Année", "Valeur"]]
pop.Valeur = pop.Valeur*1000
pop.rename(columns={"Code zone":"country", "Année":"year_survey", "Valeur": "population"}, inplace=True)
pop.head(3)


# In[13]:


# Merge
data = pd.merge(data, pop, on=["year_survey","country"], how='left')
data


# In[14]:


data[data.isna().any(axis=1)]


# Il manque les valeurs pour la population pour le Kosovo (XKX) et le Soudan (SDN).

# In[15]:


# Imputation des données manquantes sur la colonne "population"

pop_XKX_2008 = 1761474  # https://data.worldbank.org/indicator/SP.POP.TOTL?locations=XK
pop_SDN_2008 = 33060397 # https://data.worldbank.org/indicator/SP.POP.TOTL?locations=SD

data.loc[data["country"] == "XKX", ["population"]] = pop_XKX_2008
data.loc[data["country"] == "SDN", ["population"]] = pop_SDN_2008


# In[16]:


temp=data.groupby("country").agg({"population":max})
temp = pd.DataFrame(temp)
temp.reset_index(inplace=True)

POP_MOND_2010 = 6956824000 # source : https://fr.wikipedia.org/wiki/Population_mondiale
print("La population couverte par l'analyse est de {} %.".format(temp.population.sum()/ POP_MOND_2010*100))
del temp


# In[17]:


# Création d'une boucle qui calcule les indices de Gini pour chaque pays

gini_countries = []
n = 100

for country in data.country :
    lorenz_country = data.loc[data.country==country].income.values
    lorenz = np.cumsum(np.sort(lorenz_country)) / lorenz_country.sum()
    lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0
    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    S = 0.5 - AUC # surface entre la première bissectrice et la courbe de Lorenz
    gini = 2*S
    gini_countries.append(gini)
    
# Ajout des indices de Gini à une colonne de "data"
data["gini"] = gini_countries
data


# #### Réponse aux questions de la Mission 1
# 
# - Il y a cent quantiles pour chaque pays et chaque année : il s'agit donc de centiles
# - Echantillonner une population en utilisant des quantiles est une bonne méthode car elle permet d'avoir une appréhension complexe de cette population, qui n'est pas vue comme un tout artificiellement homogène, mais au contraire comme une succession de couches sociales. C'est une vraie diversité et pas une fausse unité. Et cela permet de rester davantage proche des données initiales, récoltées sur le terrain. On peut toujours aller vers l'abstraction par la suite en réunissant ces données.
#   
# -- Unité PPP : il s'agit du PIB à PPA, à parité de pouvoir d'achat : c'est un indice créé pour comparer le pouvoir d'achat dans les différents pays du monde, en faisant la synthèse du prix du panier d'achat et du salaire. Tout cela est ramené à un score : plus il est élevé, plus le pouvoir d'achat moyen est grand dans un pays.

# In[18]:


data.describe()


# In[19]:


meanprops = {'marker':'o', 'markeredgecolor':'black',
            'markerfacecolor':'firebrick', "markersize":'7'}

data_boxplots = data[["income", "gdpppp", "population", "gini"]]

for columns in data_boxplots : 
    plt.figure(figsize=(20,5))
    sns.boxplot(x=columns, data=data_boxplots, showmeans=True, meanprops=meanprops, orient="h")
    plt.title('Variation de {}'.format(columns), fontsize='23')
    plt.xlabel('{}'.format(columns), fontsize='18')
    plt.xticks(fontsize=13)
    plt.show()


# J'observe un outlier sur le boxplot du gdpppp.  
# Les autres boxplots ne montrent pas de donnée surprenante.

# In[20]:


gdpppp_outlier = data.gdpppp.max()
data.loc[data.gdpppp == gdpppp_outlier]


# Il s'agit des îles Fidji. Je remplace le gdpppp par celui qui est correct.

# In[21]:


# Imputation du gdpppp correct
gdpppp_FJI_2008 = 8460
data.loc[data["country"] == "FJI", ["gdpppp"]] = gdpppp_FJI_2008


# <a id="ch2"></a>
# ## II Préparation des données pour la MISSION 2

# In[22]:


# Je sélectionne la colonne gdpppp, qui me donne un indice moyen de la richesse par habitant, et l'indice de 
# gini, qui me donne un indice sur la répartition de cette richesse dans le pays

data_kmeans = data.groupby('gdpppp').agg({'gini':max})
data_kmeans=pd.DataFrame(data_kmeans)
data_kmeans.reset_index(inplace=True)
data_kmeans


# In[23]:


# Application d'un algorithme de classification pour déterminer des pays représentatifs des classes de revenu

# Sélection des données numérique du df
X = data_kmeans

# Centrage et Réduction
X_scaled = preprocessing.StandardScaler().fit_transform(X)
X = X_scaled.copy()

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Représentation graphique
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')

plt.savefig("images_soutenance/mission1_elbow.jpeg", format="jpeg")


# Le résultat n'est pas très clair. La courbe s'infléchit nettement à k=3. On peut toutefois voir deux autres légers infléchissements pour k=7 ou k=9. Je choisis k=7 pour avoir de la marge pour ajouter les pays extrêmes (le mieux classé et le moins bien classé).

# In[24]:


# Application de l'algorithme K-means

from fanalysis.pca import PCA


# Calcul des composantes principales
n_comp = 2
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)

# Nombre de clusters souhaités
n_clust = 7

# Clustering par K-means
km = KMeans(n_clusters=n_clust)
km.fit(X)

# Récupération des clusters attribués à chaque individu
clusters = km.labels_

# Affichage du clustering par projection des individus sur le premier plan factoriel
pca = decomposition.PCA(n_components=n_comp).fit(X)
X_projected = pca.transform(X)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=clusters.astype(np.float), cmap = 'jet', alpha=.2)
plt.title("Projection des {} individus sur le 1e plan factoriel pour k=7".format(X_projected.shape[0]))
plt.xlabel("gdpppp", fontsize=13)
plt.ylabel("gini", fontsize=13)

plt.savefig("images_soutenance/mission1_kmeans.jpeg", format="jpeg")


# In[25]:


# Création d'une colonne établissant les groupes de pays d'après le K-means
data_kmeans["cluster"]=clusters


# In[26]:


# Ajout de cette colonne au df principal
data = pd.merge(data, data_kmeans, on=["gdpppp", "gini"], how="left")
data


# In[27]:


# Affichage des pays contenus dans chaque cluster
temp=data.groupby("country").agg({"cluster":max})
temp = pd.DataFrame(temp)
temp.reset_index(inplace=True)

for i in range(0,n_clust):
    print("groupe {}".format(i))
    print(", ".join(temp.country[temp.cluster==i]))


# In[28]:


# Pour chaque cluster, choix des pays pour lesquels le plus d'indices de Gini est disponible

gini_1960_2020 = pd.read_csv("gini.csv", sep=',')
gini_2004_2011 = gini_1960_2020[["Country_Code","Country_Name","2004","2005","2006","2007","2008","2009",
                                        "2010", "2011"]] # Sélection des années de l'étude
del gini_1960_2020
gini_2004_2011.set_index("Country_Code", inplace=True)
gini_2004_2011


# In[29]:


gini_transposed = gini_2004_2011.transpose()
gini_transposed


# In[30]:


# DataFrame avec le compte des indices de gini disponibles pour chaque pays
count_val = gini_transposed.count()
count_val = pd.DataFrame(count_val)
count_val.reset_index(inplace=True)
count_val.columns=["country", "count_val"]


data = pd.merge(count_val, data, on="country", how="right")
data[data.isna().any(axis=1)]


# Il n'y a pas de recension d'indices de Gini pour Taiwan. On impute donc la valeur 0.

# In[31]:


# On impute la valeur 0 pour Taiwan
data.loc[data["country"] == "TWN", ["count_val"]] = 0


# In[32]:


# Dataframe avec seulement les pays contenant le maximum d'indices de gini pour chaque cluster
best_count = data.set_index("country").groupby(["cluster"]).agg({"count_val":max})
best_count = pd.DataFrame(best_count)
best_count.reset_index(inplace=True)

best_countries = pd.merge(data, best_count, on=["cluster", "count_val"], how='right')

data = pd.merge(data, best_count, on=["cluster", "count_val"], how='left')
data


# In[33]:


best_count


# In[34]:


data.to_csv("data", index=False)
best_countries.to_csv("best_countries", index = False)
gini_transposed.to_csv("gini_transposed")


# In[ ]:




