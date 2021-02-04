#!/usr/bin/env python
# coding: utf-8

# En observant l'ACP issue de l'analyse des quatre variables imposées, on remarque que la variation de population n'est pas corrélée à la disponibilité alimentaire : la variation de la population n'est donc pas un critère pertinent pour estimer l'élargissement du marché du poulet dans les divers pays. En effet une augmentation de la population n'implique pas nécessairement une hausse du pouvoir d'achat de cette population, nécessaire pour l'alimentation carnée.  
# Il convient de prendre en compte un autre critère : le PIB par exemple, qui a été rajouté dès le début de ce notebook.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.api as sm
from functions_updated import *
from scipy.stats import shapiro
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn import preprocessing, decomposition
from sklearn.cluster import KMeans
import tests_normalite

data = pd.read_csv("data_expose")
data.dropna(inplace=True)
data.drop(columns={"protein_supply_g_cap_day"}, inplace=True) # Redondance avec food_supply_Kcal
ue = pd.read_csv("ue.csv")


# In[2]:


data[['country_code', 'year']]=data[['country_code', 'year']].astype("category")  
data.describe()


# In[3]:


sns.pairplot(data)
plt.savefig("pairplot.jpeg", format = "jpeg")


# On remarque une corrélation nette entre food_supply et protein_supply.  
# D'autres corrélations moins nettes apparaissent.

# In[4]:


# Affichage de la matrice des corrélations
correlationMatrix = data.corr()
correlationMatrix


# In[5]:


# Affichage d'une heatmap
plt.figure(figsize=(12,6))
sns.heatmap(correlationMatrix, annot=True)
plt.savefig("matrice_correlation.jpeg", format = "jpeg")
plt.show()


# Confirmation de ce qui a été constaté sur le pairplot.  
# Observation de corrélations négatives entre la variation de la population et les autres indicateurs.  
# Corrélation forte entre le PIB et les autres indicateurs.

# In[6]:


# Affichage d'un dendrogramme

# Préparation des données pour le clustering
dendrogramme=data.drop(["country_code", "year"], axis=1)
dendrogramme.set_index(["country"], inplace=True)
X = dendrogramme.values
names = dendrogramme.index

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Clustering hiérarchique
Z = linkage(X_scaled, 'ward')

# Affichage du dendrogramme
plot_dendrogram(Z, names)
plt.savefig("dendrogramme.jpeg", format = "jpeg")


# A une distance de 10, on a 6 branches : l'ACP peut se faire sur 6 clusters.

# In[7]:


# Tableau des centroïdes avec 6 clusters 

# Création d'une colonne "cluster"
t = 6
clusters = fcluster(Z, t, criterion='maxclust')
data['cluster'] = clusters

# Tableau des centroïdes
centroides = data.groupby('cluster').mean()
centroides


# In[8]:


# Affichage des pays contenus dans chaque cluster
for i in range(1,t+1):
    print("groupe {}".format(i))
    print(", ".join(data.country[data.cluster==i]))


# In[9]:


# ACP et nuage des individus

# choix du nombre de composantes à calculer
n_comp = 7

selected_variables = ["variation_pop","food_supply_Kcal_cap_day", "ani_protein_ratio",
                     "poultry_slaughtered_tons", "PIB_hab", "poultry_imported_tons", "ratio_poultry_meat"]
X = data[selected_variables].values

# préparation des données pour l'ACP
names = data.index
features = selected_variables

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=n_comp)
pca.fit(X_scaled)

# Eboulis des valeurs propres
display_scree_plot(pca)
plt.savefig("eboulis.jpeg", format = "jpeg")

# Cercle des corrélations
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(2,3)], labels = np.array(features))

# Projection des individus
X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(2,3)], illustrative_var = data.cluster)

plt.show()


# L'éboulis des valeurs propres nous indique que les quatre premiers axes d'inertie traduisent plus de 80% de l'information, ce qui est suffisant. 

# Une lecture de l'ACP, du nuage des individus et du tableau des centroïdes amène les conclusions suivantes : 
# 
# - le cluster 1 ne présente pas d'intérêt : hormis la population, tous les indicateurs sont faibles
# - le cluster 5 présente un intérêt contrasté : il s'agit des pays à consommation de viande importante, mais dont la consommation de poulet n'est pas très élevée. Ce n'est pas un cluster de premier choix
# - le cluster 2 est intéressant : très bien représenté sur F1, F2, F3 et F4, mais avec un PIB/hab assez faible. C'est un cluster très réduit, contenant trois pays qui constituent des marchés énormes : Brésil, Etats-Unis, Chine continentale. Il est à garder pour un saut plus profond à l'international, dans un second temps.
# - le cluster 3 occupe une position très intermédiaire.
# 
# - le cluster 6 est intéresant puisqu'il est constitué de pays ayant une très forte augmentation de la population, avec un PIB/hab important. Il comprend notamment, en UE, le Luxembourg, qui semble donc un très bon pays pour un premier essai à l'international.
# - le cluster 4, qui contient la France, est intéressant dans un second temps puisqu'il importe beaucoup de poulet par rapport au nombre qu'il abat : il y a donc des circuits d'importation importants, sur lesquels ces pays s'appuient. Son PIB_hab est important.

# In[10]:


# Vérification de la normalité des variables
# Je vais effectuer différents tests : d'abord la droite de Henry, qui est un test graphique

# Droite de Henry
for i in selected_variables:
    sm.qqplot(data[i], dist=scipy.stats.distributions.norm, line='s')
    plt.title("Droite de Henry : {}".format(i), fontsize='x-large')
    plt.savefig("Henry{}.jpeg".format(i), format = "jpeg")


# Les variables food_supply_Kcal et protein_supply_g semblent suivre la droite de Henry, et pas les trois autres variables (population, dispo alimentaire en termes de protéines animales, PIB_hab).  
# Je fais toutefois un test complémentaire sur ces variables, toujours pour vérifier leur adéquation à la loi Normale.  

# In[11]:


# Test de Shapiro
tests_normalite.test_shapiro(data, selected_variables, 0.05)


# In[12]:


# Le seuil habituel de 5% est trop élevé : je le baisse à 2%
tests_normalite.test_shapiro(data, selected_variables, 0.02)


# In[13]:


# Je prends les variances des deux clusters les plus opposés, le 1 (pays pauvres) et le 1 (pays riches)
worst_cluster = 1
best_cluster = 3

my_clusters = [data.ratio_poultry_meat.loc[data.cluster==worst_cluster],                 data.ratio_poultry_meat.loc[data.cluster==best_cluster]]
print(scipy.stats.bartlett(my_clusters[0], my_clusters[1]))
print(scipy.stats.ttest_ind(my_clusters[0], my_clusters[1]))


# Test de Bartlett : la p-valeur pour le résultat du test de Bartlett est élevée, >0.5 : on ne rejette pas H0 et on en conclut que les variances sont homogènes.  
# Test des moyennes : la p-valeur est < 5% : on rejette H0 : ls moyennes ne sont pas égales.
# Concernant la variable ratio_poultry_meat, on a donc bien des groupes de clusters assez différents. Le cluster 2 est bien différent du cluster 4.

# In[14]:


# Boxplots des variables qui suivent la loi normale

meanprops = {'marker':'o', 'markeredgecolor':'black', 'markerfacecolor':'firebrick'}

# Ratio poultry_meat
plt.figure(figsize=(10, 3))
sns.boxplot(x=data.ratio_poultry_meat,y=data.cluster, data=data, showmeans=True, meanprops=meanprops, orient="h")
plt.title("Ratio poulet/viande", fontsize = '17')
plt.xlabel("Ratio poulet/viande totale (%)", fontsize='x-large')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("boxplots1.jpeg", format = "jpeg")

# Alimentation en Kcal
plt.figure(figsize=(10, 3))
sns.boxplot(x=data.food_supply_Kcal_cap_day, y=data.cluster, data=data, showmeans=True, meanprops=meanprops, orient="h")
plt.title("Variation du nombre de calories par habitant dans le monde (cal)", fontsize = '17')
plt.xlabel("Nombre de calories", fontsize='x-large')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig("boxplots2.jpeg", format = "jpeg")


# In[15]:


data


# Le ratio poulet/viande est concentré, pour moitié, entre 25% et 55%. Il va, sinon de presque 0, à plus de 85%. Il y a donc une variation très forte de ce ratio!
# Le nombre de calories varie du simple au double, selon une loi homogène puisque la moyenne est égale à la médiane.

# In[16]:


# Etude spécifique des clusters 4 et 6, les mieux représentés sur F1/F2 (le cluster 6 ne contient que cinq
# individus, donc il faut prendre le second meilleur cluster aussi)
best_clusters=[4,6]

# Sélection des meilleurs clusters
pca_best_cluster = pd.DataFrame(X_projected)
pca_best_cluster["cluster"] = clusters


pca_best_cluster = pca_best_cluster.loc[pca_best_cluster.cluster.isin(best_clusters)]
pca_best_cluster


# In[17]:


# Liste des pays contenus dans ces meilleurs clusters

liste_pays = data.drop(columns={"year"}).loc[data.cluster.isin(best_clusterS)].sort_values                                    (by=["cluster"]).reset_index(drop=True)

# Suppression de la ligne France
code_france = 68
liste_pays = liste_pays.loc[liste_pays.country_code!=68]

liste_pays.drop(columns={"country_code"})


# La liste étant longue, on peut n'afficher que les pays de l'Union Européenne, plus pertinents pour un premier pas à l'international (législation, proximité géographique).

# In[ ]:


# Liste des pays de l'UE seulement

# Création d'une liste des codes de pays appartenant à l'UE
ue = ue.rename(columns={"Code zone" : "country_code"})
ue = ue["country_code"].tolist()

# Tri du df précédent pour n'afficher que ces pays
liste_pays_ue = liste_pays.loc[liste_pays.country_code.isin(ue)]
liste_pays_ue.drop(columns={"country_code"}).sort_values                                    (by=["PIB_hab", "variation_pop"], ascending=False).reset_index(drop=True)


# Le Luxembourg est un marché très favorable, avec une augmentation importante de la population et un PIB par habitant très élevé.  
# Les consommations en protéines par personne et par jour sont proches dans les pays de l'UE (entre 100 et 118g) ; le point le plus pertinent à prendre en compte semble donc être le PIB, puis la variation de la population.
