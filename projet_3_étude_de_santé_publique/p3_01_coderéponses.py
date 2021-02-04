#!/usr/bin/env python
# coding: utf-8

# # Etude de la faim dans le monde

#  

# ## Q1 :  Donnez le résultat de votre calcul pour l'année 2013

#  

# In[1]:


import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

df_population=pd.read_csv('df_population.csv')


# In[2]:


annee_choisie=2013

print(df_population['population'].mask(df_population['year']!=annee_choisie).sum())


# ##### Ce résultat est trop important d'environ 1,4 millions --> je vais vérifier à la main (car l'échantillon est petit) s'il n'y pas de doublons dans les pays

# In[3]:


pd.set_option('display.max_rows', None)
df_population


# ##### Je remarque que la Chine est présente deux fois : une fois de manière générique et une deuxième fois séparée en quatre zones : Chine continentale, Hong-Kong, Macao et Taiwan
# 
# ##### Il me semble plus intéressant de conserver les quatre zones séparées
# ##### Un calcul rapide me confirme l'équivalence exacte entre la somme de ces quatre zones et la population de "Chine" --> je peux donc supprimer la ligne qui contient "Chine"

# In[4]:


pd.reset_option('display.max_rows')

code_pays_chine=351
df_population=df_population.loc[df_population['country_code'] != code_pays_chine]

# Je vérifie ensuite la somme des colonnes

print(df_population['population'].mask(df_population['year']!=annee_choisie).sum())


# ##### Le nombre de 7 milliards est cohérent avec la population actuelle

# In[5]:


# Création d'une variable avec ce nombre de la population mondiale pour la suite

pop=df_population['population'].mask(df_population['year']!=annee_choisie).sum()


#  

# ## Q2 : Trouvez l'équation et vérifiez-la avec l'exemple du blé en France

#  

# ##### Je postule que la formule mathématique que l'on recherche est :
# 
# #####   Dispo int
# 
# ##### = Prod° + Imp° + Variation stocks - Exp°
# 
# ##### = Dispo alim + Alim° animale + Semences + Pertes + Traitement + Autres util°
# 

# In[6]:


df_aniveg = pd.read_csv("df_aniveg.csv")

# Vérification avec l'exemple du blé en France

ble_france = df_aniveg.loc[(df_aniveg["country"] == "France") & (df_aniveg["item"] == "Blé")]

print(ble_france['production']+ble_france['import_quantity']+ble_france['stock_variation']      -ble_france['export_quantity'])

print(ble_france['food']+ble_france['feed']+ble_france['seed']+ble_france["waste"]     +ble_france['processing']+ble_france['other_uses'])


# ##### Ce calcul me donne la même valeur selon les deux équations : 20298

#  

# ## Q3 : Calculez (pour chaque pays et chaque produit) la disponibilité alimentaire en kcal puis en kg de protéines

#  

# In[7]:


# Je supprime les lignes contenant la Chine, pour éviter la redondance

df_aniveg=df_aniveg.loc[df_aniveg['country_code'] != code_pays_chine]


# In[8]:


# J'ai besoin de multiplier les colonnes "food supply" et "protein supply" par la population de chaque pays.

# 1ère étape : rajoute les données du fichier "population" à mon df

df = pd.merge(df_aniveg, df_population, on=['country_code', "year", "country"])

df.head()


# ##### J'ai des colonnes redondantes : aurais-je pu faire un .merge différent, qui aurait empêché cette redondance ?

# In[9]:


# Je crée deux nouvelles colonnes : disponibilité alimentaire en kcal, puis en kg de protéines

jours_annee=365

df['food_supply_kcal_year']=df['food_supply_kcal_capita_day']* df['population']*jours_annee

df['protein_supply_quantity_g_year']=df['protein_supply_quantity_g_capita_day']*df['population']*jours_annee


# In[10]:


# Aggrégation pour afficher ces nouvelles colonnes en fonction du pays

pays = df.mask(df['year']!=annee_choisie).groupby(['country_code', 'country'])
print (pays['food_supply_kcal_year', 'protein_supply_quantity_g_year'].sum())


# In[11]:


# Aggrégation pour afficher ces nouvelles colonnes en fonction des produits

produits = df.mask(df['year']!=annee_choisie).groupby(['item_code', 'item'])

print (produits['food_supply_kcal_year', 'protein_supply_quantity_g_year'].sum())


# In[ ]:





# # Q4 : Calculez pour chaque pays le ratio énergie/poids en kcal/kg
# # Vérifiez la cohérence du calcul avec la valeur calorique d'un oeuf

# In[ ]:





# In[12]:


# Création de deux nouvelles colonnes : ratio_energie_poids_kcal/kg, et ratio_prot_poidstotal_g_kg

df['ratio_energie_poids_kcal_kg']=df['food_supply_kcal_year'] / (df['food'].loc[df['food']!=0]*10**6)

df['ratio_prot_poidstotal_g_kg']=df['protein_supply_quantity_g_year'] / (df['food'].loc[df['food']!=0]*10**6)

# J'actualise la variable "produits" pour qu'elle contienne les deux nouvelles colonnes

produits = df.mask(df['year']!=annee_choisie).groupby(['item_code', 'item'])

pd.set_option('display.max_rows', None)
print (produits['ratio_energie_poids_kcal_kg'].mean())


# ##### La valeur calorique d'un oeuf moyenne convenue est 1450kcal pour 1kg. Le résultat (1345kcal) est cohérent.

# In[13]:


pd.reset_option('max_rows')

print (produits['ratio_prot_poidstotal_g_kg'].mean())


# ##### Le ratio protéines/poids pour l'avoine est d'un peu moins de 10% d'après les sources consultés
# ##### On a ici un peu plus de 8%, ce qui est cohérent.

# In[ ]:





# # Q5 : Citez 5 aliments parmi les 20 aliments les plus caloriques, en utilisant le ratio énergie/poids

# In[ ]:





# In[14]:


produits['ratio_energie_poids_kcal_kg'].mean().nlargest(20)


# ##### Je cite les cinq premiers aliments les plus caloriques, soit les huiles de poissons, de palmistes, de palme, de germe de maïs, de son de riz.

# # Q6 : Calculez, pour les produits végétaux uniquement, la disponibilité intérieure mondiale exprimée en kcal.

# In[ ]:





# In[15]:


# Dans mon df général je crée une colonne "Dispo intérieure en kcal"

df['dispo_int_kcal']=df['domestic_supply_quantity']*10**6*df['ratio_energie_poids_kcal_kg']

veg=df.mask(df['year']!=annee_choisie).mask(df['origin']!='vegetal').groupby(['item_code', 'item'])

print (veg['dispo_int_kcal'].sum())


# In[16]:


somme_veg = df.mask(df['year']!=annee_choisie).mask(df['origin']!='vegetal').groupby(['origin'])

print (somme_veg['dispo_int_kcal'].sum())


# # Q7 : Combien d'humains pourraient être nourris si toute la disponibilité intérieure mondiale de produits végétaux était utilisée pour de la nourriture ?

# In[ ]:





# In[17]:


# Même opération pour avoir la disponibilité intérieure de végétaux en termes de protéines

df['dispo_int_prot']=df['domestic_supply_quantity']*10**6*df['ratio_prot_poidstotal_g_kg']

# J'actualise le groupby

somme_veg = df.mask(df['year']!=annee_choisie).mask(df['origin']!='vegetal').groupby(['origin'])


# ##### J'ai besoin de données supplémentaires pour répondre à la question : les apports quotidiens recommandés, en moyenne, en kcal et en protéines.
# 
# ##### Ma source pour les kcal : j'ai pris la page Wikipedia sur les ANC, qui donne 1800kcal, sans détailler son calcul mais qui semble cohérent par rapport aux données obtenues sur d'autres sites (source2), et qui disent 2300kcal pour les hommes et 1900kcal pour les femmes, auxquels il convient d'ajouter les enfants : on tombe vraisemblablement sur un chiffre approchant les 1800kcal en moyenne, le chiffre de Wikipedia me semble donc digne de confiance. Néanmoins il est pertinent de prendre un chiffre un peu plus élevé car 1800 est vraiment l'apport minimum
# ##### source wiki : https://fr.wikipedia.org/wiki/Apports_nutritionnels_conseill%C3%A9s#cite_note-2
# ##### source2 : NCBI : https://www.ncbi.nlm.nih.gov/books/NBK234938/
# 
# ##### Pour les protéines, j'utilise un article de la revue de santé de Harvard, qui donne 0.8g/kg/j
# ##### https://www.health.harvard.edu/blog/how-much-protein-do-you-need-every-day-201506188096#:~:text=The%20Recommended%20Dietary%20Allowance%20(RDA,meet%20your%20basic%20nutritional%20requirements.
# ##### multiplié par le poids moyen mondial (62kg)
# ##### source : https://www.telegraph.co.uk/news/earth/earthnews/9345086/The-worlds-fattest-countries-how-do-you-compare.html
# 

# In[18]:


somme_veg_kcal=somme_veg['dispo_int_kcal'].sum()
somme_veg_prot=somme_veg['dispo_int_prot'].sum()

# J'effectue mes calculs sur le nombre d'humains qu'on peut nourrir :
    #1 - Avec les données concernant les calories

nb_cal_quotid=2000
    
nb_hum_veg_kcal=(somme_veg_kcal/nb_cal_quotid)/jours_annee

   #2 - Avec les données concernant les protéines

nb_prot_quotid=0.8*62
nb_hum_veg_prot=(somme_veg_prot/nb_prot_quotid)/jours_annee

print(nb_hum_veg_kcal)
print(nb_hum_veg_prot)


# ##### On pourrait nourrir 17 milliards d'être humains par rapport aux calories, et 16 milliards par rapport aux protéines

# In[19]:


print(nb_hum_veg_kcal/pop*100)
print(nb_hum_veg_prot/pop*100)


# ##### Soit 241% de la population mondiale par rapport aux calories
# ##### 230% par rapport aux protéines

# In[ ]:





# # Q8 : Combien d'humains pourraient être nourris si toute la disponibilité alimentaire en produits végétaux la nourriture végétale destinée aux animaux et les pertes de produits végétaux étaient utilisés pour de la nourriture ?

# In[ ]:





# In[20]:


df_q8=df.mask(df['year']!=annee_choisie).mask(df['origin']!='vegetal')

somme_veg_kcal_2=((df_q8['food']+df_q8['feed']+df_q8['waste'])*10**6*df_q8['ratio_energie_poids_kcal_kg']).sum()
somme_veg_prot_2=((df_q8['food']+df_q8['feed']+df_q8['waste'])*10**6*df_q8['ratio_prot_poidstotal_g_kg']).sum()

# Division pour obtenir le nombre d'humains

nb_hum_veg_kcal_q8 = somme_veg_kcal_2/nb_cal_quotid/jours_annee
nb_hum_veg_prot_q8 = somme_veg_prot_2/nb_prot_quotid/jours_annee

# Le calcul en pourcentage de la population mondiale
    
print(nb_hum_veg_kcal_q8/pop*100)
print(nb_hum_veg_prot_q8/pop*100)


# ##### On pourrait donc nourrir 120 % de la population par rapport aux calories
# ##### et 127% par rapport aux protéines

# # Q9 : Combien d'humains pourraient être nourris avec la disponibilité alimentaire mondiale ? 

# In[ ]:





# In[21]:


df_q9=df.mask(df['year']!=annee_choisie)

somme_totale_dispo_alim_mond_kcal=(df_q9['food']*10**6*df_q9['ratio_energie_poids_kcal_kg']).sum()
somme_totale_dispo_alim_mond_prot=(df_q9['food']*10**6*df_q9['ratio_prot_poidstotal_g_kg']).sum()

# On divise pour obtenir le nombre d'humains
    #1 pour les calories

nb_hum_dispo_alim_mond_kcal = somme_totale_dispo_alim_mond_kcal/nb_cal_quotid/jours_annee
    
    #2 pour les protéines
    
nb_hum_dispo_alim_mond_prot = somme_totale_dispo_alim_mond_prot/nb_prot_quotid/jours_annee

print(nb_hum_dispo_alim_mond_kcal)
print(nb_hum_dispo_alim_mond_prot)


# In[22]:


# Et on fait le calcul en pourcentage de la population mondiale

print(nb_hum_dispo_alim_mond_kcal/pop*100)
print(nb_hum_dispo_alim_mond_prot/pop*100)


# ##### On pourrait donc nourrir 144% de la population par rapport aux calories
# ##### 163% par rapport aux protéines

# In[ ]:





# # Q10:  Quelle proportion de la population mondiale est considérée comme étant en sous-nutrition ?

# In[ ]:





# In[23]:


df_sousalim=pd.read_csv('df_sousalim.csv')


# In[24]:


prop_sousalim_mond = df_sousalim['pop_starvation'].mask(df_sousalim['year']!=annee_choisie).sum()/pop*100
print(prop_sousalim_mond)


# ##### Il y a environ 11% de la population mondiale en sous-nutrition

# In[ ]:





# # Etablissez la liste des produits (ainsi que leur code) considéré comme des céréales selon la FAO.

# In[ ]:





# In[25]:


# J'importe le fichier cereales

df_cer = pd.read_csv("df_cer.csv")

liste_cer=df_cer["code_item"].tolist()


# In[ ]:





# # Repérez dans vos données les informations concernant les céréales (par exemple en créant une colonne de type booléen nommée "is_cereal").

# In[ ]:





# In[26]:


df["is_cereal"]=df['item_code'].isin(liste_cer)


# In[ ]:





# # Q11 : En ne prenant en compte que les céréales destinées à l'alimentation (humaine et animale), quelle proportion (en termes de poids) est destinée à l'alimentation animale ?

# In[27]:


prop_ani_kg=df['feed'].mask(df['year']!=annee_choisie).mask(df["is_cereal"]==False).sum()            / (df['feed'].mask(df['year']!=annee_choisie).mask(df["is_cereal"]==False).sum() +             df['food'].mask(df['year']!=annee_choisie).mask(df["is_cereal"]==False).sum())
print(prop_ani_kg)


# ##### En termes de poids, environ 46% de l'alimentation en céréales est destinée à l'alimentation animale

# # Sélectionnez parmi les données des bilans alimentaires les informations relatives aux pays dans lesquels la FAO recense des personnes en sous-nutrition.
# 
# # Repérez les 15 produits les plus exportés par ce groupe de pays.

# In[ ]:





# In[28]:


liste2=df_sousalim['country_code'].mask(df_sousalim["pop_starvation"].isin(['NaN'])).tolist()

pays_sousalim = df.mask(df['year']!=annee_choisie).mask(df['country_code'].isin(liste2) == False).groupby    (["item_code", "item"])
del liste2
pays_sousalim = pays_sousalim['export_quantity'].sum().nlargest(15)
print(pays_sousalim)


# # Parmi les données des bilans alimentaires au niveau mondial, sélectionnez les 200 plus grandes importations de ces produits 

# In[ ]:





# In[29]:


# Je crée un nouveau df avec seulement les 15 produits concernés

pays_sousalim=pays_sousalim.reset_index()
df_15produits= df.mask(df["year"]!=annee_choisie).loc[df["item_code"].isin(pays_sousalim['item_code'])]

# Je trie les valeurs par l'importance des importations

df_15produits.sort_values(by='import_quantity', ascending=False, inplace=True)

# Je ne prends que les 200 premières valeurs

df_15produits=df_15produits.iloc[:200,:]
df_15produits.reset_index(drop=True, inplace=True)
df_15produits.head()


# In[ ]:





# # Groupez ces importations par produit, afin d'avoir une table contenant 1 ligne pour chacun des 15 produits. Puis calculez deux ratios

# In[ ]:





# In[30]:


table_15produits=pd.pivot_table(df_15produits, index=["item_code", "item"], aggfunc="sum")
table_15produits


# In[31]:


table_15produits['ratio_autresutil_dispoint']=table_15produits['other_uses']/table_15produits['domestic_supply_quantity']

table_15produits['ratio_qtanim_qtnourr']=table_15produits['feed']/(table_15produits.feed+table_15produits.food)

print(table_15produits[['ratio_qtanim_qtnourr', 'ratio_autresutil_dispoint']])


# In[ ]:





# # Q12 : Donnez les 3 produits qui ont la plus grande valeur pour chacun des 2 ratios (vous aurez donc 6 produits à citer)

# In[ ]:





# In[32]:


table_15produits2 = table_15produits[['ratio_qtanim_qtnourr', 'ratio_autresutil_dispoint']]

print(table_15produits2['ratio_qtanim_qtnourr'].sort_values(ascending=False).nlargest(3))
print(table_15produits2['ratio_autresutil_dispoint'].sort_values(ascending=False).nlargest(3))


# In[ ]:





# # Q13 Combien de tonnes de céréales pourraient être libérées si les USA diminuaient leur production de produits animaux de 10% ?

# In[33]:


# Création d'un df contenant seulement les céréales aux USA

df_usa_2013=df.mask(df['year']!=annee_choisie).mask(df['country_code']!=231).mask(df['is_cereal']!=True)

diminution_etatsunis=(df_usa_2013['feed'].sum(axis=0, skipna=True))*10/100
print(diminution_etatsunis)


# ##### Il y aurait libération de 14 000 milliers de tonnes de céréales, soit 14 millions de tonnes de céréales.

# # Q14 : En Thaïlande, quelle proportion de manioc est exportée ? Quelle est la proportion de personnes en sous-nutrition?

# In[ ]:





# In[34]:


# Je trouve les valeurs de l'exportation et de l'importation de manioc pour la Thaïlande, puis je les mets en
# rapport pour trouver la proportion de manioc exporté

code_PAYS=216
code_ITEM=2532
thai_manioc_exp = df['export_quantity'].loc[(df['country_code'] == code_PAYS) & (df['item_code'] == code_ITEM)                                             & (df['year']==annee_choisie)]
thai_manioc_prod = df['production'].loc[(df['country_code'] == code_PAYS) & (df['item_code'] == code_ITEM)                                         & (df['year']==annee_choisie)]

thai_prop_manioc_exp = thai_manioc_exp/thai_manioc_prod*100
print(thai_prop_manioc_exp)


# In[35]:


##### 83% du manioc thaïlandais est exporté


# In[36]:


nb_sousalim_thai = df_sousalim['pop_starvation'].loc[(df_sousalim['country_code'] == code_PAYS)                                                   & (df_sousalim['year'] == annee_choisie)]
pop_thai = df['population'].loc[(df['year']==annee_choisie) & (df['country_code'] == code_PAYS)].unique()
                                               
thai_prop_sousalim = (nb_sousalim_thai/pop_thai)*100
print(thai_prop_sousalim)


# In[37]:


df.head()


# j'ai du mettre la fonction .unique parce qu'à un moment de mon df, probablement lors d'un merge, j'ai créé plein de lignes redondantes

# ##### La proportion de Thaïlandais en sous-nutrition est de 8,4%

# ###### Export des df pour la partie SQL

# In[38]:


df_population2=df_population[['country_code', 'country', 'year', 'population']].loc                [df_population['year'] == annee_choisie]
df_population2.columns=['code_pays', 'pays', 'année', 'population']
df_population2.to_csv("population2.csv", index=False)

df_dispo_alim=df[['country', 'country_code', 'year', 'item', 'item_code', 'origin',                  'food_supply_quantity_kg_capita_yr', 'food_supply_kcal_capita_day',                  'protein_supply_quantity_g_capita_day','fat_supply_quantity_g_capita_day']].loc                    [df['year'] == annee_choisie]
df_dispo_alim.columns = ["pays","code_pays","année","produit",'code_produit','origin'
    ,'dispo_alim_tonnes',"dispo_alim_kcal_p_j","dispo_prot", "dispo_mat_gr"]
df_dispo_alim.to_csv("df_dispo_alim.csv", index=False)

df_equilibre_prod=df[['country', 'country_code', 'year', 'item', 'item_code', 'domestic_supply_quantity',                  'feed', 'seed','waste', 'processing', 'food', 'other_uses']].loc[df['year'] == annee_choisie]
df_equilibre_prod.columns = ["pays","code_pays","année","produit",'code_produit','dispo_int'
    ,'alim_ani',"semences","pertes", "transfo", "nourriture", "autres_utilisations"]
df_equilibre_prod.to_csv("df_equilibre_prod.csv", index=False)

df_sous_nutrition=df_sousalim[['country', 'country_code', 'year', 'pop_starvation']].loc                    [df_sousalim['year'] == annee_choisie]
df_sous_nutrition.columns = ["pays","code_pays","année","nb_personnes"]
df_sous_nutrition=df_sous_nutrition.astype({'nb_personnes':float})
df_sous_nutrition.to_csv("df_sous_nutrition.csv", index=False)


# In[ ]:





# In[ ]:




