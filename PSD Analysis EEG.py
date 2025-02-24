#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:07:21 2025

@author: jc278357
"""

import mne
import os
import glob
import numpy as np

import pandas as pd
import numpy as np
import seaborn as sns

import pingouin as pg
import statsmodels.api as sm

# Définir le dossier contenant les fichiers EEG
eeg_folder = 'EEG_140TSA'

# Charger la liste des noms de canaux à partir du fichier texte
ch_names = np.loadtxt('ch_names.txt', dtype=str)

# Définir les bandes de fréquences
beta_band = [14, 30]
gamma_band = [30, 49]

# Initialiser des matrices pour stocker toutes les données
all_data_beta = []
all_data_gamma = []
subjects = []

# Définir les électrodes pour chaque région d'intérêt
frontal_electrodes = ["E1", "E2", "E3", "E4", "E5", "E8", "E9", "E10", "E11", "E12", "E15", "E16", "E18", "E19", "E20", "E22", "E23", "E24", "E25", "E26", "E27", "E32", "E118", "E123", "E124", "E125", "E128"]
central_electrodes = ["E6", "E7", "E13", "E29", "E30", "E31", "E36", "E37", "E53", "E54", "E55", "E79", "E80", "E86", "E87", "E104", "E105", "E106", "E111", "E112"]
parietal_electrodes = ["E58", "E59", "E60", "E61", "E62", "E66", "E67", "E71", "E72", "E76", "E77", "E78", "E84", "E85", "E91", "E96"]
occipital_electrodes = ["E63", "E64", "E65", "E68", "E69", "E70", "E73", "E74", "E75", "E81", "E82", "E83", "E88", "E89", "E90", "E94", "E95", "E99"]
temporal_right_electrodes = ["E92", "E93", "E97", "E98", "E100", "E101", "E102", "E103", "E107", "E108", "E109", "E110", "E113", "E114", "E115", "E116", "E117", "E120", "E121", "E122"] 
temporal_left_electrodes = ["E28", "E33", "E34", "E35", "E38", "E39", "E40", "E41", "E42", "E43", "E44", "E45", "E46", "E47", "E49", "E50", "E51", "E52", "E56", "E57"] 


def process_and_accumulate_band(eeg_file, ch_names, frequency_band, all_data):
    # Charger le fichier EEG
    raw = mne.io.read_raw_fif(eeg_file, preload=True)

    # Supprimer les canaux qui ne sont pas sur le cuir chevelu
    raw = raw.drop_channels(['E14', 'E17', 'E21', 'E48', 'E119', 'E126', 'E127'])
    raw.info['bads']# check that there are no more bads 

    # Supprimer la première et la dernière seconde pour les effets de bord
    start_time = raw.times[0] + 1.0
    end_time = raw.times[-1] - 1.0
    raw_band = raw.crop(tmin=start_time, tmax=end_time)

    # Calculer la PSD avec la méthode de Welch pour la bande de fréquences spécifiée
    psd_mean, freqs = raw_band.compute_psd(method='welch', picks='eeg', reject_by_annotation=True, fmin=frequency_band[0], fmax=frequency_band[1]).get_data(return_freqs=True)
   
    psd_meann=np.mean(psd_mean, axis=1)
    
    # Transposer la matrice de données
    data_matrix_transposed = np.transpose(psd_meann)

    # Accumuler les données dans la matrice globale
    all_data.append([data_matrix_transposed])


# Fonction pour regrouper les électrodes et calculer la moyenne pour chaque région d'intérêt
def calculate_roi_averages(data_matrix_transposed, electrodes):
    ch_names_list = list(ch_names)
    electrode_indices = [ch_names_list.index(electrode) for electrode in electrodes]
    roi_averages = np.nanmean(data_matrix_transposed[:, electrode_indices], axis=1)
    return roi_averages


# Parcourir tous les fichiers EEG dans le dossier
for eeg_file in glob.glob(os.path.join(eeg_folder, '*.fif')):
    
    # Ajouter le nom du fichier à la liste
    subjects.append(os.path.basename(eeg_file).split('.')[0])
    
    process_and_accumulate_band(eeg_file, ch_names, beta_band, all_data_beta)
    process_and_accumulate_band(eeg_file, ch_names, gamma_band, all_data_gamma)

# Créer un DataFrame pour chaque bande de fréquences avec les données accumulées
df_beta = pd.DataFrame(np.concatenate(all_data_beta, axis=0), columns=ch_names)
df_beta.insert(0, 'subject', subjects)
df_beta['subject'] = df_beta['subject'].str.replace('-RS_eeg', '')


# Ajouter des colonnes pour les moyennes des électrodes de chaque région d'intérêt
df_beta['Frontal'] = calculate_roi_averages(df_beta.values[:, 1:], frontal_electrodes)
df_beta['Central'] = calculate_roi_averages(df_beta.values[:, 1:], central_electrodes)
df_beta['Occipital'] = calculate_roi_averages(df_beta.values[:, 1:], occipital_electrodes)
df_beta['Parietal'] = calculate_roi_averages(df_beta.values[:, 1:], parietal_electrodes)
df_beta['Temporal_Right'] = calculate_roi_averages(df_beta.values[:, 1:], temporal_right_electrodes)
df_beta['Temporal_Left'] = calculate_roi_averages(df_beta.values[:, 1:], temporal_left_electrodes)

# Créer un DataFrame pour chaque bande de fréquences avec les données accumulées
df_gamma = pd.DataFrame(np.concatenate(all_data_gamma, axis=0), columns=ch_names)
df_gamma.insert(0, 'subject', subjects)
df_gamma['subject'] = df_gamma['subject'].str.replace('-RS_eeg', '')


# Ajouter des colonnes pour les moyennes des électrodes de chaque région d'intérêt
df_gamma['Frontal'] = calculate_roi_averages(df_gamma.values[:, 1:], frontal_electrodes)
df_gamma['Central'] = calculate_roi_averages(df_gamma.values[:, 1:], central_electrodes)
df_gamma['Occipital'] = calculate_roi_averages(df_gamma.values[:, 1:], occipital_electrodes)
df_gamma['Parietal'] = calculate_roi_averages(df_gamma.values[:, 1:], parietal_electrodes)
df_gamma['Temporal_Right'] = calculate_roi_averages(df_gamma.values[:, 1:], temporal_right_electrodes) 
df_gamma['Temporal_Left'] = calculate_roi_averages(df_gamma.values[:, 1:], temporal_left_electrodes)

#-------------------------------------------------------------------------------------------------------------
#create the full dataset with EI and clinical variables 
pheno = pd.read_excel('fei_dataset.xlsx') ##open dataset with clinical variables
pheno = pheno.iloc[:388, :] ## reshape 
pheno.loc[:, 'sex'] = pheno['sex'].replace({ 0: 'Homme', 1: 'Femme'})
pheno.loc[:, 'group'] = pheno['group'].replace({0: 'Controls', 1: 'ASD', 2: 'Relatives'})

df_pheno_beta_ROI= pd.merge(df_beta, pheno, on='subject')
df_pheno_beta_ROI = df_pheno_beta_ROI.drop_duplicates(subset='subject', keep='first')
df_pheno_beta_ROI.to_csv('/Users/julie/OneDrive/Bureau/Stage/Stage M2/Python/résultats_36controls/Dataset_pheno_beta_ROI.csv', index=False)
#df_pheno_beta.to_excel('Dataset_final_beta.xlsx', index=False)


df_pheno_gamma_ROI= pd.merge(df_gamma, pheno, on='subject')
df_pheno_gamma_ROI = df_pheno_gamma_ROI.drop_duplicates(subset='subject', keep='first')
df_pheno_gamma_ROI.to_csv('/Users/julie/OneDrive/Bureau/Stage/Stage M2/Python/résultats_36controls/Dataset_pheno_gamma_ROI.csv', index=False)
#df_pheno_gamma.to_excel('Datasetfinal_gamma.xlsx', index=False)


#-----------------------------------------------------------------------------------------------------------
columns_to_convert = ['Frontal', 'Central', 'Occipital', 'Parietal', 'Temporal_Right', 'Temporal_Left']
df_pheno_beta_ROI[columns_to_convert] = df_pheno_beta_ROI[columns_to_convert].astype(float)


df_pheno_beta_ROI= pd.read_csv('/volatile/home/jc278357/Documents/Analyse EEG/résultats_127TSA/Régression multiple/Dataset_pheno_beta_ROI.csv')
df_pheno_beta_ROI = df_pheno_beta_ROI[df_pheno_beta_ROI['subject'] != 'REC-180228-A']
df_pheno_gamma_ROI= pd.read_csv('/Users/julie/OneDrive/Bureau/Stage/Stage M2/Python/résultats_140TSA/Dataset_pheno_gamma_ROI.csv')
df_pheno_gamma_ROI = df_pheno_gamma_ROI[df_pheno_gamma_ROI['subject'] != 'REC-180228-A']

#------------------------------------------------------------------------------------------------------------------
# Define hypo and hyper column indices
hypo_columns_indices = [2, 15, 16, 17, 18, 19, 20, 21, 23, 26, 28, 29, 30, 31, 32, 33]
hyper_columns_indices = [1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 22, 24, 25, 34, 35, 36, 37, 38]

# Define frequency bands and DataFrames
frequency_bands = {'beta': [14, 30], 'gamma': [30, 49]}
dataframes = {'beta': df_pheno_beta_ROI, 'gamma': df_pheno_gamma_ROI}

# Process each frequency band
for band, indices in frequency_bands.items():
    hypo_columns_band = [f'DUNN{col}' for col in hypo_columns_indices]
    hyper_columns_band = [f'DUNN{col}' for col in hyper_columns_indices]
    
    dataframes[band][hypo_columns_band] = dataframes[band][hypo_columns_band].apply(pd.to_numeric, errors='coerce')
    dataframes[band][hyper_columns_band] = dataframes[band][hyper_columns_band].apply(pd.to_numeric, errors='coerce')

    # Process and compute dSSP score for the current band
    dataframes[band]['hypo'] = dataframes[band][hypo_columns_band].mean(axis=1)
    dataframes[band]['hyper'] = dataframes[band][hyper_columns_band].mean(axis=1)

    dataframes[band]['hyper'] = dataframes[band]['hyper'].replace(0, np.nan)
    dataframes[band]['hypo'] = dataframes[band]['hypo'].replace(0, np.nan)

    dataframes[band]['dSSP'] = dataframes[band]['hypo'] / dataframes[band]['hyper']

    mean_dSSP_band = dataframes[band]['dSSP'].mean()
    std_dSSP_band = dataframes[band]['dSSP'].std()

    dataframes[band]['dSSP'] = (dataframes[band]['dSSP'] - mean_dSSP_band) / std_dSSP_band


df_pheno_beta_ROI.loc[:, 'sex'] = df_pheno_beta_ROI['sex'].replace({'Homme':0 , 'Femme':1})
df_pheno_gamma_ROI.loc[:, 'sex'] = df_pheno_gamma_ROI['sex'].replace({'Homme':0 , 'Femme':1})

# Pour BETA = valeurs numériques
df_hB = df_pheno_beta_ROI[df_pheno_beta_ROI['group'] == 'ASD']
df_hB = df_hB.applymap(pd.to_numeric, errors='coerce')  # Convertit tout en numérique, remplace les non-convertibles par NaN

# Pour GAMMA = valeurs numériques
df_hG = df_pheno_gamma_ROI[df_pheno_gamma_ROI['group'] == 'ASD']
df_hG = df_hG.applymap(pd.to_numeric, errors='coerce')  # Idem


#----------------------conserver les sujets avec ados et adi ----------------------------------------------------------------

# Vérifier que les deux colonnes contiennent des valeurs numériques
df_hB_filtered = df_hB[
    pd.to_numeric(df_hB['ados_css'], errors='coerce').notna() &
    pd.to_numeric(df_hB['adi_crr'], errors='coerce').notna()
]

# Idem pour GAMMA
df_hG_filtered = df_hG[
    pd.to_numeric(df_hG['ados_css'], errors='coerce').notna() &
    pd.to_numeric(df_hG['adi_crr'], errors='coerce').notna()
]

df_hB_filtered.to_excel('/Users/julie/OneDrive/Bureau/Stage/Stage M2/Python/résultats_127TSA/Dataset_pheno_beta_ROI_127TSA.xlsx',index=True)
df_hG_filtered.to_excel('/Users/julie/OneDrive/Bureau/Stage/Stage M2/Python/résultats_127TSA/Dataset_pheno_gamma_ROI_127TSA.xlsx',index=True)


#%%-------------------------------------------------régression linéaire multiple --------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm

def run_Lm_analysis(df, output_filename):
    regions = ['Frontal', 'Central', 'Parietal', 'Occipital', 'Temporal_Left', 'Temporal_Right']
    results = pd.DataFrame(columns=['Region', 'Model', 'Intercept', 'Slope_hypo', 'Slope_age', 'Slope_sex', 'p-value_hypo', 'p-value_age', 'p-value_sex', 'R-squared', 'AIC'])
    
    for region in regions:
        # Définir les variables indépendantes
        x = df[['hypo', 'age_years', 'sex']]  # Inclure hyper, age et sex
        x = sm.add_constant(x)  # Ajouter une constante pour l'intercept
        
        y = df[region]
        
        # Régression linéaire multiple
        model_multiple = sm.OLS(y, x, missing='drop')
        results_summary_multiple = model_multiple.fit()
        aic_multiple = results_summary_multiple.aic
        
        # Stocker les résultats
        results = results.append({
            'Region': region,
            'Model': 'Multiple',
            'Intercept': results_summary_multiple.params[0],
            'Slope_hypo': results_summary_multiple.params[1],
            'Slope_age': results_summary_multiple.params[2],
            'Slope_sex': results_summary_multiple.params[3],
            'p-value_hypo': results_summary_multiple.pvalues[1],
            'p-value_age': results_summary_multiple.pvalues[2],
            'p-value_sex': results_summary_multiple.pvalues[3],
            'R-squared': results_summary_multiple.rsquared,
            'AIC': aic_multiple
        }, ignore_index=True)

    results.to_excel(output_filename, index=False)

# Appeler la fonction pour les données ASD
run_Lm_analysis(df_hB_filtered, '/Users/julie/OneDrive/Bureau/Stage/Stage M2/Python/résultats_127TSA/Régression multiple/Lm_ASD_beta_hypo_age_sex.xlsx')
run_Lm_analysis(df_hG_filtered, '/Users/julie/OneDrive/Bureau/Stage/Stage M2/Python/résultats_127TSA/Régression multiple/Lm_ASD_gamma_hypo_age_sex_age_sex.xlsx')

#Correction 96 p-val
pvals = [.20, .32, .64, .16, .20, .66]
reject, pvals_corr = pg.multicomp(pvals, method='fdr_bh')
print(reject, pvals_corr)

#%%----------------------------------------------------------Calcul sur f2----------------------------------------------------
def cohen_f2(r_squared):
    """
    Calcule le f2 de Cohen à partir du R2.
    
    :param r_squared: Le coefficient de détermination (R2) du modèle
    :return: La valeur du f2 de Cohen
    """
    if r_squared == 1:
        return float('inf')  # Évite la division par zéro
    return r_squared / (1 - r_squared)

# Exemple d'utilisation
r2 = 0.09  # Remplacez par votre valeur de R2
effect_size = cohen_f2(r2)
print(f"La taille d'effet (f2 de Cohen) est : {effect_size:.3f}")

# Interprétation
if effect_size < 0.02:
    print("Effet faible")
elif effect_size < 0.15:
    print("Effet modéré")
else:
    print("Effet fort")
    
#%%-----------------------------------------------------PLOT linear regression-----------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_linear_regression(df, x_col, y_col):

    # Utiliser lmplot pour afficher la régression linéaire
    sns.lmplot(x=x_col, y=y_col, data=df, aspect=1.5, height=5, ci=95, line_kws={'color': 'black'}, scatter_kws={'color': 'black'})
    
    # Ajouter des labels et un titre
    plt.title("Right Temporal", fontweight='bold', fontsize=16)
    #plt.xlabel('ADI CRR score', fontsize=12)
    plt.ylabel('PSD gamma (V²/Hz)', fontsize=12)
    
    # Afficher le plot
    plt.show()

# Exemple d'utilisation (en supposant que df est ton DataFrame et que 'hypo' et 'beta' sont les colonnes correspondantes)
plot_linear_regression(df_hG_filtered, x_col='adi_crr', y_col='Temporal_Right')

#%%-----------------------------------------------------Plot effect size------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# Définir les régions cérébrales
regions = ['Frontal', 'Central', 'Parietal', 'Occipital', 'Right Temporal', 'Left Temporal']

# Valeurs d'effet f² (remplacez-les par vos données réelles)
f2_values = [0.14, 0.35, 0.20, 0.12, 0.22, 0.20]

# Créer le graphique
plt.figure(figsize=(10, 6))

# Ajouter des zones de couleur pour les seuils définis par vos consignes
plt.axvspan(0.02, 0.15, color='lightgreen', alpha=0.5, label='Small effect (f² ≥ 0.02)')
plt.axvspan(0.15, 0.35, color='khaki', alpha=0.5, label='Medium effect (f² ≥ 0.15)')
plt.axvspan(0.35, max(f2_values) + 0.1, color='lightsalmon', alpha=0.5, label='Large effect (f² ≥ 0.35)')

# Tracer les tailles d'effet
plt.scatter(f2_values, range(len(regions)), color='black')

# Personnaliser le graphique
plt.yticks(range(len(regions)), regions)
plt.xlabel('Effect Size (f²)')

# Afficher la légende
plt.legend(loc='lower right')

# Ajouter une grille pour améliorer la lisibilité
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Ajuster la mise en page et afficher
plt.tight_layout()
plt.show()

#%%------------------------------------------Evaluer correlation entre hypo et hyper------------------------------------------
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


def compute_correlation(df):
    correlation = df['hypo'].corr(df['hyper'])
    print(f"Corrélation de Pearson entre hypo et hyper : {correlation:.4f}")

compute_correlation(df_hB_filtered)



def get_p_value(df):
    x = df[['hypo']]  # S'assurer que c'est un DataFrame
    x = sm.add_constant(x)  # Ajouter une constante pour l'intercept
    y = df['hyper']
    
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()
    
    p_value = results.pvalues[1]  # p-value associée à "hypo"
    print(f"P-value : {p_value:.5f}")

# Appel de la fonction
get_p_value(df_hB_filtered) 



def plot_correlation(df):
    sns.regplot(x=df['hypo'], y=df['hyper'], line_kws={"color": "red"}, ci=None)
    plt.xlabel("Hypo")
    plt.ylabel("Hyper")
    plt.title("Relation entre hypo et hyper")
    plt.show()

# Exemple d'utilisation
plot_correlation(df_hB_filtered)

