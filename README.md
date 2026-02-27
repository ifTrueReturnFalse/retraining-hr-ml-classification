# Projet OC – Classification Machine Learning RH / HR Machine Learning Classification

[![Jupyter](https://img.shields.io/badge/Jupyter-ffffff?logo=Jupyter)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=fff)](#)
[![SHAP](https://img.shields.io/badge/SHAP-000000?logo=shap&logoColor=fff)](#)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-blue)

---

## 🇫🇷 Version française

Ceci est le dépôt GitHub d’un projet réalisé dans le cadre de ma formation **Développeur IA** avec OpenClassrooms.

L’objectif principal est de développer et réentraîner un modèle de **classification** dans un contexte de Ressources Humaines (RH).
Le projet met l'accent sur la préparation des données, la gestion de la redondance des variables et l'explicabilité du modèle.

L'analyse et la modélisation ont été réalisées via un **Notebook Jupyter**, en utilisant des pipelines `scikit-learn` pour assurer la reproductibilité et `SHAP` pour l'interprétabilité.

### Fonctionnalités Clés

- **Préparation des données** : Conversion de données textuelles en valeurs numériques et nettoyage.
- **Analyse de corrélation** : Identification et suppression des variables fortement corrélées (redondantes) pour optimiser le modèle.
- **Modélisation** : Création de pipelines d'apprentissage automatique avec `scikit-learn`.
- **Explicabilité** : Utilisation de la librairie **SHAP** pour interpréter les prédictions du modèle.
- **Visualisation** : Graphiques pour l'analyse exploratoire et les résultats.

## Technologies utilisées

| Pile Technique | Outil | Rôle |
|:---|:---|:---|
| Langage | Python | Langage principal pour le Machine Learning |
| Manipulation des données | Pandas / NumPy | Nettoyage et manipulation des structures de données |
| Machine Learning | Scikit-learn | Entraînement des modèles de classification et Pipelines |
| Explicabilité | SHAP | Interprétation des résultats du modèle |
| Visualisation | Matplotlib / Seaborn | Création de graphiques pour l'analyse de données |
| Gestion de projet | uv | Gestionnaire de dépendances et d'environnement virtuel performant |

## Installation & utilisation

> J'ai utilisé uv pour gérer mes dépendances (remplaçant Poetry ou Pip), il est recommandé de l'avoir sur votre machine pour installer au mieux ce projet.

1. Cloner le dépôt

```bash
git clone https://github.com/ifTrueReturnFalse/retraining-hr-ml-classification.git
cd retraining-hr-ml-classification
```

2. Installer les dépendances

```bash
uv sync
```

3. Lancer le notebook

Activez l'environnement et lancez Jupyter :

```bash
uv run jupyter notebook
```

## Aperçu de l'analyse

Le projet intègre des fonctions utilitaires spécifiques pour améliorer la qualité du modèle :

Une analyse fine des corrélations a été mise en place pour réduire la dimensionnalité.

- **Matrice de corrélation** : Calcul et filtrage de la matrice triangulaire supérieure.
- **Seuil de tolérance** : Identification automatique des colonnes dépassant un seuil de corrélation (ex: 0.9) pour éviter la multicolinéarité.

L'utilisation de `Pipeline` permet d'encapsuler les étapes de prétraitement et le modèle final.

- Assure que les transformations appliquées au jeu d'entraînement sont reproduites à l'identique sur le jeu de test.
- Facilite le réentraînement et le déploiement du modèle.

---

## 🇬🇧 English Version

This repository contains a project completed as part of my **AI Developer** training with OpenClassrooms.

The main objective is to develop and retrain a **classification model** within a Human Resources (HR) context. The project focuses on data preparation, handling feature redundancy, and model explainability.

The analysis and modeling were performed using a **Jupyter Notebook**, utilizing `scikit-learn` pipelines to ensure reproducibility and `SHAP` for interpretability.

## Key Features

- **Data Preparation**: Converting text data to numeric values and cleaning.
- **Correlation Analysis**: Identifying and removing highly correlated (redundant) features to optimize the model.
- **Modeling**: Creating Machine Learning pipelines with `scikit-learn`.
- **Explainability**: Using the **SHAP** library to interpret model predictions.
- **Visualization**: Charts for exploratory analysis and results.

## Tech Stack

| Stack	| Tool	| Role| 
|:---|:---|:---| 
| Language	| Python	| Primary language for Machine Learning | 
| Data Manipulation	| Pandas / NumPy	| Data cleaning and structure manipulation| 
| Machine Learning	| Scikit-learn	| Classification model training and Pipelines| 
| Explainability	| SHAP	| Model result interpretation| 
| Visualization	| Matplotlib / Seaborn	| Creation of charts for data analysis| 
| Project Management	| uv	| High-performance dependency and virtual environment manager| 

## Installation & Usage

> This project uses uv for dependency management (replacing Poetry or Pip). It is recommended to have it installed on your machine to set up the project environment correctly.

1. Clone the repository:

```bash
git clone https://github.com/ifTrueReturnFalse/retraining-hr-ml-classification.git
cd retraining-hr-ml-classification
```

2. Install dependencies:

```bash
uv sync
```

3. Launch the notebook:

Activate the environment and start Jupyter:

```bash
uv run jupyter notebook
```

## Analysis Overview

The project integrates specific utility functions to improve model quality:

A detailed correlation analysis was implemented to reduce dimensionality.

- **Correlation Matrix**: Calculation and filtering of the upper triangular matrix.
- **Tolerance Threshold**: Automatic identification of columns exceeding a correlation threshold (e.g., 0.9) to avoid multicollinearity.
- 
The use of `Pipeline` encapsulates preprocessing steps and the final model.

- Ensures that transformations applied to the training set are reproduced identically on the test set.
- Facilitates model retraining and deployment.
