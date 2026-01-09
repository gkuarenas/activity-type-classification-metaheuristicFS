# Metaheuristic-Driven Feature Selection for Activity Type Classification

## Overview
This repository contains an experimental pipeline for **classifying user activity types** using **biometric/physiological data**, with a focus on improving performance through **metaheuristic-based feature selection**.

The workflow benchmarks **six machine learning models** both:
1) **before** feature selection (using the full feature set), and  
2) **after** feature selection (using optimized feature subsets).
   
<img width="3840" height="2160" alt="Methodological Framework" src="https://github.com/user-attachments/assets/41caeed0-9bb1-48d4-9e10-1db4a7c3fd0a" />

## What’s inside
### Machine learning models evaluated
- KNN
- SVM
- Random Forest
- XGBoost
- LightGBM
- MLP

### Binarized metaheuristics evaluated for feature selection
- bPSO (Binary Particle Swarm Optimization)
- bGWO (Binary Grey Wolf Optimizer)
- bHHO (Binary Harris Hawks Optimization)
- bGA (Binary Genetic Algorithm)
- bWOA (Binary Whale Optimization Algorithm)
- bMPA (Binary Marine Predators Algorithm)
- bEO (Binary Equilibrium Optimizer)
- bJAYA (Binary Jaya Algorithm)

These algorithms search for an optimal **binary feature mask** (select/not select) using a transfer-function-based binarization approach.

## Dataset
Uses the **Health and Fitness Dataset** (publicly available on Kaggle).  
The dataset contains biometric/physiological and activity-related attributes used to predict activity type.

## Feature engineering (high level)
In addition to preprocessing and encoding, several engineered features are included to better capture user-specific intensity and fitness patterns, such as:
- Δ Heart Rate (ΔHR)
- Physical Stress Indicator
- Caloric Efficiency
- Maximum Heart Rate (HRmax)
- VO₂ max (VO2max)

## Evaluation
### Classification metrics
Models are evaluated using standard classification metrics (e.g., accuracy, precision, recall, F1-score), including per-class analysis.

### Feature selection objective
Feature selection is guided by a fitness function that balances:
- **classification performance** (lower error), and
- **subset compactness** (fewer selected features)
