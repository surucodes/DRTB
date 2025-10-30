# Drug Resistant Tuberculosis - Gaussian Naive Bayes (modularized)

This repository contains a notebook and a modularized Python package for training a Gaussian Naive Bayes classifier to predict drug-resistant tuberculosis.

What's changed
- Added a `drtb` package with modular code: data loading, preprocessing, model training, and metrics.
- Added `run_train.py` to run the full pipeline from the command line.
- Added `requirements.txt` listing dependencies.

Quick start
1. Create a virtual environment and install dependencies:

	# macOS zsh
	python3 -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt

2. Run training (dataset `dr_dataset.csv` should be in the repository root):

	python run_train.py

Notebook
The original `Naive_Bayes_Classifier_of_drug_resistant_tb.ipynb` is preserved. It can be updated to import functions from the new `drtb` package to keep it thin.

If you'd like, I can update the notebook to use the new modules and add unit tests / CI next.
