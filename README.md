# online-shoppers-purchasing-intention-ml

data taken from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) and can easily be downloaded using the download button or [this link](https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip) directly.

## problem statement
Online shopping is a land full of lurking shoppers without reaching the purchase phase. This could be for many reasons that may be out of the seller's hand. in this project, we focus on the leads that get customers to potentially buy. we analyze 12K customer sessions to see what makes a customer decide to commit to a purchase and the seller getting revenue so the merchant can make a more informed financial decisions with the leads and get more buying clients.

This is a solution for the merchant for better business decisions and also to get to the right clients that would be satisfied with the purchase and be willing to be a buyer.

The models in `models/` are all named in this format: `<model_family>_<param1>=<val1>_<param2>=<val2>.pkl`

## Project Structure & Deliverables
For a project, you repository/folder should contain the following:

- [ ] README.md with
  - [x] Description of the problem
  - [ ] Instructions on how to run the project
- [x] Data
  - [x] You should either commit the dataset you used or have clear instructions how to download the datase
- [ ] Notebook (suggested name - notebook.ipynb) with
  - [x] Data preparation and data cleaning
  - [x] EDA, feature importance analysis
  - [x] Model selection process and parameter tuning
- [ ] Script train.py (suggested name)
  - [ ] Training the final model
  - [ ] Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)
- [ ] Script predict.py (suggested name)
  - [ ] Loading the model
  - [ ] Serving it via a web service (with Flask or specialized software - BentoML, KServe, etc)
- [ ] Files with dependencies
  - [x] Pipenv and Pipenv.lock if you use Pipenv or equivalents: conda environment file, requirements.txt or **pyproject.toml**
- [ ] Dockerfile for running the service
- [ ] Deployment
  - [ ] URL to the service you deployed or
  - [ ] Video or image of how you interact with the deployed service

