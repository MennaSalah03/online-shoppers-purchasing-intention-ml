# online-shoppers-purchasing-intention-ml

data taken from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) and can easily be downloaded using the download button or [this link](https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip) directly.

## problem statement
Online shopping is a land full of observer shoppers without reaching the purchase. in this project, we analyze 12K customer sessions to see what makes a customer decide to commit to a purchase and the seller getting revenue.


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
  - [ ] Model selection process and parameter tuning
- [ ] Script train.py (suggested name)
  - [ ] Training the final model
  - [ ] Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)
- [ ] Script predict.py (suggested name)
  - [x] Loading the model
- [ ] serve.py 
  - [ ] Serving model via a web service (with Flask or specialized software - BentoML, KServe, etc)
- [x] Files with dependencies
  - [x] Pipenv and Pipenv.lock if you use Pipenv or equivalents: conda environment file, requirements.txt or **pyproject.toml**
- [ ] Dockerfile for running the service
- [ ] Deployment
  - [ ] URL to the service you deployed or
  - [ ] Video or image of how you interact with the deployed service