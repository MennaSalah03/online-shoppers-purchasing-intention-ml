# online-shoppers-purchasing-intention-ml

data taken from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) and can easily be downloaded using the download button or [this link](https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip) directly.

## problem statement
Online shopping is a land full of lurking shoppers without reaching the purchase phase. This could be for many reasons that may be out of the seller's hand. in this project, we focus on the leads that get customers to potentially buy. we analyze 12K customer sessions to see what makes a customer decide to commit to a purchase and the seller getting revenue so the merchant can make a more informed financial decisions with the leads and get more buying clients.

This is a solution for the merchant for better business decisions and also to get to the right clients that would be satisfied with the purchase and be willing to be a buyer.

The models in `models/` are all named in this format: `<model_family>_<param1>=<val1>_<param2>=<val2>.pkl`

## Trained Models
all the models that were exportede are in the The models in `models/` and named in this format: `<model_family>_<param1>=<val1>_<param2>=<val2>.pkl`

The exported models are considered the best ones from each family (logistic regression, decision tree, random forest).

| Model | auc | acc | f1
| ---   | --- | --- | --- |
| LR | 0.896 | 0.871 | 0.478 |
| DT | 0.914 | 0.897 | 0.592 |
| **RT** | 0.917 | 0.900 | 0.606 |

The chosen model for serving in both the scripts and docker image is the script-trained version of the fine-tuned random tree model and it's named `train_script_rf_d=10_s=10_n=40.pkl`. If you want to try serving a different model you can add the path to the `config.py` and replace the model path variable. In case of a docker image you will have to retrain the whole model of course before serving.

## Try the app

### Option 1: Try it instantly via Docker (Easiest)

You can pull and run the pre-built image directly from Docker Hub. The Docker commands are identical across all operating systems.

**1. Run the container**

```bash
docker run -d -p 8000:8000 menna011/shopping-intention-service:0.1.0

```

**2. Test that it is running**

You can test it on the browser or using the command line.

* **Linux / macOS / Git Bash (using curl):**
```bash
curl http://localhost:8000/health

```

* **Windows (using PowerShell):**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health

```

**3. Run the test script**
Once the server is running, execute the Python test script (`serving_testing.py`) to send sample data to the API:

```bash
python src/online_shoppers_purchasing_intention_ml/serving_testing.py

```

---

### Option 2: Train and serve from scratch locally

If you want to train the Random Forest model yourself and serve it from your local source code, use `uv`. These commands work perfectly on both Linux/macOS and Windows terminals.

**1. Install dependencies**

```bash
# Syncs the exact environment locked in uv.lock
uv sync

```

**2. Train the model**

```bash
# This reads the CSV, trains the model, and saves model.bin
uv run python src/online_shoppers_purchasing_intention_ml/train.py

```

**3. Start the FastAPI server**

```bash
# Start the server (leave this terminal window open)
uv run uvicorn online_shoppers_purchasing_intention_ml.service:app --host 0.0.0.0 --port 8000

```

**4. Test the endpoints**
Open a **new, separate terminal window** and run your test script to verify predictions:

```bash
uv run python online_shoppers_purchasing_intention_ml/serving_testing.py

```

---

## Video demonstration

In this short video, I test the model that is being served on a fasapi API using the browser and the the test file `serving_testing.py`

https://github.com/user-attachments/assets/4d7b8a81-b7b2-486a-8e98-41759bf4d928