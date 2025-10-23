# Sales-Forecasting-Project

 --- 
 
1. Create Master Weather + Store Subcluster + Holiday + Sales.ipynb
- Data cleaning: selecting and renaming columns, ensuring that columns have the correct format and dtypes, filtering out outliers and data not of interest
- Merging spreadsheets to create the target dataset to apply Machine Learning

---

2. Create Public Holiday Final.ipynb
- Loading downloaded html file from website
- Data cleaning: selecting columns, building dataframes (using Pandas), manipulating column names
- Merging data from different years to form final public holiday data

---

3. Daily Sales Supervised ML with Dupe Dates.ipynb (analysing how public holiday, store subcluster, weather and historical sales data influence sales with multiple supervised ML models)
- Loading dataset created from 1.
- Data cleaning: labelling categories and applying labelencoder(), ensuring data in specific columns have the correct dtypes, spliting columns into "features" and "target"
- Implemented supervised ML models (Polynomial regression, XGBoost Regressor, LightGBM, Feed Forward Neural Network)
*used train_test_split() on dataset, fit the models and compared model predictions to testing data -> analysing evaluation metrics, implementing GridSearchCV for hyperparameter tuning -> showing how much it improved the model, plot feature importance table and used it to eliminate zero-importance columns, built an Embedding and Dense Model, implemented regularisation techniques such as ReduceLROnPlateau() and EarlyStopping(), Plotted a Loss Curve Plot that compares training with validation loss

---

4. DAILY Sales Weather Project.ipynb - Calculate daily correlation between sales and weather with a few ML models
- Data cleaning: filtering outliers, manipulating store data, implementing SMOTE (to address 13:1 disproportional data), merging dataframes
- Created a correlation chart with a conclusion
- Experimented with ML models such as XGBoost and a first attempt of a Multi Layer Perceptrons Neural Network (which was clearly underfitting - led me to believe that weather was insufficient to predict sales data)

---

5. Data for LSTM Model FINAL.ipynb - Jupyter notebook to create the data for the LSTM model
- 🔵 Subcluster data: "Master Store Subcluster" in Lakehouse -> AI_Sandbox -> LH_External_Data -> tables
- 🔵 Daily Sales data: "Daily Sales - Store & VM" in Lakehouse -> AI_Sandbox -> LH_External_Data -> tables
- 🟢 Weather data: can scrape using "Scrape Weather FINAL.ipynb" in this repo
- 🔵 🔴 Public Holiday data: "PH for non-PH" in Microsoft Teams Planner "FM OPS BI" under "Sales Forecasting"
- Need to set where does the code keep output file "Data for LSTM model.csv".

---

6. HOURLY Sales Weather Project FINAL.ipynb - Jupyter notebook to calculate daily correlation between sales and weather with a few ML models
- 🔵 Store data: "Master Store" in Lakehouse -> AI_Sandbox -> LH_External_Data -> tables
- 🔵 Hourly Sales data: (I filtered out only May 2025 data from "Hourly Sales - Store & VM" in Lakehouse -> AI_Sandbox -> LH_External_Data -> tables)
- 🟢 Weather data: can scrape using "Scrape Weather FINAL.ipynb" in this repo

---

7. LSTM Final Model.ipynb - Jupyter notebook to create LSTM and make forecast more about it below.
- 🟢 df: can create using 5. "Data for LSTM Model FINAL.ipynb" in this repo OR 🔵 "Data for LSTM model.csv" in Microsoft Teams Planner "FM OPS BI" under "Sales Forecasting"

---

8. LSTM-tested-features - all of the features that I have tested that did not make the final model
- 🟢 df: can create using "5. Data for LSTM Model FINAL.ipynb" in this repo OR 🔵 "Data for LSTM model.csv" in Microsoft Teams Planner "FM OPS BI" under "Sales Forecasting"
- Need to set where does the code keep output file "Sales_Forecasting_LSTM_Model.h5".

---

9. Sales_Forecasting_LSTM_Model.h5 - model that is created when running model in "LSTM Final Model.ipynb"

---

10. Scrape Weather FINAL.ipynb - Jupyter notebook to scrape weather data from the internet
Do not need any data files, but need to set directory - where the code gets "Master Store.csv" file and where the code keeps final "Weather Data Updated.csv" file.

---

11. Time series.ipynb - Jupyter notebook to categories Public Holiday into categories
- 🔵 "Master Weather + Store Subcluster + Holiday + Sales" csv file in Lakehouse -> AI_Sandbox -> LH_External_Data -> files
- 🔵 Daily Sales data: "Daily Sales - Store & VM" in Lakehouse -> AI_Sandbox -> LH_External_Data -> tables

---

12. Running LSTM without retraining model.ipynb - Jupyter notebook to run my current model without retraining another LSTM model (to make predictions/forecasting)
- 🔵 Under 9. "Sales_Forecasting_LSTM_Model.h5" in this repo
- 🟢 df: can create using 5. "Data for LSTM Model FINAL.ipynb" in this repo OR 🔵 "Data for LSTM model.csv" in Microsoft Teams Planner "FM OPS BI" under "Sales Forecasting"

---

# 4 Main Projects:

1. [📈 Sales Forecasting with Multi-Input LSTM](#-Sales-Forecasting-Multi-Input-LSTM)
2. [Daily Sales Supervised ML Studies](#Daily-Sales-Supervised-ML-Studies)
3. [Daily Sales and Weather Studies](#Daily-Sales-and-Weather-Studies)
4. [Hourly Sales and Weather Studies](#Hourly-Sales-and-Weather-Studies)

---

# 📈 Sales Forecasting with Multi‑Input LSTM

> **Accurate 14‑day window sales & transaction forecasts for FamilyMart Stores, powered by TensorFlow/Keras, categorical embeddings & residual static context.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-lightgrey)

---

## Table of Contents

1. [Features](#lstm-features)
2. [Data](#lstm-data)
3. [Environment/Setup](#lstm-env-setup)
4. [Reproducing the Results](#lstm-reproducing)
5. [Training](#lstm-training)
6. [Evaluation & Metrics](#lstm-eval)
7. [Forecasting & Visualisation](#lstm-forecasting)
8. [Project Structure](#lstm-structure)

---

<a name="lstm-features"></a>

## Features

* **Hybrid input:** numeric weather & sales time‑series, binary flags **and** categorical embeddings (store no, state and subcluster codes).
* **Residual static context:** 2‑layer MLP with skip‑connection broadcasts static store attributes into every LSTM timestep.
* **Time‑aware split per store:** avoids train/val/test leakage across dates.
* **Permutation feature importance** for both *Net Amount* and *Transaction Count* targets.
* **200‑step autoregressive forecasting** & optional 21‑day per‑store interactive dashboard (Plotly + ipywidgets).
* **EarlyStopping** + **ReduceLROnPlateau** for robust convergence.
* **Reproducible:** single script / notebook end‑to‑end, explicit random seed control.

<a name="lstm-data"></a>

## Data

* **Source CSV**: `Data for LSTM model.csv`

| Column type                  | Example columns                                                            | Notes                                     |
| ---------------------------- | -------------------------------------------------------------------------- | ----------------------------------------- |
| **Time‑varying categorical** | `Name`, `Day`, `Month`                                                     | Encoded & *embedded* if > 6 unique values |
| **Static categorical**       | `Store_No`, `State`, `SubCluster`                                          | Embedded, then tiled across timesteps     |
| **Binary flags**             | `Rain?`, `Puasa`, `Public Holiday`                                         | Mapped to 0/1                             |
| **Numeric**                  | `Net_Amount`, `TC`, `Days_after_Opening`, `Average Daily Temperature (°C)` | Min‑Max scaled 0‑1                        |

---

<a name="lstm-env-setup"></a>

## Environment & Dependencies

<details>
<summary>Click to view <code>requirements.txt</code></summary>

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.7     
scikit-learn>=1.3
tensorflow==2.15.0   
keras-tuner>=1.4.2
plotly>=5.18
ipywidgets>=8.1

```

</details>

---

<a name="lstm-reproducing"></a>

## Reproducing the Results

**Match software versions** – see `requirements.txt`.
**Make sure that the data file is correct**.

---

<a name="lstm-training"></a>

## Training

Key hyper‑parameters:

| Flag             | Default | Description                              |
| ---------------- | ------- | ---------------------------------------- |
| `--window`       | 14      | Look‑back window length (days)           |
| `--lstm_units`   | 64      | Hidden size of LSTM layer                |
| `--static_dense` | 128     | Width of residual MLP on static features |
| `--dropout`      | 0.25    | Dropout rate everywhere                  |
| `--lr`           | 1e‑3    | Initial learning rate                    |

---

<a name="lstm-eval"></a>

## Evaluation & Metrics

After training, the script prints:

* **MAE / RMSE / R²** for *Net\_Amount* and *TC*,
  plus intuitive qualitative bands (🔵 Excellent | 🟢 Good | 🟡 Okay | 🔴 Poor).
* PNG plots: loss curves, parity scatter, first‑200‑window overlay.

---

<a name="lstm-forecasting"></a>

## Forecasting & Visualisation

* **200‑step autoregressive demo**
* **21‑day per‑store dashboard** with Plotly + dropdown widget
* Optional **95 % CI** ribbons (constant σ derived from residuals)

---

<a name="lstm-structure"></a>

## Project Structure

```
.
├── data/                     # to obtain data for LSTM
│   └── Data for LSTM Model FINAL.ipynb
├── models/                   # trained model (.h5) saved after training
│   └── Sales_Forecasting_LSTM_Model.h5
├── notebooks/               
│   ├── LSTM Final Model.ipynb    
├── README.md                 
.

```

# Daily Sales Supervised ML studies

> Multimodel experiments (Polynomial Regression, XGBoost, LightGBM & Feed‑Forward NN) to predict **Daily Net Amount (RM)** and **Transaction Count (TC)** from store data, public‑holiday data and local weather data.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-lightgrey)

---

## Table of Contents

1. [Data Preparation](#ml-data-preparation)
2. [Baseline: Polynomial Regression](#ml-baseline)
3. [Tree Boosting](#ml-tree-boosting)

   * 3.1 [XGBoost](#ml-xgboost)
   * 3.2 [LightGBM – four drafts](#ml-lightgbm)
4. [Feed-Forward Neural Network](#ml-ffnn)
5. [Training & Hyper-parameters](#ml-training)
6. [Evaluation Metrics](#ml-eval)
7. [Environment / Setup](#ml-env)
8. [Reproducing the Results](#ml-reproducing)
9. [Project Structure](#ml-structure)

---

<a name="ml-data-preparation"></a>

## Data Preparation

* **Source CSV**: `Master Weather + Store Subcluster + Holiday + Sales.csv`

* **Categoricals** (label‑encoded):

  | Column                       | Description                                       |
  | ---------------------------- | ------------------------------------------------- |
  | `Store_No`                   | Unique store number                               |
  | `Name`                       | Holiday name                                      |
  | `State`                      | Malaysian state                                   |
  | `Day`                        | Day‑of‑week string (Mon…Sun)                      |
  | `CODE (subcluster 1)`        | Store subcluster 1                                |
  | `CODE FY26 1 (subcluster 2)` | Store subcluster 2                                |
  | `CODE FY26 2 (subcluster 3)` | Store subcluster 3                                |
  | `Rain?`                      | 1 if rainfall at any point during the day, else 0 |
  | `Public Holiday`             | 1 if Public Holiday in that state, else 0         |

* **Numericals** (standard‑scaled where noted):

  | Column                           | Units | Notes                            |
  | -------------------------------- | ----- | -------------------------------- |
  | `Days_after_Opening`             | days  | Days a store has been opened for |
  | `Average Daily Temperature (°C)` | °C    | Different for all store location |
  | `Days From Holiday`              | days  | −ve = days *before* PH           |
  | `Puasa Count`                    | days  | Day number during Ramadan        |

* **Targets** (not scaled in tree models):

  * `Net_Amount` (RM)
  * `TC` – Transaction count

---

<a name="ml-baseline"></a>

## Baseline: Polynomial Regression

Simple degree‑2 polynomial on standard‑scaled numericals (categoricals left as integer ids). Wrapped in `MultiOutputRegressor` to predict both targets simultaneously.

---

<a name="ml-tree-boosting"></a>

## Tree Boosting

### <a name="ml-xgboost"></a>XGBoost

* Objective: `reg:squarederror`
* 3‑fold GridSearch over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`.
* Separate models for **Net\_Amount** and **TC**.

### <a name="ml-lightgbm"></a>LightGBM

Four consecutive drafts investigated feature‑drops & categorical handling:

1. **Draft 1** – full feature set
2. **Draft 2** – converts seven high‑cardinality columns to categorical dtypes
3. **Draft 3** – drops `Store_No`
4. **Draft 4** – additionally drops `Rain?`, `Days From Holiday`, `Puasa Count`

Every draft repeats a grid‑search (3‑fold) on common hyper‑parameters: `num_leaves`, `max_depth`, `learning_rate`, `n_estimators`, `reg_lambda`, `min_child_samples`.

---

<a name="ml-ffnn"></a>

## Feed‑Forward Neural Network

* **Architecture**: shared embedding‑tower for 9 categoricals + numeric branch → dense trunk → two task‑specific heads.
* **Embeddings**: size `⎣log₂(cardinality)⎦ + 1`.
* **Hidden layers**: 128‑64‑32 with `ReLU`, `BatchNorm`, `Dropout 0.3`.
* **Loss**: MSE (separate for each head); **Optimiser**: Adam (lr 5 × 10⁻⁴).
* **Callbacks**: EarlyStopping (patience = 3) & ReduceLROnPlateau.
* Training for *max* 50 epochs, batch‑size 32.

---

<a name="ml-training"></a>

## Training & Hyper‑parameters

| Flag / Setting      | Default   | Notes                        |
| ------------------- | --------- | ---------------------------- |
| `test_size`         | 0.20      | `train_test_split` seed = 42 |
| `poly_degree`       | 2         | Baseline model               |
| `ffnn_epochs`       | 50        | Early‑Stopping \~10–15       |
| `ffnn_batch_size`   | 32        |                              |
| `xgb_n_estimators`  | 100‑200   | grid‑search                  |
| `lgb_learning_rate` | 0.05–0.10 | grid‑search                  |

---

<a name="ml-eval"></a>

## Evaluation & Metrics

For each target the notebooks print:

* **RMSE, MAE, R²**
* Ratios vs population `mean` (MAE) and `std` (RMSE) → coloured qualitative bands:

  * 🔵 Excellent 🟢 Good 🟡 Okay 🔴 Poor

---

<a name="ml-env"></a>

## Environment / Setup

<details><summary>requirements.txt</summary>

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
xgboost>=2.0
lightgbm>=4.2

```

</details>

---

<a name="ml-reproducing"></a>

## Reproducing the Results

**Match software versions** – see `requirements.txt`.
**Make sure that the data file is correct**.

---

<a name="ml-structure"></a>

## Project Structure

```
.
├── data/                   
│   └── Create Master Weather + Store Subcluster + Holiday + Sales.ipynb
├── notebooks/               
│   ├── Daily Sales Supervised ML with Dupe Dates.ipynb  
├── README.md                 
.
```

# Daily Sales & Weather Studies

> Exploring how **local weather (rain & temperature)** influences store‑level daily FamilyMart sales in May 2025, then benchmarking several regressors (XGBoost & Neural Networks) to predict Net Amount (RM) and Transaction Count (TC).

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-lightgrey)

---

## Table of Contents

1. [Features](#dailywx-features)
2. [Data](#dailywx-data)
3. [Environment & Dependencies](#dailywx-env)
4. [Reproducing the Results](#dailywx-reproducing)
5. [Training](#dailywx-training)
6. [Evaluation & Metrics](#dailywx-eval)
7. [Project Structure](#dailywx-structure)

---

<a name="dailywx-features"></a>
## Features

* **Step-by-step method for pulling in raw data, cleaning and organizing it, and getting it ready for analysis** – cleans master‑store, hourly sales & weather CSVs and filters only 24‑hour outlets which were open during May 2025.
* **Daily aggregation** – collapses 0‑23 h to Total_Daily_Net_Amount (RM) & Total_Daily_TC (transactions).
* **Rain detection** – maps weather‑code sequences → binary Rain? (Yes/No) and **balances classes** with SMOTE for fair modelling.
* **Exploratory stats** – Pearson correlation and one‑hot rain vs sales analysis.
* **Modelling suite**

  * *XGBoost* baselines & 5‑fold GridSearch tuning for both targets.
  * *MLP (scikit‑learn)* baseline + grid‑search.
  * *TensorFlow MLP* with early‑stopping & LR‑scheduling.
* **Plots & artefacts** – feature‑importance bar plots, loss curves.

---

<a name="dailywx-data"></a>
## Data

Source CSV:
   * Master Store.csv
   * Sales All Stores May 2025.csv
   * Weather Data Updated.csv

| Column Type             | Example columns                            | Notes                                               |
| ----------------------- | ------------------------------------------ | --------------------------------------------------- |
| **Identifier**          | Store_No                                   | numeric id, only 24 h stores kept                   |
| **Temporal**            | Date, Hour                                 | UTC→SGT conversion for weather, later aggregated    |
| **Targets**             | Total_Daily_Net_Amount, Total_Daily_TC     | Net amount negated (–ve to +ve) then summed per day |
| **Weather numeric**     | Average Daily Temperature (°C)             | mean of hourly temps                                |
| **Weather categorical** | Rain?                                      | derived Yes/No flag using WMO codes                 |

---

<a name="dailywx-env"></a>
## Environment & Dependencies

<details>
<summary>Click to view <code>requirements.txt</code></summary>

pandas>=1.5
numpy>=1.23
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
statsmodels>=0.14
meteostat>=1.7


</details>

*Tested on Windows 10 (Python 3.9, 32 GB RAM)*

---

<a name="dailywx-reproducing"></a>
## Reproducing the Results

1. pip install -r requirements.txt.
2. Load raw CSVs into data/ with exact filenames

---

<a name="dailywx-training"></a>
## Training

Key hyper‑parameters examined:

| Flag / Param                     | Default     | Model      | Description               |
| -------------------------------- | ----------- | ---------- | ------------------------- |
| n_estimators                     | 100 / 300   | XGBoost    | boosting rounds           |
| learning_rate                    | 0.1         | XGBoost    | η shrinkage               |
| max_depth                        | 3 / 5 / 7   | XGBoost    | tree depth                |
| subsample / colsample_bytree     | 0.8–1.0     | XGBoost    | row / col sampling        |
| hidden_layer_sizes               | (128,64,32) | scikit‑MLP | grid‑searched alt (64,32) |
| alpha                            | 1e‑4        | scikit‑MLP | L2 regulariser            |
| epochs                           | 100         | TF MLP     | early‑stops \~20          |
| batch_size                       | 32          | TF MLP     |                           |

---

<a name="dailywx-eval"></a>
## Evaluation & Metrics

Each script/notebook prints for **Net Amount & TC**:

* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)
* Coefficient of Determination (R²)
  
---

<a name="dailywx-structure"></a>
## Project Structure 

```
.
├── data/
│   ├── Master Store.csv
│   ├── Sales All Stores May 2025.csv
│   └── Weather Data Updated.csv
├── notebooks/
│   └── DAILY Sales Weather Project.ipynb
└── README.md

```

# Hourly Sales & Weather Studies

> Investigating how temperature and detailed weather codes influence **hour‑by‑hour** Net Amount (RM) and transaction counts (TC) across different FamilyMart stores, then benchmarking linear models and XGBoost.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-lightgrey)

---

## Table of Contents

1. [Features](#hourlywx-features)
2. [Data](#hourlywx-data)
3. [Environment & Dependencies](#hourlywx-env)
4. [Reproducing the Results](#hourlywx-reproducing)
5. [Training](#hourlywx-training)
6. [Evaluation & Metrics](#hourlywx-eval)
7. [Project Structure](#hourlywx-structure)

---

<a name="hourlywx-features"></a>
## Features

* **Step-by-step method for pulling in raw data, cleaning and organizing it, and getting it ready for analysis**

  * Parse master‑store file, derive Start_h/End_h from Operating_Hours.
  * Filter out closed, not‑open‑yet or nameless stores.
  * Expand each store into every trading hour and map hourly sales rows.
  * Convert UTC weather timestamps → local SG/MY time; extract hour & join on (Date, Hour, Store_No).
* **Exploratory statistics** – Pearson r for sales vs temperature & weather codes; one‑hot correlation table.
* **Modelling suite**

  * Baseline **Linear / Ridge / Lasso / ElasticNet** models (with optional Temp×Hour interaction).
  * **XGBoost** draft & 5‑fold GridSearch tuning with feature‑importance plots.
* **Plots & artefacts** – correlation tables, feature‑importance bar charts, train‑vs‑test RMSE comparison.

---

<a name="hourlywx-data"></a>
## Data

Source CSVs:
   * Master Store.csv
   * Sales All Stores May 2025.csv
   * Weather Data Updated.csv

| Column Type             | Example columns                       | Notes                                    |
| ----------------------- | ------------------------------------- | ---------------------------------------- |
| **Identifiers**         | Store_No, Date, Hour            | Hourly granularity, Date in YYYY‑MM‑DD |
| **Sales targets**       | Net_Amount, TC                    | Hourly; net values negated (–ve to +ve)  |
| **Weather numeric**     | Temperature (°C)                    | Derived from Temperature (°C) in raw   |
| **Weather categorical** | Weather_Code, Weather_Description | WMO code (int) & human string            |
| **Store metadata**      | Start_h, End_h                    | For operating‑window filtering           |

---

<a name="hourlywx-env"></a>
## Environment & Dependencies

<details>
<summary>Click to view <code>requirements.txt</code></summary>

pandas>=1.5
numpy>=1.23
matplotlib>=3.7
scikit-learn>=1.3
statsmodels>=0.14
meteostat>=1.7
pytz


</details>

*Developed on Python 3.9 (Windows 10 32 GB RAM).*
TensorFlow is optional (included for future NN experiments).

---

<a name="hourlywx-reproducing"></a>
## Reproducing the Results

1. pip install -r requirements.txt.
2. Load raw CSVs into data/ with exact filenames

---

<a name="hourlywx-training"></a>
## Training

Key hyper‑parameters explored:

| Flag / Param             | Default                | Model      | Description            |
| ------------------------ | ---------------------- | ---------- | ---------------------- |
| n_estimators           | 100/300                | XGBoost    | boosting rounds        |
| learning_rate          | 0.05‑0.1               | XGBoost    | η                      |
| max_depth              | 3‑7                    | XGBoost    | tree depth             |
| subsample/colsample    | 0.8‑1.0                | XGBoost    | row/column sampling    |
| reg_alpha / reg_lambda | 0‑1 / 1‑10             | XGBoost    | L1 / L2 regularisation |
| hidden_layer_sizes     | (128,64,32) vs (64,32) | scikit MLP | grid‑searched          |
| alpha (L2)             | 0.0001‑0.001           | scikit MLP |                        |

---

<a name="hourlywx-eval"></a>
## Evaluation & Metrics

For both **Net\_Amount** and **TC** the scripts print:

* **RMSE**, **MAE**, **R²** on hold‑out 20 % test set.

---

<a name="hourlywx-structure"></a>
## Project Structure

```

.
├── data/
│   ├── Master Store.csv
│   ├── Sales All Stores May 2025.csv
│   └── Weather Data Updated.csv
├── notebooks/
│   └── HOURLY Sales Weather Project FINAL.ipynb
└── README.md

```
