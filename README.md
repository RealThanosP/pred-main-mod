# Predictive Maintenance Machine Learning Model

This repository offers a comprehensive framework for **predictive maintenance**, leveraging machine learning techniques to anticipate equipment failures in industrial settings. It covers everything from data preprocessing and feature engineering to model training and deployment via an interactive dashboard.

---

## 🚀 Features

- **Data Preprocessing** – Clean and transform raw industrial sensor data.
- **Feature Engineering** – Extract spectral, statistical, and health indicators.
- **Model Training** – Train classification (failure prediction) and regression (RUL estimation) models.
- **Visualization Dashboard** – Streamlit-based dashboard for real-time monitoring and visualization.
- **Modular Design** – Easy to extend, customize, and plug into other systems.

---

## 📁 Repository Structure

```bash
pred-main-mod/
├── data/                   # Raw and processed datasets
├── data_tables/            # Aggregated feature tables
├── models/                 # Trained models (RandomForest, TensorFlow, etc.)
├── notebooks/              # Jupyter Notebooks for exploratory data analysis
├── scripts/                # Processing and training scripts
├── dashboard/              # Streamlit app files
├── utils/                  # Helper functions
├── README.md               # Project readme file
├── requirements.txt        # Python dependencies
└── LICENSE                 # License file

```
## ⚙️ Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/RealThanosP/pred-main-mod.git
cd pred-main-mod
```

### 2. Create a virtual environment

It's recommended to use a virtual environment to manage dependencies:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Or activate it (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```bash 
pip install -r requirements.txt
```

### 4. Modify your config file accordingly
Add the configuration you wish for your data in the config.json.

### 5. Paste your csv like data in the data folder
Paste the data you want to make a prediction for in the data folder. Now you are all set to use the model.

### 6. Run the data_processing.py
Run data_processing.py and the extracted tables

## 🧪 Example usage from terminal:
```bash
python model_prediction.py --model models/failure_flag_model.pkl --data data_tables/new_sensor_input.csv
```

