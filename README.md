# Codtech-task-1
# Task 1 – Data Pipeline Development (ETL)

## 📌 Objective
Develop an automated ETL (Extract → Transform → Load) pipeline using Python libraries like `pandas` and `scikit-learn` to clean and transform a customer dataset.

---

## 🧩 Dataset: `sample_customer_data.csv`

| Column       | Description                          |
|--------------|--------------------------------------|
| CustomerID   | Unique customer ID                   |
| Name         | Customer name                        |
| Age          | Age of the customer (some missing)   |
| Salary       | Monthly income (some missing)        |
| Gender       | Male / Female                        |
| City         | Customer’s city                      |

---
## ⚙️ ETL Pipeline Steps

### 1. **Extract**
- Load raw data from `sample_customer_data.csv` using `pandas`.

### 2. **Transform**
- Fill missing numerical values with **median**
- Fill missing categorical values with **mode**
- One-hot encode categorical columns (`Gender`, `City`)
- Normalize numerical columns (`Age`, `Salary`) using `StandardScaler`

### 3. **Load**
- Save the cleaned and transformed dataset to `processed_customer_data.csv`

---

## 🧪 Tools Used
- Python
- Pandas
- scikit-learn
- Google Colab

- ## 📁 Files in This Folder

| File                          | Description                            |
|-------------------------------|----------------------------------------|
| `ETL_pipeline.ipynb`          | Jupyter notebook with complete ETL code and outputs |
| `sample_customer_data.csv`    | Input dataset                          |
| `processed_customer_data.csv` | Final output after transformation      |
| `README.md`                   | This documentation file                |

---

## 🚀 How to Run (Google Colab)

1. Open the notebook `ETL_pipeline.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload `sample_customer_data.csv` in the first cell
3. Run all cells to execute ETL steps
4. Download the output `processed_customer_data.csv` file

## ✅ Notes

- StandardScaler standardizes numeric values (mean = 0, std = 1), so **negative values are expected**
- You may switch to MinMaxScaler if positive-only output is required
