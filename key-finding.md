# Flight Fare Prediction – Baseline Model Findings

## **1. Project Overview**

This project predicts **Total Fare** for flights using historical flight data. Features include:

* Flight details (`Airline`, `Source`, `Destination`, `Aircraft Type`)
* Booking information (`Booking Source`, `Class`, `Stopovers`)
* Dates and times (`Departure Date & Time`, `Arrival Date & Time`)
* Numerical metrics (`Base Fare`, `Tax & Surcharge`, `Duration`, `Days Before Departure`)

The **goal** is to build a predictive model, evaluate it, and document insights.

---

## **2. Data Preprocessing & Data Leakage Prevention**

### **Step 0: Column Renaming & Type Conversion**

* Renamed columns for consistency (`Total Fare (BDT)` → `Total Fare`)
* Converted numeric columns safely to numeric type using `pd.to_numeric`.

### **Step 1: Handle Missing Values**

* **Numeric features**: filled missing values with **median** (computed on entire column before train-test split).
* **Categorical features**: filled missing values with `"Unknown"`.

### **Step 2: Remove Negative Fare Values**

* Rows with negative values in `Base Fare`, `Tax & Surcharge`, `Total Fare` were removed.

### **Step 3: Feature Engineering (Dates)**

* Extracted features: `Departure Month`, `Departure Day`, `Departure Weekday`, `Arrival Month`, `Arrival Day`, `Arrival Weekday`.
* Dropped original datetime columns to avoid duplication.

### **Step 4: Encode Categorical Variables**

* **Ordinal encoding**:

  * `Stopovers`: Direct → 0, 1 Stop → 1, 2 Stops → 2
  * `Class`: Economy → 0, Business → 1, First → 2
  * `Booking Source`: Online Website → 0, Travel Agency → 1, Direct Booking → 2
* **One-hot encoding**:

  * `Airline`, `Source`, `Destination`, `Aircraft Type`, `Season`

### **Step 5: Separate Target Before Scaling**

* Target column (`Total Fare`) is **separated before scaling**.
* **Base Fare** and **Tax & Surcharge** removed from features to prevent leakage into `Total Fare` prediction.

> **Why this prevents data leakage:**
>
> * Scaling, encoding, and any transformations applied **only to training features**, never including the target.
> * Features derived from the target (like Base Fare + Tax) were excluded before training.
> * This ensures the model **does not “peek” at the target** during preprocessing.

### **Step 6: Train-Test Split**

* Split dataset **80% training / 20% testing** **before scaling**, to avoid leaking test information.

### **Step 7: Feature Scaling**

* Only **numerical features** in the training set were scaled with `StandardScaler`.
* The same scaler was applied to the test set, **without fitting**.

> This preserves the integrity of the test set and ensures the model is evaluated fairly.

### **Step 8: Target Transformation**

* Target variable (`Total Fare`) log-transformed using `log1p` to handle skew.
* Prediction outputs were inverse-transformed (`expm1`) before evaluation.

---

## **3. Outlier Analysis**

* Using **IQR method**, ~2,724 outliers (~6% of training data) were detected.
* Optionally, these could be:

  * **Dropped** → reduces extreme residuals.
  * **Winsorized** → capped at threshold values to preserve most data.

> For the baseline model, outliers were retained, but later models could explore handling them.

---

## **4. Baseline Model – Linear Regression**

### **Training**

* Trained on **log-transformed Total Fare** to reduce skew effects.
* Categorical variables encoded, numeric features scaled.

### **Evaluation Metrics**

| Metric | Value     |
| ------ | --------- |
| R²     | 0.6546    |
| MAE    | 28,515.94 |
| RMSE   | 47,985.02 |

### **Observations**

1. **Strengths**

   * Predicts small fares reasonably well.
   * Log-transform improves performance for long-tail distribution.
   * No data leakage: model only sees features from training data.

2. **Weaknesses**

   * Medium and high fares show large residuals.
   * Linear model cannot capture non-linear interactions.
   * Outliers still affect high-end fare predictions.

### **Residual Analysis**

* Residuals vs predicted values show patterns at extremes.
* Most mid-range fares show random scatter, indicating acceptable linear fit.

### **Actual vs Predicted (Top 10)**

| Actual | Predicted | Residual |
| ------ | --------- | -------- |
| 3,444  | 3,755     | -312     |
| 69,615 | 36,218    | 33,397   |
| 2,599  | 3,652     | -1,053   |
| 46,214 | 129,366   | -83,152  |
| 56,483 | 35,816    | 20,667   |
| 11,410 | 33,494    | -22,085  |
| 73,871 | 76,148    | -2,277   |
| 93,285 | 79,426    | 13,859   |
| 2,184  | 3,175     | -990     |
| 7,945  | 6,567     | 1,378    |

---

## **5. Key Takeaways**

* **Data leakage avoided** by:

  * Splitting dataset before scaling and transformations.
  * Excluding features derived from the target (`Base Fare + Tax`).
  * Scaling applied only on training features.
* Linear Regression baseline performs reasonably (R² ~0.65).
* High-value fares and outliers still challenge prediction.
* Proper preprocessing and encoding are **critical** to ensure fair evaluation.

---

## **6. Next Steps**

1. Experiment with **more complex models**:

   * Random Forest, XGBoost, LightGBM (non-linear relationships)
   * Ridge/Lasso regression (regularization)
2. Explore **outlier handling** to improve performance.
3. Use **cross-validation** to ensure model stability.
4. Analyze **feature importance** for interpretability.

