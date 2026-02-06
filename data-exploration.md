# Exploratory Data Analysis – Assumptions and Limitations

## 1. Dataset Overview

The dataset contains:

* 57,000 records
* 17 columns
* Memory usage of approximately 7.4 MB

### Column List

1. Airline
2. Source
3. Source Name
4. Destination
5. Destination Name
6. Departure Date & Time
7. Arrival Date & Time
8. Duration (hrs)
9. Stopovers
10. Aircraft Type
11. Class
12. Booking Source
13. Base Fare (BDT)
14. Tax & Surcharge (BDT)
15. Total Fare (BDT)
16. Seasonality
17. Days Before Departure

---

## 2. Missing Value Assessment

Using `df.isnull()` and `df.info()`, all 57,000 rows contain non-null values across all 17 columns.

### Assumption

No missing value imputation is required at this stage.

---

## 3. Data Type Assessment

### Numerical Columns

* Duration (hrs) – float64
* Base Fare (BDT) – float64
* Tax & Surcharge (BDT) – float64
* Total Fare (BDT) – float64
* Days Before Departure – int64

### Categorical Columns

* Airline
* Source
* Source Name
* Destination
* Destination Name
* Stopovers
* Aircraft Type
* Class
* Booking Source
* Seasonality

### Date Columns (Currently Stored as String)

* Departure Date & Time
* Arrival Date & Time

### Assumption

Departure Date & Time and Arrival Date & Time must be converted to datetime format during preprocessing.

---

## 4. Categorical Data Analysis

### Cardinality

* Airline: 24 unique values
* Source: 8 unique values
* Source Name: 8 unique values
* Destination: 20 unique values
* Destination Name: 20 unique values
* Stopovers: 3 unique values
* Aircraft Type: 5 unique values
* Class: 3 unique values
* Booking Source: 3 unique values
* Seasonality: 4 unique values

### Structural Observations

* Source and Source Name represent airport code and full airport name.
* Destination and Destination Name represent airport code and full airport name.
* Stopovers contains: Direct, 1 Stop, 2 Stops.
* Class contains: Economy, Business, First Class.
* Booking Source contains: Online Website, Travel Agency, Direct Booking.

### Assumption

Airport codes correctly map to their corresponding airport names. No inconsistent categorical entries were detected during initial inspection.

---

## 5. Numerical Range Analysis

Descriptive statistics were generated for:

* Duration (hrs)
* Base Fare (BDT)
* Tax & Surcharge (BDT)
* Total Fare (BDT)
* Days Before Departure

### Observed Ranges

* Duration (hrs): 0.50 to 15.83
* Base Fare (BDT): 1,600.98 to 449,222.93
* Tax & Surcharge (BDT): 200.00 to 73,383.44
* Total Fare (BDT): 1,800.98 to 558,987.33
* Days Before Departure: 1 to 90

### Observations

* No negative values are present.
* Duration values include both short-haul and long-haul flights.
* Fare-related columns show high variance.
* Days Before Departure is bounded between 1 and 90.

---

## 6. Outlier Detection (IQR Method)

Outliers were calculated for:

* Duration (hrs)
* Base Fare (BDT)
* Tax & Surcharge (BDT)
* Total Fare (BDT)
* Days Before Departure

### Outliers Detected

* Duration (hrs): 5,984
* Base Fare (BDT): 3,772
* Tax & Surcharge (BDT): 1,226
* Total Fare (BDT): 3,409
* Days Before Departure: 0

### Interpretation

Outliers are concentrated in pricing and duration-related columns. Days Before Departure does not contain statistical outliers.

Lower bounds calculated by the IQR method are negative for some fare columns; however, no actual negative values exist in the dataset.

---

## 7. Business Logic Assumptions

The dataset includes:

* Base Fare (BDT)
* Tax & Surcharge (BDT)
* Total Fare (BDT)

### Assumption

Total Fare (BDT) should equal:

Base Fare (BDT) + Tax & Surcharge (BDT)

This relationship will be validated during the data cleaning phase.

---

## 8. Identified Limitations

1. Date columns are stored as string and require conversion.
2. No external validation source is available to verify pricing or duration accuracy.
3. The dataset does not contain flight distance, passenger demand, or aircraft capacity.
4. Seasonality is predefined and not derived from actual departure dates.

---

## Conclusion Before Data Cleaning

* The dataset is structurally complete.
* Numerical features contain statistical outliers.
* Categorical features are consistent based on initial inspection.
* Datetime transformation and business rule validation are required before modeling.
