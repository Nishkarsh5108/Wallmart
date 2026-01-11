# Feature Engineering Experiments: Walmart Sales Prediction

This document outlines the strategies and reasoning for feature engineering to improve the sales prediction model.

## 1. Store Number Encoding

Since store IDs are categorical, I need a way to represent them numerically for the model.

* **One-Hot Encoding:** Creates a binary column for each store. This provides a "base value" for each store's sales, which other parameters then adjust.
* **Target/Mean Encoding:** Assigns each store its average sales value from the training data. This effectively captures the "base value" without adding many columns.
* **Embedding Encoding:** Learns a dense vector (e.g., `[0.21, -0.03, 0.88]`) to represent store characteristics.

#### Final Verdict: Target/Mean Encoding

While embeddings are powerful, they risk overfitting given we only have 45 stores. One-hot encoding would create 45 sparse columns, making the data messy. **Target/Mean Encoding** is the best middle ground.

To implement this effectively, I will:

* Use **Out-of-Fold (OOF) encoding** to prevent leakage.
* Calculate mean values using only the training split and apply those same means to the test set.
* Combine this with **Lag Features**, specifically last week's sales and a 4-week rolling average.

---

## 2. Validation Strategy

To simulate real-world forecasting, I will use **Rolling Window Validation**:

1. **Train:** 2010–2011 → **Test:** Early 2012
2. **Train:** 2010–2012 → **Test:** Mid 2012

## 3. Date and Holiday Features

These features will be generated before the train-test split as they are based on the calendar and do not cause data leakage.

* **Time Components:** Year, Week of Year, and Season (Spring, Summer, Fall, Winter). they describe seasonality
* **Holiday Context:** `is_year_end`, `is_year_mid`, weeks until the next holiday, and weeks since the last holiday. there are non official holidays in middle and end of year, people tend to buy more around holidays
* **Time Index:** `time_idx` (total weeks since the start of the dataset) to capture long-term trends.
* **Interaction Feature:** `unemployment * holiday_flag` to see if economic status changes holiday spending habits.

---

## 4. Environmental & Economic Features

### Temperature

When temprature is comfortable: a bit hot in winters and a bit warm in summers; lets people go out of their homes, and this can increase footfall, thereby increasing sales

I want to capture how weather deviations affect shopping behavior rather than just the raw temperature.

* **Z-Score:** `(Temperature - Mean) / SD` to identify seasonal anomalies.
* **Flags:** `abnormally_hot` (Z-score > 1) and `abnormally_cold` (Z-score < -1). (confidence interval of 2 was too much)

### Fuel Price

* **Sudden Changes:** Tracking rapid spikes or drops in fuel costs.
* **Month-End Pressure:** Checking if high fuel prices at the end of the month (when budgets are tight) impact sales.
* **Interaction:** `fuel_price * unemployment`.

### CPI (Consumer Price Index)

* **Trend Analysis:** A 4-week rolling trend (`cpi.diff(1).rolling(4).mean()`) to see if inflation is accelerating.
* **Interactions:** `CPI * unemployment` and `CPI * time_idx`.

### Unemployment

Unemployment has a nuanced effect on Walmart sales. While general spending might drop, sales of essentials often stay steady or increase due to programs like SNAP (EBT cards).

* I could have observed high-value items (electronics) drop while daily essentials remain stable. but the dataset doesn't describe about catagory wise sales breakdown

<pre class="vditor-reset" placeholder="" contenteditable="true" spellcheck="false"><hr data-block="0"/></pre>

# MODEL

I explored different research papers and some kaggle notebooks, and evaluated several candidate architectures for the Walmart sales prediction task.

### 1. Candidate Models

#### A. Gradient Boosting (XGBoost / LightGBM)

* **Performance:** Consistently consistently achieves top accuracy on tabular retail datasets. The research highlights an XGBoost model achieving an $R^2 \approx 0.98$ on specific Walmart studies, outperforming linear methods.
* **Pros:** Handles non-linearities, missing data, and feature interactions (e.g., `Holiday * Unemployment`) exceptionally well. Fast training times.
* **Cons:** "Black box" nature requires post-hoc interpretability tools like SHAP. Requires manual feature engineering for time-series dynamics (lags, rolling means).

#### B. Random Forest

* **Performance:** A robust baseline that handles non-linearity without heavy tuning.
* **Pros:** Less prone to overfitting than un-tuned boosting models. easy to explain via feature importance.
* **Cons:** Generally yields lower accuracy than tuned boosting models on complex data. Can be slow at prediction time with many trees.

#### C. Classical Time Series (ARIMA / Prophet)

* **Performance:** Excel at capturing explicit trends and seasonality.
* **Pros:** Highly interpretable decomposition (Trend + Season + Holiday). Prophet handles holidays well out-of-the-box.
* **Cons:** limited ability to model complex exogenous variables (CPI, Unemployment) and their interactions. Linearity assumptions often fail on volatile retail demand.

#### D. Deep Learning (LSTM / N-BEATS / TFT)

* **Performance:** State-of-the-art for capturing long-term dependencies and complex sequential patterns. Transformers (TFT) offer built-in interpretability via attention weights.
* **Pros:** Automatic feature extraction (can learn seasonal patterns from raw series).
* **Cons:** Data-hungry and computationally expensive. Complex to tune and deploy compared to tree-based models.

### 2. Final Verdict: XGBoost

I selected **XGBoost** as my primary model for the following reasons:

1. **Proven Efficacy:** The research explicitly identifies XGBoost as a top performer for this specific dataset structure, with potential $R^2$ scores nearing 0.98.
2. **Feature Flexibility:** Our dataset includes rich exogenous features (Fuel Price, CPI, Unemployment) and categorical data (Store ID, Holidays). XGBoost handles these heterogeneous data types better than ARIMA/LSTMs.
3. **Balance of Speed & Power:** It offers a pragmatic compromise between the simplicity of Random Forest and the complexity of Deep Learning.
4. **Interpretability:** By integrating **SHAP (Shapley Additive Explanations)**, I can unlock the "black box," providing stakeholder-ready explanations for every prediction (e.g., quantifying the dollar impact of a specific holiday).

### 3. Implementation Strategy

* **Objective:** Regression with Squared Error.
* **Sale differences across stores:**  the model was trained on log-transformed weekly sales, effectively optimizing relative error while retaining squared-error stability
* **Key Hyperparameters:** Learning rate (`eta`), Max depth (to control interactions), and Subsample (to prevent overfitting).
* **Explainability:** Full integration of SHAP TreeExplainer to visualize global feature importance and local prediction drivers.
