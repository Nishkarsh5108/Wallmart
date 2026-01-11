# Walmart Sales Predictor 🛒

Predicting how much a store will sell next week is super important so they don't run out of stock or have too much stuff sitting around (inventory costs money!). This project uses machine learning to predict weekly sales for Walmart stores based on historical data.

## What is this?

I took a dataset containing historical sales, dates, holiday info, temperature, fuel prices, and economic indicators (CPI, Unemployment) to teach a computer model how to guess future sales.

* **`src/main.ipynb`**: The main code where the feature engineering and model training happens.
* **`experiments.md`**: A diary of the features I tried and strategies I used. Read this to understand *why* I manipulated the data the way I did.
* **`conclusion.md`**: The final report card. It breaks down the scores and interprets the graphs.

---

## How to Run It

1. **Install dependencies:**
   You will need Python and these libraries. Run this in your terminal:

   ```bash
   pip install pandas numpy xgboost scikit-learn matplotlib seaborn shap
   ```
2. **Run the notebook:**
   Open `src/main.ipynb` in your favorite editor (VS Code, JupyterLab) and run all cells.
3. **Expected Output:**

   * The code processes the data (cleaning dates, calculating time lags).
   * It trains an **XGBoost** model using a sliding window strategy (simulating real time passing).
   * It prints the **R² Score** (accuracy) and **RMSE** (average error in dollars).
   * It generates plots in the `plots/` directory showing which features matter most.

---

## The "Brain" of the Project 🧠

I used two advanced concepts to make this work. Here is the simple explanation:

### 1. XGBoost (The Prediction Engine)

Think of XGBoost as a team of students taking a test together.

* **Student 1** tries to guess the sales. They make some mistakes.
* **Student 2** doesn't start from scratch; they look *only* at the mistakes Student 1 made and try to fix them.
* **Student 3** looks at the mistakes Student 2 made and fixes those.

This repeats hundreds of times. In the end, you combine all their small "fixes" to get a super accurate prediction. This technique is called **Gradient Boosting**.

**The Math-y bit:**
If $y$ is the real sales number and $\hat{y}_i$ is the prediction at step $i$:

$$
\hat{y}_i = \hat{y}_{i-1} + \eta \cdot f_i(x)
$$

* $f_i(x)$ is the new "student" (a weak learner tree) trying to predict the error (residual) of the previous step.
* $\eta$ (eta) is the "learning rate"—I don't let any single student change the answer too much, just a little nudge to prevent over-correcting.

### 2. SHAP (The Explainer)

Machine learning models can be like black boxes—data goes in, answers come out, and nobody knows why. **SHAP** (SHapley Additive exPlanations) breaks open the box.

Imagine 3 friends pay \$20 for a shared pizza. SHAP calculates exactly how much each person should pay based on how many slices they actually ate.

In my model: (from the waterfall plot at `..plots/basic/waterfall`) (1e5)

* **Base Value:** The average sales across all stores (e.g., \$13.7).
* **Feature Contribution:**
  * If `Weeks_To_Next_Holiday = 1`, SHAP  says: "Add ~+$0.01M" (Shopping rush!).
  * If  `is_month_end_and_High_Fuel_Price `, SHAP says: "Subtract ~-$0.01M" (People have no money and they are driving less).

It helps me trust the model because I can verify that it's using logic that makes sense (e.g., sales go up in December), rather than memorizing random noise.

---

## Key Strategies (The "Secret Sauce")

I didn't just throw raw data at the model. I engineered "smart" features based on how people actually shop:

* **Holiday Distance:** Instead of just a "Is it a holiday?" Yes/No flag, I calculated **"How many weeks until the next holiday?"**. People shop for gifts *before* Christmas, not just on Christmas day.
* **Target Encoding:** I replaced the `Store ID` (which is just a random number like 1, 2, 3) with the **Average Sales** of that store. This gives the model a strong baseline immediately.
* **Lag Features:** I fed the model the sales from **Last Week**. If sales were high last week, they usually stay high. This captures momentum.
* **Handeliing scale differences across stores :** I trained the model on `log transformsed weekly sales`, effectively optimising relative error while retaing squared error staibility

*For a full breakdown of these strategies, read `experiments.md`.*

## Results

The model performed exceptionally ell, achieving an **R² score of ~0.98**. and **MAPE 4.45%** This means it successfully predicted weekly sales with mean average 4.45% error (relative to the store's actual sales)

*For detailed performance metrics and plot interpretations, read `conclusion.md`.*
