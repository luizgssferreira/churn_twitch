# Churn Twitch


## 📌 Context

An end-to-end project on churn prediction and Recency, Frequency, Value (RFV) analysis for Twitch users. It identifies at-risk users and segments customers based on engagement and value, enabling targeted retention strategies and personalized marketing to maximize customer lifetime value.

We utilize several advance techiniques, that are described in the steps bellow:

## Steps 

Starting from the transactional database 'database.db' which i will refer as the bronze layer we will do this:

- Create the Feature Store, with 62 features related of how users interact in time with the Twitch points system
- Create cohorts of user interaction in the points system
- Construction of the target variable based upon the business rule
- Construction of the Analytical Base Table (ABT)
- Create the Out of Time (OOT) database 
- Construction of the Recency Frequency Value (RFV) for user segmentation
- Training of predictive model using SEMMA
- Deploy using MLFLOW

## 💼 Bussiness Problem

Twitch is the most relevant plataform for livestreams in the world, everyday, millions are connect and interacting with live content for diverse public, such as games, music, arts but also programming and data science live. Twtich has enmbended a points system to recompense from diferent interactions with the chat bot, and this points can be futher redeemed in prizes associated with the plataform, such as highlighted mesages, emotes, but also it gives the possibilide of the content creator to have it own prize system, where users can change this points to products that are avalible by the twitch content creator, this created an market with milions of transactions per day, wich can be capitilized by the content creator. In this sense, losing active members in the community is a loss in potential capital growth for the creator, so, knowing with users are more probable or unprobled to exit the active base, and how this active base can be  segmented can give meaninful insights for decesion making focusing the retention of those groups. 


## Dataset

As our users interact during broadcasts, they earn points. They can then accumulate them to exchange for rewards in the store, as well as perform live actions. This will use a real world dataset from Twitch content creator Teo Calvo (Teo Me Why) loyality system, with transactional from the Twitch Points System and can be acessed in this link: https://www.kaggle.com/datasets/teocalvo/teomewhy-loyalty-system/data

### Database Schema (Bronze Layer) 

#### `transactions_products`

| Column              | Description                                              |
|---------------------|----------------------------------------------------------|
| `idTransactionCart` | Unique identifier for the transaction-product pair.      |
| `idTransaction`     | Unique identifier for the transaction.                  |
| `NameProduct`       | Name of the product included in the transaction.         |
| `QuantityProduct`   | Quantity of the product in the transaction.              |

---

#### `customers`

| Column             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `idCustomer`       | Unique identifier for the customer (primary key).                          |
| `PointsCustomer`   | Current points balance, updated with every new transaction for the customer.|
| `flEmail`          | Flag indicating if the customer has a registered email (0 = No, 1 = Yes).  |

---

#### `transactions`

| Column             | Description                                                                     |
|--------------------|---------------------------------------------------------------------------------|
| `idTransaction`    | Unique identifier for the transaction (primary key).                           |
| `idCustomer`       | Foreign key referencing the customer.                                          |
| `dtTransaction`    | Date and time of the transaction (UTC-0).                                      |
| `pointsTransaction`| Points associated with the transaction (positive = points earned, negative = points redeemed). |

---

## 🛠 Data pre-processing and creating the Feature Store database (Silver Layer) 

### What is a Feature Store? 

The Feature Store is a elegant data system used for machine learning, it will serve as a central hub for storing, processing and acessing our intereset functions. In this way we will have a organic enviroment with reusable features in future machine models, by doing this approach we have a a automated process for creating the features, permiting novel ingestions, novel features and novel churn definitions by just changing a few lines of the code ir our queries. In this way we will not directly modify the production databse, but create a novel database, using feature engeenigeering. 

We will start from our bronze layer 'database.db' and use SQL CTE's to create our feature_store.db, with will be the base for the Analytical Base Table where will train and test or churn models and create the user segmentation by RFV 

The Feature Store database will be composed by this four tables: 

# Feature Store: `fs_general.sql`
---

## Feature Store Queries

The following queries are part of the feature store and are used to generate feature sets for analysis. Each query aggregates customer transaction data, providing insights into customer behavior over different time periods.

### Query: fs_hour.sql

**Description**: This query calculates features related to transaction times and customer points accumulation, focusing on daily patterns of activity.

#### Tables Used
- **transactions**: Contains customer transaction data, including transaction IDs, amounts, and timestamps.

#### Variables and Descriptions

| **Variable**          | **Description**                                                                                       |
|------------------------|-------------------------------------------------------------------------------------------------------|
| dtRef               | Reference date for the features.                                                                      |
| idCustomer          | Unique identifier for the customer.                                                                   |
| pointsMorning       | Total points earned in the morning (8:00 AM - 12:00 PM).                                              |
| pointsAfternoon     | Total points earned in the afternoon (12:00 PM - 6:00 PM).                                            |
| pointsEvening       | Total points earned in the evening (6:00 PM - 11:00 PM).                                              |
| pctPointsMorning    | Percentage of total points earned in the morning.                                                     |
| pctPointsAfternoon  | Percentage of total points earned in the afternoon.                                                   |
| pctPointsEvening    | Percentage of total points earned in the evening.                                                     |
| transactionsMorning | Total number of transactions made in the morning.                                                     |
| transactionsAfternoon | Total number of transactions made in the afternoon.                                                 |
| transactionsEvening | Total number of transactions made in the evening.                                                     |
| pctTransactionsMorning | Percentage of total transactions made in the morning.                                              |
| pctTransactionsAfternoon | Percentage of total transactions made in the afternoon.                                          |
| pctTransactionsEvening | Percentage of total transactions made in the evening.                                              |

---

### Query: fs_points.sql

**Description**: This query calculates weekly and lifetime points balances, cumulative points earned, redeemed points, and average points per day for each customer.

#### Tables Used
- **transactions**: Contains customer transaction data, including transaction IDs, amounts, and timestamps.

#### Variables and Descriptions

| **Variable**          | **Description**                                                                                       |
|------------------------|-------------------------------------------------------------------------------------------------------|
| dtRef               | Reference date for the features.                                                                      |
| idCustomer          | Unique identifier for the customer.                                                                   |
| balancePointsW3     | Total points balance for first week.                                                            |
| balancePointsW2     | Total points balance for the second week.                                                            |
| balancePointsW1     | Total points balance for the third week.                                                               |
| pointsCumulativeW3  | Total cumulative points earned in the third week.                                                   |
| pointsCumulativeW2  | Total cumulative points earned in the second week.                                                   |
| pointsCumulativeW1  | Total cumulative points earned in the first week.                                                      |
| pointsRedeemW3      | Total points redeemed in the third week.                                                            |
| pointsRedeemW2      | Total points redeemed in the second week.                                                            |
| pointsRedeemW1      | Total points redeemed in the first week.                                                               |
| pointsBalance       | Lifetime total points balance.                                                                        |
| pointsCumulativeLife| Lifetime cumulative points earned.                                                                    |
| pointsRedeemLife    | Lifetime total points redeemed.                                                                       |
| daysLife            | Account age in days, from the earliest transaction date to the reference date.                        |
| pointsPerDay        | Average points redeemed per day over the customer's lifetime.                                         |

---

### Query: fs_general.sql

**Description**: This query calculates recency, frequency, total points for the last 21 days, and other customer metrics like base age and email flag.

#### Tables Used
- **transactions**: Contains customer transaction data, including transaction IDs, amounts, and timestamps.
- **customers**: Contains customer data, including email flag.

#### Variables and Descriptions

| **Variable**      | **Description**                                                                                       |
|-------------------|-------------------------------------------------------------------------------------------------------|
| dtRef             | Reference date for the features.                                                                      |
| idCustomer        | Unique identifier for the customer.                                                                   |
| recencyDays       | Days since the most recent transaction (recency).                                                     |
| frequencyDays     | Number of distinct days with transactions in the last 21 days (frequency).                             |
| pointsValue       | Total points earned in the last 21 days.                                                              |
| baseAgeDays       | Number of days since the customer's last transaction (base age).                                      |
| flEmail           | Flag indicating if the customer has an email registered.                                              |

---

### Query: fs_products.sql

**Description**: This query calculates product-specific features such as quantities, points, and percentages for the most frequent products purchased by customers in the last 21 days.

#### Tables Used
- **transactions**: Contains customer transaction data, including transaction IDs, amounts, and timestamps.
- **transactions_product**: Contains details about products in each transaction, including product names and quantities.

#### Variables and Descriptions

| **Variable**          | **Description**                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------|
| dtRef                 | Reference date for the features.                                                                      |
| idCustomer            | Unique identifier for the customer.                                                                   |
| qtyChatMessage        | Total quantity of 'ChatMessage' product purchased by the customer.                                     |
| qtyAttendanceList     | Total quantity of 'Lista de presença' product purchased by the customer.                              |
| qtyRedeemPony         | Total quantity of 'Resgatar Ponei' product purchased by the customer.                                 |
| qtyPointsExchange     | Total quantity of 'Troca de Pontos StreamElements' product purchased by the customer.                 |
| qtyStreakPresence     | Total quantity of 'Presença Streak' product purchased by the customer.                                |
| qtyAirflowLover       | Total quantity of 'Airflow Lover' product purchased by the customer.                                  |
| qtyRLover             | Total quantity of 'R Lover' product purchased by the customer.                                        |
| pointsChatMessage     | Total points earned from 'ChatMessage' product.                                                       |
| pointsAttendanceList  | Total points earned from 'Lista de presença' product.                                                |
| pointsRedeemPony      | Total points earned from 'Resgatar Ponei' product.                                                   |
| pointsPointsExchange  | Total points earned from 'Troca de Pontos StreamElements' product.                                    |
| pointsStreakPresence  | Total points earned from 'Presença Streak' product.                                                  |
| pointsAirflowLover    | Total points earned from 'Airflow Lover' product.                                                    |
| pointsRLover          | Total points earned from 'R Lover' product.                                                          |
| pctChatMessage        | Percentage of total quantity of 'ChatMessage' product purchased relative to all product quantities.   |
| pctAttendanceList     | Percentage of total quantity of 'Lista de presença' product purchased relative to all product quantities.|
| pctRedeemPony         | Percentage of total quantity of 'Resgatar Ponei' product purchased relative to all product quantities.|
| pctPointsExchange     | Percentage of total quantity of 'Troca de Pontos StreamElements' product purchased relative to all product quantities.|
| pctStreakPresence     | Percentage of total quantity of 'Presença Streak' product purchased relative to all product quantities.|
| pctAirflowLover       | Percentage of total quantity of 'Airflow Lover' product purchased relative to all product quantities. |
| pctRLover             | Percentage of total quantity of 'R Lover' product purchased relative to all product quantities.     |
| avgChatLive           | Average quantity of 'ChatMessage' product purchased per distinct transaction day.                    |
| maxQuantityProduct    | The product with the highest quantity purchased by the customer in the last 21 days.                  |


## Pipeline for Creating the Feature Store (`exec.sh`)

In this project, we leverage the `exec.sh` pipeline to trigger `execute.py`, which handles the integration of SQL queries with Python for ingesting and constructing the `feature_store.db`. This approach ensures that every time new entries are inserted into the production database, we can elegantly perform the data ingestion process via a Bash script.

### Workflow Overview
- **SQL Queries**: The queries are parameterized with `{date}` as a placeholder. This allows us to specify the time period when ingesting data into the feature store.
- **Date Range Handling**: Using the `argparser`, the user can set the start and end dates for the feature store ingestion. For this specific project, the start date (`x`) and end date (`y`) are configured, allowing us to work with `z` cohorts.
- **SQLAlchemy Integration**: The integration between SQLite queries and Python code is facilitated through SQLAlchemy. It enables seamless interaction with both the origin and target databases.

The Python script (`execute.py`) performs the following tasks:
1. **SQL Query Execution**: The script imports SQL queries, formats them with the desired date range, and executes them against the production database.
2. **Data Ingestion**: After fetching the data, the script deletes existing records from the target database for the specified date, then appends the new data.
3. **Batch Processing**: It processes the data for a date range and appends it to the target feature store database, allowing for continuous integration of new data over time.

## Creating the Analytical Base Table (ABT)

### Creating the Target Variable: Flag Churn

A customer is considered to have churned if they have not engaged with the service or made a transaction for **at least 21 days after their last recorded activity**, and their account has no further activity within that period.

In the given `abt.sql` query, a **customer** is marked as having **churned** (i.e., the `flChurn` flag is set to 1) if there is no record for that customer in the `fs_general` table **21 days after** their last recorded activity (`dtRef`).

#### Logic:
1. The query checks whether a customer has a transaction (or record) in the `fs_general` table exactly **21 days after** their last recorded date (`t1.dtRef`).
2. If no such record exists (`t2.idCustomer IS NULL`), the `flChurn` flag is set to **1**, indicating that the customer has churned (i.e., no transaction or activity 21 days after their last one).
3. If a record does exist for the customer 21 days after their last recorded activity, the `flChurn` flag is set to **0**, indicating the customer has not churned.

### Joining Feature Store Tables and Target to Create the ABT

Once the target variable is defined, the `abt.sql` query joins all the features created in the feature store into a single **Analytical Base Table (ABT)**. This ABT serves as the foundation for modeling.

Note that the project flow is designed to be flexible: if the business definition of churn changes, we can simply adjust this section of the query and re-run the training process to accommodate the new business requirements!


# SEMMA Model Integration with MLflow for Churn Prediction

This script integrates the SEMMA methodology with MLflow for churn prediction modeling. The SEMMA model is a widely-used data mining methodology developed by SAS that guides the data science process. SEMMA stands for:

- **Sample**: Extract a representative sample of the data that is large enough to be meaningful but small enough to manage.
- **Explore**: Identify patterns and anomalies in the data.
- **Modify**: Transform the data to focus the model by creating, selecting, and transforming variables.
- **Model**: Apply modeling techniques to the data to create predictive models.
- **Assess**: Evaluate the accuracy and reliability of the model.

This script follows these principles and uses MLflow to track and log experiments. Various classifiers are evaluated with hyperparameter tuning, and models are assessed using metrics such as accuracy, ROC AUC, precision, and recall.

## Steps:

1. **Data Import & Preprocessing**:
    - Data is loaded from an SQLite database using a SQL query (`abt.sql`).
    - The dataset is split into **training** and **out-of-time (OOT)** sets. The OOT set consists of the most recent data, representing a future snapshot for model validation, simulating real-world performance.

2. **Feature Engineering**:
    - Categorical variables are encoded using `OneHotEncoder` to transform features as needed.
    - The model is configured to handle time-series-like data using stratified sampling, ensuring proper temporal validation for churn prediction.

3. **Model Selection & Hyperparameter Tuning**:
    - Multiple classifiers (e.g., `DecisionTreeClassifier`) are evaluated using `GridSearchCV` to identify the best hyperparameters.
    - A **BaggingClassifier** is used with a **DecisionTreeClassifier** as the base estimator.
    - Hyperparameters such as `n_estimators`, `max_samples`, `max_features`, `bootstrap`, and `bootstrap_features` are tuned.

4. **Model Evaluation**:
    - Model performance is evaluated using **cross-validation** on the training set, and the best hyperparameters are selected based on the **ROC AUC score**.
    - The models' performance is assessed on **train**, **test**, and **out-of-time (OOT)** sets. Metrics such as **accuracy**, **ROC AUC**, **precision**, and **recall** are logged into MLflow.

5. **MLflow Integration**:
    - MLflow is used to log metrics and parameters during model training and hyperparameter tuning.
    - The best-performing model is serialized using `pandas.Series` and saved for future use.

In this workflow, `train_mlflow.py` and `predict_mlflow.py` handle the SEMMA process. Several classifiers were tested in MLflow, with the OOT data providing the best indication of model adjustment. After evaluating the performance metrics and hyperparameter testing with GridSearchCV, the top three models selected were **RandomForest**, **GradientBoosting**, and **BaggingClassifier** using decision trees. After careful evaluation, the **RandomForest** model was chosen for production, with the following performance metrics:

| **Base** | **Accuracy** | **ROC AUC** | **Precision** | **Recall** |
|----------|--------------|-------------|---------------|------------|
| Train    | 0.77         | 0.85        | 0.72          | 0.78       |
| Test     | 0.76         | 0.81        | 0.70          | 0.83       |
| OOT      | 0.73         | 0.80        | 0.68          | 0.65       |

## In-Depth Analysis of Model Performance in the Context of Customer Churn

In the script `best_model.py`, we conduct an in-depth analysis of our model's performance, focusing on its implications for addressing the churn problem in our business context. The key findings from this analysis are visualized in the `train > plots` directory and discussed below.

### Evaluating Model Recall through Cumulative Gains

To assess our model's ability to identify churners, we analyze how it performs across different probability thresholds. The model ranks users by their likelihood of churn, allowing us to prioritize the top segments based on predicted risk. Let’s break this down:

1. **Top 20% of Users by Predicted Churn Probability**  
   By targeting the top 20% of users most likely to churn (as predicted by the model), we capture nearly 40% of all actual churners in the dataset. This is **double the effectiveness** of a baseline random model, which would only capture 20% of churners in the same segment.

2. **Top 30% of Users**  
   Expanding to the top 30% of users, we capture about 50% of actual churners.

3. **Full Dataset**  
   By covering approximately 70% of the dataset, we can capture 100% of actual churners. This progression illustrates the model's ability to prioritize effectively, as seen in the cumulative gain plot below:

   **[Insert cumulative gain plot here]**

### Lift Curve: Model Effectiveness Compared to Random Baseline

The lift curve further demonstrates the effectiveness of our model. It quantifies how much better the model performs compared to random guessing. For example:

- When we focus on the 100 users with the highest predicted probabilities of churn:
  - **Model Performance**: Among these 100 users, 88% are true churners.
  - **Baseline Performance**: Without the model, only 44% of these users would be true churners, based on the overall churn rate.

This indicates the model is nearly **twice as effective** at identifying churners compared to a random baseline. These results are consistent with the lift curve, showcasing the model's strong predictive power.

### KS Statistic: Measuring Model Discrimination

To further validate the model's performance, we use the KS (Kolmogorov-Smirnov) statistic. This metric measures the model's ability to distinguish between churners (class 1) and non-churners (class 0). 

#### How the KS Statistic Works
- The KS statistic calculates the maximum distance between the cumulative distribution of churners and non-churners across ordered probabilities.
- A higher KS value indicates better discrimination. Ideally, the model should produce well-separated probability distributions for the two classes.

#### Results
Using the following code:

```python
skplt.metrics.plot_ks_statistic(y_test, y_test_proba, title="KS Statistic (Class 1 = Churn)")
```

[inser ks plot here]

This results show a strong separtion between the two curves, confirming that the model effectively distinguish between churners and non-churners

## Segmenting active user database usin Recency, Frequency and Value

## User Life Cycle trought rfv_recency.py

In this section we will assigns a lifecycle stage to a user based on their recency and age in the base.
What is User Life Cycle? 

Its a way to describe the different steps of an user engagement, we can use this to see a overview of our active base, classifying and counting the number for each life cycle step 

We will fetch the most recent results in our fs_general table. We will use the cumulative recency curve to agroup our in 6 stages of user life cycle based upon the recency and the age in the active base
it will follow this rules

```python

  if row['baseAgeDays'] <= 7:
        return 'New User'
    elif row['recencyDays'] <= 2:
        return 'Super Active User'
    elif row['recencyDays'] <= 6:
        return 'Active User'
    elif row['recencyDays'] <= 12:
        return 'Cold Active'
    elif row['recencyDays'] <= 18:
        return 'Unengaged'
    else:
        return 'Pre-Churn'


then we will use an decision tree model to atuomatate the classification 

X = result[['recencyDays', 'baseAgeDays']].values  # Using recencyDays mean as features
y = result.index  # Lifecycle stages as labels

# Train a Decision Tree Classifier
clf = tree.DecisionTreeClassifier(min_samples_leaf=1, max_depth=None, random_state=42)
clf.fit(X, y)

# Save the trained model and feature names for future use
model = {
    "model": clf,
    "features": ['recencyDays', 'baseAgeDays']  # Use the column used in training
}

````

[recency plot] 

The code in rfv_recency.py will result in this table: 

| **LifeCycle**         | **recencyDays (mean)** | **count** | **baseAgeDays (mean)** |
|------------------------|------------------------|-----------|-------------------------|
| Active User           | 3.59                  | 82        | 71.45                  |
| Cold Active           | 9.25                  | 63        | 82.84                  |
| New User              | 2.89                  | 56        | 3.63                   |
| Pre-Churn             | 21.00                 | 23        | 50.87                  |
| Super Active User     | 1.23                  | 137       | 81.04                  |


We see two interesting things here. Super Active Users have in mean the largest age in the base, losing only to Cold Actives, they also have the lower recency. With mean that this class, with represent the largest group in our database, are older user that have high recency. 

The second is that Pre-Churn users have the lower mean rencency, they interact with the live a lot less than any other category, this relation is so high that an pre-churn user have the recency in the live 20x lesser than a super active user, and are newer by almost haf in base age than the superactive ones but also have the lowest mean age in the base.

This is helpfull because we want to adress differently users based upon their lifestyle stage, thinking about retention.

## User Frequency and value trought rfv_frecency.py

We the LifeCycle table at hands the next step will be creating the segmentation though frequency and value. We will use our frequencyDays variable and pointsValue for value, witch represents the total points earned in the last 21 days.  

[rfv scatter] 

I will start with a simple scatterplot of frequency vs value. Then i wil contruct a "dull" clustering algorithm using aglomerative clustering. this will help as a sugestion of how our users can be segmented under those two features. With this initial painting, we can start based upon the headlines of our model, drawing our thresholds. Here i divide these initial groupíng in 12 segments as following: 

        "LL": "Low Value, Low Frequency",
        "LM": "Low Value, Medium Frequency",
        "LH": "Low Value, High Frequency",
        "LV": "Low Value, Very High Frequency",
        "ML": "Medium Value, Low Frequency",
        "MM": "Medium Value, Medium Frequency",
        "MH": "Medium Value, High Frequency",
        "MV": "Medium Value, Very High Frequency",
        "HL": "High Value, Low Frequency",
        "HM": "High Value, Medium Frequency",
        "HH": "High Value, High Frequency",
        "HV": "High Value, Very High Frequency"

This is what the segmentation looks like:

[rfv thresholds plot]

Then i can use this thresholds from the above figure to re-run a cluster algorithm, and apply thoes labes to our users. 


```python
def rf_cluster(row):
    """
    Assign RF clusters based on pointsValue and frequencyDays using predefined thresholds.
    """
    if row['pointsValue'] < 500:
        if row['frequencyDays'] < 2.5:
            return "LL"  # Low Value, Low Frequency
        elif row['frequencyDays'] < 8.5:
            return "LM"  # Low Value, Medium Frequency
        elif row['frequencyDays'] < 13.5:
            return "LH"  # Low Value, High Frequency
        else:
            return "LV"  # Low Value, Very High Frequency

    elif row['pointsValue'] < 1400:
        if row['frequencyDays'] < 2.5:
            return "ML"  # Medium Value, Low Frequency
        elif row['frequencyDays'] < 8.5:
            return "MM"  # Medium Value, Medium Frequency
        elif row['frequencyDays'] < 13.5:
            return "MH"  # Medium Value, High Frequency
        else:
            return "MV"  # Medium Value, Very High Frequency

    else:
        if row['frequencyDays'] < 2.5:
            return "HL"  # High Value, Low Frequency
        elif row['frequencyDays'] < 8.5:
            return "HM"  # High Value, Medium Frequency
        elif row['frequencyDays'] < 13.5:
            return "HH"  # High Value, High Frequency
        else:
            return "HV"  # High Value, Very High Frequency

# Apply RF segmentation rules
df['rf_cluster'] = df.apply(rf_cluster, axis=1)
````
Finally we will achieve this: 

[plot clluster final]

Using the same logic as explained in the recency, we can apply a simple tree algorithm to take care of labeling.
```python
clf = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=1, max_depth=None)
clf.fit(df[['frequencyDays', 'pointsValue']], df['rf_cluster'])

# Save the model and features for future use
model_freq_value = pd.Series(
    {
        "model": clf,
        "features": ['frequencyDays', 'pointsValue']
    }
)
```

This is how our active base is distributed. 

 RF Cluster | Description                           | Number of Customers | Percentage from Active Base (%) |
|------------|---------------------------------------|---------------------|-------------------------|
| LL         | Low Value, Low Frequency              | 218                 | 37.71%                  |
| LM         | Low Value, Medium Frequency           | 112                 | 19.47%                  |
| MH         | Medium Value, High Frequency          | 34                  | 5.92%                   |
| MM         | Medium Value, Medium Frequency        | 18                  | 3.14%                   |
| HV         | High Value, Very High Frequency       | 14                  | 2.43%                   |
| HH         | High Value, High Frequency            | 8                   | 1.39%                   |
| LH         | Low Value, High Frequency             | 5                   | 0.87%                   |
| MV         | Medium Value, Very High Frequency     | 2                   | 0.35%                   |
| HL         | High Value, Low Frequency             | 1                   | 0.17%                   |
| ML         | Medium Value, Low Frequency           | 1                   | 0.17%                   |


We see here that the largest half our (56%) is composed by low value and low to medium frequency users. 

## 📈 Insights and Conclusions
With all this information in hands, in customer_profile.py we will wrap up all three information in a decision-making table:
by uniting the probabilities from our churn model the LifeCycle segmentation and the Frequency Value segmentation we achieve this:

| dtRef       | idCustomer                             | Life Cycle Segmentation | Frequency Value Segmentation | Churn Probability | dtUpdate                    |
|-------------|----------------------------------------|-------------------------|------------------------------|------------|-----------------------------|
| 2024-06-07  | 000ff655-fa9f-4baa-a108-47f581ec52a1  | Cold Active             | Low Value, Low Frequency     | 0.516375   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 0019bb9e-26d4-4ebf-8727-fc911ea28a92  | Super Active User       | Low Value, Low Frequency     | 0.058285   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 00a6d4c8-38e5-4e92-981e-4e4ac5084546  | Super Active User       | Medium Value, High Frequency | 0.010680   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 02cb6641-a115-4c53-a6ab-22b332d8e5e2  | Unengaged               | Low Value, Low Frequency     | 0.794619   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 0368a5c3-bc5d-45f2-b8a0-07e74ea1574f  | Active User             | Low Value, Low Frequency     | 0.274932   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 038d7c90-b7c3-46d2-b7ea-3db6fc63d625  | New User                | Low Value, Low Frequency     | 0.457405   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 043c4972-df0d-4181-8b77-dd6c351fdebc  | Active User             | Low Value, Medium Frequency  | 0.028294   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 044e95f2-8756-42bd-a0a3-d50708beb8da  | Active User             | Low Value, Medium Frequency  | 0.344889   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 04612efc-dc61-4434-90a1-afc1b43767c1  | Super Active User       | Low Value, High Frequency    | 0.074835   | 2024-11-28 19:30:56.922235  |
| 2024-06-07  | 05bc8281-1c44-4682-a4a0-4726e30340e4  | New User                | Low Value, Low Frequency     | 0.562112   | 2024-11-28 19:30:56.922235  |

With is a compreensive table the informs for the date of reference the Churn Probability for our churn ML model, the Life Cycle segmentation and The Frequency value segmentaion for all user in our active database.
This is interising because we now have a full report for decision-making. For instance, we can account for different strategies when abording this customers aiming for retantion, one can relly on statregy that increse the frequency and value for unengaged, and cold active, while others can think of strategies for new users. We see a full logic here, as it appears that our super active user are the least probable of churning, as see above, while unengaged are fairly high. 

[superactive vs unengaged proba plot]

## 📜 Project Structure

This is how our project directories are organized:
```
├── README.md
├── data/
│   ├── database.db
│   └── feature_store.db
├── models/
│   ├── cluster_fv_01.pkl
│   ├── cluster_rf_01.pkl
│   └── rf_01.pkl
├── src/
│   ├── feature_store/
│   │   ├── exec.sh
│   │   ├── execute.py
│   │   ├── fs_general.sql
│   │   ├── fs_hour.sql
│   │   ├── fs_points.sql
│   │   ├── fs_products.sql
│   │   └── fs_transactions.sql
│   ├── predict/
│   │   ├── customer_profile.py
│   │   ├── etl.sql
│   │   └── predict_mlflow.py
│   ├── rfv/
│   │   ├── rfv_frequency.py
│   │   └── rfv_recency.py
│   ├── train/
│   │   ├── plots/
│   │   │   ├── cumulative_gain_curve.png
│   │   │   ├── ks_statistic.png
│   │   │   ├── lift_curve.png
│   │   │   ├── precision_recall_curve.png
│   │   │   └── roc_curve.png
│   │   ├── abt.sql
│   │   ├── best_model.py
│   │   ├── eda.py
│   │   ├── train.py
│   │   └── train_mlflow.py
│   ├── pipeline.sh
├── project_structure.py
├── requirements.txt

```

## 🚧 Project Next Steps 

As a next step i am to integrate this results in a GO.APP so we will have the complete cicle of our porject, that goes from feature engenieering trought modeling and finally an go app. I am also aim to test some feature managing, removing some variables that are not so contributive to the random forest, and try do adjust again, but overall i think we have a solid product, that can be helpful for twitch user retention strategies
