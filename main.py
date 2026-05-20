import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =========================
# 2. Standardize column names
# =========================
df.columns = df.columns.str.strip().str.lower()

# =========================
# 3. Fix data inconsistencies
# =========================
df["totalcharges"] = df["totalcharges"].replace(" ", pd.NA)
df["totalcharges"] = pd.to_numeric(df["totalcharges"])
df = df.dropna()

# =========================
# 4. Explore variable types
# =========================
target = "churn"

categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if target in categorical_cols:
    categorical_cols.remove(target)

print("Dataset shape:", df.shape)

print("\nCategorical variables:")
print(categorical_cols)

print("\nNumerical variables:")
print(numerical_cols)

print("\nTarget variable:")
print(target)

# =========================
# 5. Summary statistics
# =========================
print("\nSummary statistics:")
print(df[["tenure", "monthlycharges", "totalcharges"]].describe())

# =========================
# 6. Contract summary
# =========================
contract_summary = pd.DataFrame({
    "count": df["contract"].value_counts(),
    "percentage": (df["contract"].value_counts(normalize=True) * 100).round(2)
})

print("\nContract summary:")
print(contract_summary)

# =========================
# 7. Prepare data for modeling
# =========================
df = df.drop("customerid", axis=1)

# Binary columns
binary_map = {
    "yes": 1,
    "no": 0,
    "male": 1,
    "female": 0
}

binary_cols = [
    "gender",
    "partner",
    "dependents",
    "phoneservice",
    "paperlessbilling",
    "churn"
]

for col in binary_cols:
    df[col] = df[col].str.lower().map(binary_map)

# Multi-category columns
multi_cols = [
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paymentmethod"
]

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Convert boolean dummy columns to integers
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# =========================
# 8. Final check
# =========================
print("\nFinal dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())



#Assiment 2-----------------------------------------------

# Set style
sns.set_style("whitegrid")

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Churn Rates Across Customer Demographics', fontsize=16, fontweight='bold')

# 1. GENDER
gender_churn = df.groupby('gender')['churn'].agg(['sum', 'count'])
gender_churn['churn_rate'] = (gender_churn['sum'] / gender_churn['count'] * 100).round(2)
churn_counts = df.groupby('gender')['churn'].value_counts().unstack()
churn_counts.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'], width=0.6)
axes[0, 0].set_title('Churn by Gender', fontweight='bold')
axes[0, 0].legend(['No Churn', 'Churned'])

# 2. SENIOR CITIZEN
churn_senior = df.groupby('seniorcitizen')['churn'].value_counts().unstack()
churn_senior.index = ['Non-Senior', 'Senior']
churn_senior.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'], width=0.6)
axes[0, 1].set_title('Churn by Senior Citizen Status', fontweight='bold')
axes[0, 1].legend(['No Churn', 'Churned'])

# 3. PARTNER
churn_partner = df.groupby('partner')['churn'].value_counts().unstack()
churn_partner.index = ['No Partner', 'Has Partner']
churn_partner.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'], width=0.6)
axes[1, 0].set_title('Churn by Partner Status', fontweight='bold')
axes[1, 0].legend(['No Churn', 'Churned'])

# 4. DEPENDENTS
churn_dependents = df.groupby('dependents')['churn'].value_counts().unstack()
churn_dependents.index = ['No Dependents', 'Has Dependents']
churn_dependents.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'], width=0.6)
axes[1, 1].set_title('Churn by Dependent Status', fontweight='bold')
axes[1, 1].legend(['No Churn', 'Churned'])



plt.tight_layout()
plt.show()

# Print churn rates
print("\nChurn Rates Summary:")
print(f"By Gender: {(gender_churn['churn_rate']).to_dict()}")
print(f"Senior Citizen: {(df[df['seniorcitizen']==1]['churn'].mean() * 100):.2f}%")
print(f"Has Partner: {(df[df['partner']==1]['churn'].mean() * 100):.2f}%")
print(f"Has Dependents: {(df[df['dependents']==1]['churn'].mean() * 100):.2f}%")


print("-----------churn in relation to services----------------")


df_service = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Standardize column names
df_service.columns = df_service.columns.str.strip().str.lower()

# Clean totalcharges
df_service["totalcharges"] = df_service["totalcharges"].replace(" ", pd.NA)
df_service["totalcharges"] = pd.to_numeric(df_service["totalcharges"])
df_service = df_service.dropna()


# Analyze churn by service columns
service_cols = ["internetservice", "techsupport", "onlinesecurity"]

for col in service_cols:
    print(f"\nChurn by {col}:")
    print(
        pd.crosstab(
            df_service[col],
            df_service["churn"],
            normalize="index"
        ).round(3) * 100
    )

for col in service_cols:
    churn_rate = (
        pd.crosstab(df_service[col], df_service["churn"], normalize="index")["Yes"] * 100
    )

    churn_rate.plot(kind="bar")
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel(col)
    plt.xticks(rotation=0)
    plt.show()


#Bar chart — churn by contract type
contract_churn = (
    pd.crosstab(df_service["contract"], df_service["churn"], normalize="index")["Yes"] * 100
)

contract_churn.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Contract Type")
plt.xticks(rotation=0)
plt.show()


#Boxplot — churn vs tenure

df_service.boxplot(column="tenure", by="churn")

plt.title("Tenure by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure")
plt.show()

#Boxplot — churn vs tenure
df_service.boxplot(column="tenure", by="churn")

plt.title("Tenure by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Tenure")
plt.show()

#Boxplot — churn vs monthly charges

df_service.boxplot(column="monthlycharges", by="churn")

plt.title("Monthly Charges by Churn")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.show()


#1. Customers on month-to-month contracts churn much more.
#2. Customers who churn usually have lower tenure.
#3. Customers who churn tend to pay higher monthly charges.




# Assignment 3 Churn Prediction Modeling

X = df.drop("churn", axis=1)
y = df["churn"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=3000)

model.fit(X_train_scaled, y_train)

print("Model training completed.")


# Create the model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

print("Random Forest model training completed.")


# Create model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)

# Train model
xgb_model.fit(X_train, y_train)

print("XGBoost model training completed.")

#----------------------------------------------------
# Predictions
y_pred_log = model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Create comparison table
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_log),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_xgb)
    ],
    "Precision": [
        precision_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_xgb)
    ],
    "Recall": [
        recall_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_xgb)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_xgb)
    ]
})

results = results.round(4)
print(results)

print("Feature importance from Random Forest------------------------------")

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.head(10)

#visualization
top_features = feature_importance.head(10)

top_features.sort_values("Importance").plot(
    kind="barh",
    x="Feature",
    y="Importance",
    legend=False
)

plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("")
plt.show()


## Confusion Matrix
import matplotlib.pyplot as plt

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)

# Visualization
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix - XGBoost")
plt.show()


## ROC Curve

# Predict probabilities
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

# ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)

# AUC score
auc_score = roc_auc_score(y_test, y_prob_xgb)

# Plot ROC curve
plt.figure(figsize=(6, 5))

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve - XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.show()


#Assignment 4


if df["churn"].dtype == "object":
    churned_customers = df[df["churn"].str.lower() == "yes"]
else:
    churned_customers = df[df["churn"] == 1]

# Calculate average monthly revenue loss
avg_monthly_loss = churned_customers["monthlycharges"].mean()

# Calculate average total revenue loss
avg_total_loss = churned_customers["totalcharges"].mean()

# Number of churned customers
num_churned = churned_customers.shape[0]

# Estimated total monthly churn loss
total_monthly_loss = avg_monthly_loss * num_churned

# Display results
print("Churn Cost Simulation")
print("-" * 40)

print(f"Number of churned customers: {num_churned}")

print(f"\nAverage monthly revenue loss per churned customer: "
      f"${avg_monthly_loss:.2f}")

print(f"Average total revenue loss per churned customer: "
      f"${avg_total_loss:.2f}")

print(f"\nEstimated total monthly revenue loss from churn: "
      f"${total_monthly_loss:,.2f}")



# Select churned customers

if (df["churn"].dtype == "object") :
    churned_customers = df[df["churn"].str.lower() == "yes"]
else:
    churned_customers = df[df["churn"] == 1]

# Basic metrics
num_churned = len(churned_customers)

avg_monthly_revenue = churned_customers["monthlycharges"].mean()

print("Average monthly revenue per churned customer:",round(avg_monthly_revenue, 2))

print("Total churned customers:", num_churned)
## Strategy 1: Discount Offer

#Assumptions:
#- 20% discount offered
#- 30% of churned customers are retained

# Strategy assumptions
discount_rate = 0.20
retention_rate_discount = 0.30

# Customers retained
retained_discount = int(num_churned * retention_rate_discount)

# Revenue saved
saved_revenue_discount = retained_discount * avg_monthly_revenue

# Cost of discounts
discount_cost = saved_revenue_discount * discount_rate

# Net gain
net_gain_discount = saved_revenue_discount - discount_cost

print("DISCOUNT STRATEGY")
print("-" * 30)

print("Customers retained:", retained_discount)

print(f"Revenue saved: ${saved_revenue_discount:,.2f}")

print(f"Discount cost: ${discount_cost:,.2f}")

print(f"Net gain: ${net_gain_discount:,.2f}")



## Strategy 2: Loyalty Perks

#Assumptions:
#- Loyalty perks cost $10 per customer
#- 25% of churned customers are retained

# Strategy assumptions
retention_rate_loyalty = 0.25
perk_cost_per_customer = 10

# Customers retained
retained_loyalty = int(num_churned * retention_rate_loyalty)

# Revenue saved
saved_revenue_loyalty = retained_loyalty * avg_monthly_revenue

# Loyalty program cost
loyalty_cost = retained_loyalty * perk_cost_per_customer

# Net gain
net_gain_loyalty = saved_revenue_loyalty - loyalty_cost

print("LOYALTY PERKS STRATEGY")
print("-" * 30)

print("Customers retained:", retained_loyalty)

print(f"Revenue saved: ${saved_revenue_loyalty:,.2f}")

print(f"Loyalty cost: ${loyalty_cost:,.2f}")

print(f"Net gain: ${net_gain_loyalty:,.2f}")
#-----------------------------------

# Copy dataframe
seg_df = df.copy()

# Scale the full feature dataset
X_scaled = scaler.transform(X)

# Predict churn probabilities
seg_df["churn_probability"] = model.predict_proba(X_scaled)[:, 1]

# -------------------------------
# Customer Value Segmentation
# -------------------------------

avg_value = seg_df["monthlycharges"].mean()

seg_df["value_segment"] = np.where(
    seg_df["monthlycharges"] >= avg_value,
    "High Value",
    "Low Value"
)

# -------------------------------
# Churn Risk Segmentation
# -------------------------------

seg_df["risk_segment"] = np.where(
    seg_df["churn_probability"] >= 0.5,
    "High Risk",
    "Low Risk"
)

# -------------------------------
# Combine Segments
# -------------------------------

seg_df["customer_segment"] = (
    seg_df["risk_segment"] + " + " + seg_df["value_segment"]
)

# Count customers in each segment
segment_counts = seg_df["customer_segment"].value_counts()

# Show results
print(segment_counts)


plt.figure(figsize=(8,5))

segment_counts.plot(kind="bar")

plt.title("Customer Segmentation")
plt.xlabel("Customer Segment")
plt.ylabel("Number of Customers")

plt.xticks(rotation=10)

plt.show()