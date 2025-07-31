import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, \
    matthews_corrcoef
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

# ========== 1. Create a save directory ==========
os.makedirs("models", exist_ok=True)
os.makedirs("predictions", exist_ok=True)


def calculate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    sn = recall_score(y_true, y_pred)  # Sensitivity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sp = tn / (tn + fp)  # Specificity
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "ACC": acc,
        "AUC": auc,
        "F1": f1,
        "Precision": pre,
        "Sensitivity": sn,
        "Specificity": sp,
        "MCC": mcc
    }


# Read feature data
df1 = pd.read_csv('./scaffold_features.csv')
df2 = pd.read_csv('./client_features.csv')
df = pd.concat([df1, df2], ignore_index=True)

X = df.drop(columns=['protein_id', 'label'], errors='ignore')
y = df['label'].values

# Output total number of features
print(f"Original feature dimension: {X.shape[1]}")

# Feature Selection: RF Importance + RFE
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)


importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices][:20], y=X.columns[indices][:20])
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")

selector = RFE(rf, n_features_to_select=50)
selector.fit(X, y)
X_selected = X.loc[:, selector.support_]

rfe_rank = selector.ranking_
plt.figure(figsize=(12, 6))
sns.barplot(x=X.columns, y=rfe_rank)
plt.xticks(rotation=90)
plt.title("RFE Feature Ranking")
plt.tight_layout()
plt.savefig("rfe_ranking.png")

# Divide 10% independent test set
X_trainval, X_test, y_trainval, y_test = train_test_split(X_selected, y, test_size=0.1, stratify=y, random_state=42)

# 5-fold cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the model
model_list = [
    ('AdaBoost', AdaBoostClassifier(n_estimators=200, learning_rate=0.1,random_state=42)),
    ('XGBoost', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=10,  random_state=42)),
    ('GBDT', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=5, random_state=42)),
    ('LightGBM', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,random_state=42))
]

cv_all_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval, y_trainval), start=1):
    print(f"\n===== Fold {fold} =====")
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

    for name, model in model_list:
        print(f"Training model: {name}")
        model.fit(X_train, y_train)

        joblib.dump(model, f"./models/{name}_fold{fold}.pkl")

        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)[:, 1]
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_prob)
        train_metrics['Model'] = name
        train_metrics['Set'] = 'Train'
        train_metrics['Fold'] = fold

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
        val_metrics['Model'] = name
        val_metrics['Set'] = 'Validation'
        val_metrics['Fold'] = fold

        cv_all_results.extend([train_metrics, val_metrics])

df_all_cv = pd.DataFrame(cv_all_results)
df_all_cv.to_csv("all_models_cv_results.csv", index=False)



# ========== 2. Model independent testing==========
test_results = []

for name, model in model_list:
    print(f"\n Training and testing the model：{name}")

    model.fit(X_trainval, y_trainval)

    joblib.dump(model, f"./models/{name}_final.pkl")

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    pred_df = pd.DataFrame({
        "True_Label": y_test,
        "Pred_Prob": y_test_prob,
        "Pred_Label": y_test_pred
    })
    pred_df.to_csv(f"./models/{name}_test_predictions.csv", index=False)

    metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
    metrics["Model"] = name
    test_results.append(metrics)

# ========== 3. Aggregate independent test results for all models ==========
df_results = pd.DataFrame(test_results)
df_results.to_csv("individual_models_test_results.csv", index=False)
print("\nindividual_models_test_results.csv")
