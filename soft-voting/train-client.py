import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score, precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import dump
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample


def load_data(input_csv):
    df = pd.read_csv(input_csv)
    X = df.drop(columns=["Label"]).select_dtypes(include=["number"])
    y = df["Label"]
    return X, y


def evaluate(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    SN = tp / (tp + fn) if (tp + fn) else 0
    SP = tn / (tn + fp) if (tn + fp) else 0
    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_proba)
    Pre = precision_score(y_true, y_pred)

    return {
        "SN": SN,
        "SP": SP,
        "ACC": ACC,
        "MCC": MCC,
        "F1": F1,
        "AUC": AUC,
        "Pre": Pre
    }


def main(args):
    X, y = load_data(args.input_csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42)
    print(f"\nDataset division: Number of training sets ={len(y_train)}, Number of independent test sets ={len(y_test)}")

    os.makedirs(args.output_dir, exist_ok=True)
    independent_test_path = os.path.join(args.output_dir, "independent_test.csv")
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(independent_test_path, index=False)
    print(f"\nIndependent test sets saved to: {independent_test_path}")

    # Calculate the ratio of negative and positive samples for XGBoost scale_pos_weight
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    neg_pos_ratio = neg / pos
    print(f"\nThe number of negative samples in the training set = {neg}, the number of positive samples = {pos}, and the negative-positive sample ratio ={neg_pos_ratio:.4f}")

    # Calculate the ratio of negative to positive samples (for log)
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    neg_pos_ratio = neg / pos
    print(f"\nThe number of negative samples in the training set = {neg}, the number of positive samples = {pos}, and the negative-positive sample ratio ={neg_pos_ratio:.4f}")


    # Downsample the positive samples to the number of negative samples (neg)
    X_pos = X_train[y_train == 1]
    y_pos = y_train[y_train == 1]

    X_neg = X_train[y_train == 0]
    y_neg = y_train[y_train == 0]

    X_pos_down, y_pos_down = resample(
        X_pos, y_pos,
        replace=False,
        n_samples=len(y_neg), # Downsample to the same number of negative samples
        random_state=42
    )

    X_train_resampled = pd.concat([X_neg, X_pos_down], axis=0)
    y_train_resampled = pd.concat([y_neg, y_pos_down], axis=0)


    shuffle_idx = np.random.permutation(len(y_train_resampled))
    X_train_resampled = X_train_resampled.iloc[shuffle_idx].reset_index(drop=True)
    y_train_resampled = y_train_resampled.iloc[shuffle_idx].reset_index(drop=True)

    print(f"下Number of training set samples after sampling:{len(y_train_resampled)} (\u6b63Number of samples ={sum(y_train_resampled==1)}, \u8d1fNumber of samples ={sum(y_train_resampled==0)})")


    # Define a model with class weights
    adaboost = AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                        use_label_encoder=False, eval_metric='logloss',
                        random_state=42, verbosity=0,
                        scale_pos_weight=neg_pos_ratio)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                n_jobs=-1, min_samples_split=2, class_weight='balanced')
    gbdt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=5, random_state=42)
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                          random_state=42, verbose=-1, class_weight='balanced')

    classifiers = [
        ('adaboost', adaboost),
        ('xgb', xgb),
        ('rf', rf),
        ('gbdt', gbdt),
        ('lgbm', lgbm),
    ]

    soft_voting = VotingClassifier(estimators=classifiers, voting='soft')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_results = []
    all_val_preds = []  

    for train_index, val_index in skf.split(X_train_resampled, y_train_resampled):
        print(f"=== {fold}-fold cross validation ===")
        X_train_fold, X_val_fold = X_train_resampled.iloc[train_index], X_train_resampled.iloc[val_index]
        y_train_fold, y_val_fold = y_train_resampled.iloc[train_index], y_train_resampled.iloc[val_index]

        print(f"{fold}: number of training sets={len(y_train_fold)},Number of validation sets ={len(y_val_fold)}")

        soft_voting.fit(X_train_fold, y_train_fold)

        val_pred = soft_voting.predict(X_val_fold)
        val_proba = soft_voting.predict_proba(X_val_fold)[:, 1]
        val_result = evaluate(y_val_fold, val_pred, val_proba)

        print(f"Validation set results:")
        for k, v in val_result.items():
            print(f"{k}: {v:.4f}")
        val_result["Fold"] = fold

        all_results.append(val_result)

        val_pred_df = pd.DataFrame({
             "Fold": fold,
             "True Label": y_val_fold.values,
             "Predicted Label": val_pred,
             "Predicted Probability": val_proba
        })
        all_val_preds.append(val_pred_df)

       
        fold += 1


    val_preds_df = pd.concat(all_val_preds, axis=0)
    val_preds_save_path = os.path.join(args.output_dir, "cross_validation_all_folds_predictions.csv")
    val_preds_df.to_csv(val_preds_save_path, index=False)
    print(f"\nAll fold validation set prediction results have been saved to: {val_preds_save_path}")

    all_results_df = pd.DataFrame(all_results)
    all_results_path = os.path.join(args.output_dir, "cross_validation_all_folds.csv")
    all_results_df.to_csv(all_results_path, index=False)
    print(f"\nAll cross validation fold results have been saved to: {all_results_path}")
    avg_result = {k: np.mean([result[k] for result in all_results]) for k in all_results[0].keys()}

    print("\n===Five-fold cross validation average results ===")
    for k, v in avg_result.items():
        print(f"{k}: {v:.4f}")

    avg_result_df = pd.DataFrame([avg_result])
    avg_result_path = os.path.join(args.output_dir, "cross_validation_average_result.csv")
    avg_result_df.to_csv(avg_result_path, index=False)
    print(f"\nThe average cross validation results have been saved to: {avg_result_path}")

    print("\n=== Training the final model ===")

    soft_voting.fit(X_train_resampled, y_train_resampled)

    test_pred = soft_voting.predict(X_test)
    test_proba = soft_voting.predict_proba(X_test)[:, 1]
    test_result = evaluate(y_test, test_pred, test_proba)
    print("\nIndependent test set results：")
    for k, v in test_result.items():
        print(f"{k}: {v:.4f}")

    model_path = os.path.join(args.output_dir, "voting_model-five.joblib")
    dump(soft_voting, model_path)
    print(f"\nModel saved to:{model_path}")

    test_results_df = pd.DataFrame({
        "True Label": y_test,
        "Predicted Label": test_pred,
        "Predicted Probability": test_proba
    })
    test_results_path = os.path.join(args.output_dir, "test_results.csv")
    test_results_df.to_csv(test_results_path, index=False)
    print(f"\nThe independent test set prediction results have been saved to: {test_results_path}")

    test_result_df = pd.DataFrame([test_result])
    test_result_path = os.path.join(args.output_dir, "independent_test_result.csv")
    test_result_df.to_csv(test_result_path, index=False)
    print(f"\nIndependent test set evaluation metrics have been saved to: {test_result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Integrated features CSV with labels")
    parser.add_argument("--test_csv", type=str, required=True, help="Independent test set feature CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for saving data and models")
    args = parser.parse_args()
    main(args)


