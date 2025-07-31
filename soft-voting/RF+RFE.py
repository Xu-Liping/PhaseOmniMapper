import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

def check_column_difference(pos_df, neg_df):
    pos_cols = set(pos_df.columns)
    neg_cols = set(neg_df.columns)
    only_in_pos = pos_cols - neg_cols
    only_in_neg = neg_cols - pos_cols
    common_cols = pos_cols & neg_cols

    print("\n=== Listing Comparison Report ===")
    print(f"Positive sample unique column：{sorted(only_in_pos) if only_in_pos else 'no'}")
    print(f"Negative sample unique columns：{sorted(only_in_neg) if only_in_neg else 'no'}")
    print(f"Total number of columns：{len(common_cols)} columns")

def load_data(pos_csv, neg_csv):
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)
    check_column_difference(pos_df, neg_df)

    pos_df['Label'] = 1
    neg_df['Label'] = 0
    df = pd.concat([pos_df, neg_df], ignore_index=True)

    if 'Protein_ID' in df.columns:
        df = df.drop(columns=['Protein_ID'])

    X = df.drop(columns=['Label'], errors='ignore')
    X = X.select_dtypes(include=['number'])
    y = df['Label']

    nan_counts = X.isna().sum()
    if nan_counts.any():
        print("\n There are columns containing NaNs:")
        print(nan_counts[nan_counts > 0])
    else:
        print("\n No NaNs")

    print(f"Feature Dimension: {X.shape}, Label distribution: {y.value_counts().to_dict()}")
    return X, y, X.columns

def plot_feature_importances(importances, feature_names, out_file):
    sorted_idx = importances.argsort()[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_importance = importances[sorted_idx]

    plt.figure(figsize=(12, max(6, len(feature_names) // 5)))
    plt.barh(range(len(sorted_features)), sorted_importance[::-1], color="#482957")
    plt.yticks(range(len(sorted_features)), sorted_features[::-1], fontsize=10)
    plt.xlabel("Feature Importance", fontsize=12, fontweight="bold")
    plt.title("Selected Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

def plot_rfecv_curve(rfecv, out_file, max_features_to_plot=40, final_feature_num=17):
    mean_scores = rfecv.cv_results_["mean_test_score"]
    std_scores = rfecv.cv_results_["std_test_score"]
    num_features = len(mean_scores)

    end = min(max_features_to_plot, num_features)
    x_vals = np.arange(1, end + 1)
    mean_vals = mean_scores[:end]
    std_vals = std_scores[:end]
    
    fig, ax = plt.subplots(figsize=(8, 5)) 

    ax.plot(x_vals, mean_vals, color="#9C64B9", label="Mean CV Accuracy", linewidth=2)
    ax.fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals,
                    color="#D6C9E6", alpha=0.6)

    if final_feature_num <= end:
        ax.axvline(x=final_feature_num, color='#555555', linestyle='--', linewidth=1.5)
        ax.text(final_feature_num + 0.5, max(mean_vals)*0.98, f"{final_feature_num} features",
                rotation=90, color='black', fontsize=16, fontweight='bold', va='top')

    best_idx = np.argmax(mean_vals)
    ax.scatter(x_vals[best_idx], mean_vals[best_idx], color='#6A0DAD', s=40, zorder=5)
    ax.text(x_vals[best_idx] + 0.5, mean_vals[best_idx], "Best", fontsize=16, color='red')

    ax.set_title("RFECV Accuracy vs Number of Features", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.grid(False)  
    fig.tight_layout()
    fig.savefig(out_file, dpi=500)
    plt.close(fig)


def main(args):
    X, y, feature_names = load_data(args.pos_csv, args.neg_csv)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("feature_selection", RFECV(
            estimator=rf,
            step=1,
            cv=StratifiedKFold(5),
            scoring='accuracy',
            verbose=1,
            n_jobs=-1
        )),
        ("classifier", rf)
    ])

    pipeline.fit(X, y)
    rfecv = pipeline.named_steps['feature_selection']
    selected_features = feature_names[rfecv.support_]

    print(f"\n Selected number of features: {rfecv.n_features_}")
    print("Selected Features List：")
    for i, f in enumerate(selected_features):
        print(f"{i+1}. {f}")

    plot_feature_importances(
        rf.feature_importances_,
        selected_features,
        args.importance_plot
    )

    plot_rfecv_curve(rfecv, args.rfecv_plot, max_features_to_plot=40, final_feature_num=rfecv.n_features_)


    output_df = X[selected_features].copy()
    output_df['Label'] = y
    output_df.to_csv(args.output_csv, index=False)
    print(f"\n The simplified feature matrix has been saved to: {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Selection with RFECV and Random Forest")
    parser.add_argument("--pos_csv", type=str, required=True, help="client features csv")
    parser.add_argument("--neg_csv", type=str, required=True, help="non-LLPS features csv")
    parser.add_argument("--output_csv", type=str, required=True, help="Export selected features to CSV")
    parser.add_argument("--importance_plot", type=str, default="feature_importance.png", help="Feature Importance Plot")
    parser.add_argument("--rfecv_plot", type=str, default="rfecv_curve.jpg", help="RFECV curve")
    args = parser.parse_args()

    main(args)
