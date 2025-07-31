import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==== Path Settings ====
INPUT_FEATURE_CSV = "./single-protein.csv"
OUTPUT_PREDICTION_CSV = "./single-protein_predictions.csv"
ENCODER_PATH = "./onehot_encoder.pkl"
MEAN_STD_PATH = "./mean_std.npz"
FEATURE_COLUMNS_PATH = "./feature_columns.txt"
MODEL_PATH = "./final_model.pth"
ACCURACY_REPORT_PATH = "./results/accuracy_report1.txt"


PREDICTION_THRESHOLD = 0.5

TRUE_LABEL_COLUMN = "is_idr"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Model Definition ===
class MLPWithMultiheadAttentionFlexible(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, mlp_units=512, dropout_rate=0.6):
        super().__init__()
        self.embed_dim = embed_dim
        self.new_dim = ((input_dim + embed_dim - 1) // embed_dim) * embed_dim
        self.seq_len = self.new_dim // embed_dim

        self.input_proj = nn.Linear(input_dim, self.new_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.new_dim, mlp_units),
            nn.BatchNorm1d(mlp_units),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units, mlp_units // 2),
            nn.BatchNorm1d(mlp_units // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units // 2, mlp_units // 4),
            nn.BatchNorm1d(mlp_units // 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units // 4, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x.view(batch_size, self.seq_len, self.embed_dim)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.reshape(batch_size, -1)
        return self.mlp(attn_output).squeeze(1)


# === Feature preprocessing ===
def preprocess_features(df, encoder, mean, std, expected_columns):
    # 1. Explicitly specify feature types
    cat_features = ['amino_acid', 'dssp_sec']
    protT5_cols = sorted([c for c in df.columns if c.startswith('protT5_')])
    numeric_features = protT5_cols + ['hydrophobicity', 'polarity', 'asa', 'phi', 'psi', 'flexibility']
    
    print(f"Original data dimensions: {df.shape}")
    print(f"Category characteristics: {len(cat_features)}个, Numerical features: {len(numeric_features)}个")
    print(f"Original data column names:{df.columns.tolist()}")

    # 2. OneHot Encoding - Uses the column order from training
    cat_data = df[cat_features].fillna('-')
    cat_enc = encoder.transform(cat_data)
    
    # Directly use the category part of the feature column name saved during training
    cat_columns = [col for col in expected_columns 
                  if col.startswith('amino_acid_') or col.startswith('dssp_sec_')]
    
    cat_df = pd.DataFrame(
        cat_enc, 
        columns=encoder.get_feature_names_out(cat_features),
        index=df.index
    )
    cat_df = cat_df[cat_columns]  # Only keep the columns that existed during training
    
    print(f"OneHot encoded dimensions:{cat_df.shape}")

    # 3. Numerical normalization
    num_columns = [col for col in expected_columns if col not in cat_columns]
    num_data = df[numeric_features].copy()
    
    # Handling missing columns
    missing_num = set(num_columns) - set(num_data.columns)
    for col in missing_num:
        num_data[col] = 0.0
    
    # Only select columns that actually exist for normalization
    existing_cols = list(set(num_data.columns) & set(num_columns))
    num_std = pd.DataFrame(index=df.index)
    for col in num_columns:
        if col in existing_cols:
            num_std[col] = (num_data[col] - mean[col]) / std[col]
        else:
            num_std[col] = 0.0
    
    print(f"Dimension after numerical normalization: {num_std.shape}")

    # 4. Merge features - in training order
    all_features = pd.concat([cat_df, num_std], axis=1)[expected_columns]
    print(f"Merged dimension: {all_features.shape}")
    
    # Check if there are NaN values
    if all_features.isnull().values.any():
        print("Warning: Eigen matrix contains NaN values!")
        all_features.fillna(0, inplace=True)
    
    return all_features.values


def calculate_accuracy(true_labels, pred_labels):

    if true_labels is None or pred_labels is None:
        return None, None, None

    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]

    accuracy = accuracy_score(true_labels, pred_labels)

    cm = confusion_matrix(true_labels, pred_labels)

    report = classification_report(true_labels, pred_labels, target_names=["非IDR", "IDR"])
    
    return accuracy, cm, report

def predict():

    df = pd.read_csv(INPUT_FEATURE_CSV)
    print(f"Reading Data: {df.shape}")
    print(f"Enter data column name: {df.columns.tolist()}")
    

    true_labels = None
    if TRUE_LABEL_COLUMN in df.columns:
        true_labels = df[TRUE_LABEL_COLUMN].values
        print(f"Find the true label column '{TRUE_LABEL_COLUMN}', the accuracy will be calculated")
    else:
        print(f"No true label column found '{TRUE_LABEL_COLUMN}', the accuracy cannot be calculated")


    encoder = joblib.load(ENCODER_PATH)
    mean_std = np.load(MEAN_STD_PATH)
    
    with open(FEATURE_COLUMNS_PATH) as f:
        expected_columns = [line.strip() for line in f]
    
    print(f"Expected feature dimensions: {len(expected_columns)}")
    
    mean = pd.Series(mean_std['mean'], index=expected_columns)
    std = pd.Series(mean_std['std'], index=expected_columns)

    features = preprocess_features(df, encoder, mean, std, expected_columns)
    print(f"Feature dimensions after processing: {features.shape}")

    input_dim = features.shape[1]
    print(f"Model input dimension: {input_dim}")
    model = MLPWithMultiheadAttentionFlexible(input_dim=input_dim, embed_dim=64, num_heads=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        
        # Make sure the logits have the correct shape
        if len(logits.shape) == 0:
            # Single sample case
            probs = torch.sigmoid(logits).unsqueeze(0).cpu().numpy()
        else:
            probs = torch.sigmoid(logits).cpu().numpy()
    
    print(f"Prediction probability dimension: {probs.shape}")
    print(f"Top 5 predictions: {probs[:5]}")

    pred_labels = (probs >= PREDICTION_THRESHOLD).astype(int)
    print(f"Predicted label distribution: {np.unique(pred_labels, return_counts=True)}")

    if len(probs) == len(df):
        df['IDR_score'] = probs
        df['IDR_pred'] = pred_labels
        print(f"Successfully added prediction scores and labels to the data frame")
    else:
        print(f"The prediction result dimension ({len(probs)}) does not match the data frame length ({len(df)})!")
        # Attempted fix: Truncate or pad
        min_len = min(len(probs), len(df))
        df = df.iloc[:min_len]
        df['IDR_score'] = probs[:min_len]
        df['IDR_pred'] = pred_labels[:min_len]

    accuracy = None
    confusion_mat = None
    report = None
    
    if true_labels is not None:
        accuracy, confusion_mat, report = calculate_accuracy(true_labels, pred_labels)
        
        if accuracy is not None:
            print(f"\n{'='*50}")
            print(f"Model performance evaluation:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Confusion Matrix:\n{confusion_mat}")
            print(f"Classification Report:\n{report}")
            print(f"{'='*50}")

            with open(ACCURACY_REPORT_PATH, "w") as f:
                f.write(f"protein: {os.path.basename(INPUT_FEATURE_CSV)}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Confusion Matrix:\n")
                f.write(np.array2string(confusion_mat))
                f.write("\n\nClassification Report:\n")
                f.write(report)
            print(f"Accuracy report saved to: {ACCURACY_REPORT_PATH}")

    df.to_csv(OUTPUT_PREDICTION_CSV, index=False)
    print(f"The prediction is completed and the results are saved in：{OUTPUT_PREDICTION_CSV}")

    output_df = pd.read_csv(OUTPUT_PREDICTION_CSV)
    print(f"Output file column names: {output_df.columns.tolist()}")

    print(f"\nOutput file preview:")
    if 'IDR_score' in output_df.columns and 'IDR_pred' in output_df.columns:
        preview_cols = []
        for col in ['position', 'residue', 'aa', 'amino_acid', 'resid']:
            if col in output_df.columns:
                preview_cols.append(col)
        
        if preview_cols:
            preview_cols.extend(['IDR_score', 'IDR_pred'])
            if TRUE_LABEL_COLUMN in output_df.columns:
                preview_cols.append(TRUE_LABEL_COLUMN)
            print(output_df[preview_cols].head())
        else:
            preview_cols = ['IDR_score', 'IDR_pred']
            if TRUE_LABEL_COLUMN in output_df.columns:
                preview_cols.append(TRUE_LABEL_COLUMN)
            print(output_df[preview_cols].head())
    else:
        print(output_df.head())


if __name__ == "__main__":
    print(f"Use prediction threshold: {PREDICTION_THRESHOLD}")
    print(f"True label column name: '{TRUE_LABEL_COLUMN}'")
    predict()