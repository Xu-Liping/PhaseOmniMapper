import os
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
import joblib

# ==== Setting random seed and device ====
seed = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== parameter====
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-5
PATIENCE = 5
WEIGHT_DECAY = 1e-3

# ==== Dataset ====
class ResidueDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==== MLP with MultiheadAttention ====
class MLPWithMultiheadAttentionFlexible(nn.Module):
    def __init__(self, input_dim, embed_dim=64, num_heads=4, mlp_units=512, dropout_rate=0.6):
        super().__init__()
        self.embed_dim = embed_dim
        self.new_dim = ((input_dim + embed_dim - 1) // embed_dim) * embed_dim
        self.seq_len = self.new_dim // embed_dim
        print(f"Original input dimensions: {input_dim}")
        print(f"Dimensions after mapping: {self.new_dim} (seq_len={self.seq_len}, embed_dim={self.embed_dim})")

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

# ==== Preprocessing function  ====
def preprocess_features(train_df, val_df, test_df):
    cat_features = ['amino_acid', 'dssp_sec']
    protT5_cols = [c for c in train_df.columns if c.startswith('protT5_')]
    numeric_cols = protT5_cols + ['hydrophobicity', 'polarity', 'asa', 'phi', 'psi', 'flexibility']

    train_cat = train_df[cat_features].fillna('-')
    val_cat = val_df[cat_features].fillna('-')
    test_cat = test_df[cat_features].fillna('-')

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(train_cat)

    train_cat_enc = encoder.transform(train_cat)
    val_cat_enc = encoder.transform(val_cat)
    test_cat_enc = encoder.transform(test_cat)

    train_cat_df = pd.DataFrame(train_cat_enc, columns=encoder.get_feature_names_out(cat_features), index=train_df.index)
    val_cat_df = pd.DataFrame(val_cat_enc, columns=encoder.get_feature_names_out(cat_features), index=val_df.index)
    test_cat_df = pd.DataFrame(test_cat_enc, columns=encoder.get_feature_names_out(cat_features), index=test_df.index)

    train_cat_df = train_cat_df.loc[:, sorted(train_cat_df.columns)]
    val_cat_df = val_cat_df.loc[:, sorted(val_cat_df.columns)]
    test_cat_df = test_cat_df.loc[:, sorted(test_cat_df.columns)]

    train_numeric = train_df[numeric_cols]
    val_numeric = val_df[numeric_cols]
    test_numeric = test_df[numeric_cols]

    train_features = pd.concat([train_cat_df, train_numeric], axis=1)
    val_features = pd.concat([val_cat_df, val_numeric], axis=1)
    test_features = pd.concat([test_cat_df, test_numeric], axis=1)

    mean = train_features.mean()
    std = train_features.std() + 1e-6

    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    test_features = (test_features - mean) / std


    for df in [val_features, test_features]:
        missing_cols = set(train_features.columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[train_features.columns]

    return train_features, val_features, test_features, encoder

# ==== Evaluation Function ====
def evaluate(model, dataloader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            probs = torch.sigmoid(model(X))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    preds = np.array(all_probs) > 0.5
    return {
        'accuracy': accuracy_score(all_labels, preds),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'f1': f1_score(all_labels, preds),
        'precision': precision_score(all_labels, preds),
        'recall': recall_score(all_labels, preds)
    }

# ==== Training function ====
def train_model(model, train_loader, val_loader, criterion, optimizer):
    best_auc = 0.0
    patience_counter = 0
    best_model_path = "./models/best_model.pth"
    history = []

    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader)
        history.append({"epoch": epoch + 1, **val_metrics})

        print(f"Epoch {epoch+1:02d}: " + ", ".join([f"{k}={v:.4f}" for k,v in val_metrics.items()]))

        if val_metrics['roc_auc'] > best_auc:
            best_auc = val_metrics['roc_auc']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    pd.DataFrame(history).to_csv("./models/val_metrics.csv", index=False)
    return best_model_path


if __name__ == "__main__":
    data = pd.read_csv("./idr_residues_features.csv")
    os.makedirs("./models", exist_ok=True)

    protein_ids = data['protein_id'].unique()
    test_proteins = np.random.choice(protein_ids, size=int(len(protein_ids)*0.1), replace=False)
    test_df = data[data['protein_id'].isin(test_proteins)].reset_index(drop=True)

    test_pos = test_df[test_df['is_idr'] == 1]
    test_neg = test_df[test_df['is_idr'] == 0]
    test_neg_downsampled = resample(test_neg, replace=False, n_samples=int(len(test_pos)/2), random_state=seed)

    test_df = pd.concat([test_pos, test_neg_downsampled]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print("Number of independent test set samples:", len(test_df))
    print("Number of positive samples in the independent test set (is_idr=1)：", (test_df['is_idr'] == 1).sum())
    print("Number of negative samples in the independent test set (is_idr=0)：", (test_df['is_idr'] == 0).sum())

    train_df = data[~data['protein_id'].isin(test_proteins)].reset_index(drop=True)
    train_pos = train_df[train_df['is_idr'] == 1]
    train_neg = train_df[train_df['is_idr'] == 0].sample(n=int(len(train_pos)*0.5), random_state=seed)

    balanced_train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df, val_df = train_test_split(
        balanced_train_df, test_size=0.2, stratify=balanced_train_df['is_idr'], random_state=seed
    )

    y_train = train_df['is_idr']
    y_val = val_df['is_idr']
    y_test = test_df['is_idr']

    X_train, X_val, X_test, encoder = preprocess_features(train_df, val_df, test_df)

    feature_columns_path = "./models/feature_columns.txt"
    with open(feature_columns_path, "w") as f:
         for col in X_train.columns:
            f.write(col + "\n")

    print(f"Feature column names saved to {feature_columns_path}")

    # Save encoder and standardization parameters mean/std
    joblib.dump(encoder, "./models/onehot_encoder.pkl")
    mean = X_train.mean()
    std = X_train.std() + 1e-6
    np.savez("./models/mean_std.npz", mean=mean.values, std=std.values)


    train_loader = DataLoader(ResidueDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ResidueDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(ResidueDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = MLPWithMultiheadAttentionFlexible(input_dim=X_train.shape[1], embed_dim=64, num_heads=4).to(device)

    pos_weight = torch.tensor([len(train_neg)/len(train_pos)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_model_path = train_model(model, train_loader, val_loader, criterion, optimizer)

    print("\nLoad the best model and evaluate it on the test set:")
    model.load_state_dict(torch.load(best_model_path))
    metrics = evaluate(model, test_loader)
    pd.DataFrame([metrics]).to_csv("./models/test_metrics.csv", index=False)
    print("Test set evaluation metrics:")
    for k, v in metrics.items():
         print(f"{k}: {v:.4f}")

    torch.save(model.state_dict(), "./models/final_model.pth")

