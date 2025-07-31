import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import mixed_precision
from GraphFromPDB import graph_from_pdb, Dataset
from Model import GraphAttentionNetwork
import gc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib import font_manager


# Enable mixed precision training
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Setting the random seed
np.random.seed(42)
tf.random.set_seed(42)

# Reading Data
df = pd.read_csv("./data.csv")
seq_length = [len(i) for i in df.seq.values]

# Read protein embedding
U50_embeddings = np.load('./data.npy')
U50_embeddings_list = []
start_idx = 0
for length in seq_length:
    end_idx = start_idx + length
    U50_embeddings_list.append(U50_embeddings[start_idx:end_idx, :])
    start_idx = end_idx
    
# Ensure that the independent test set is strictly 10%
test_size_count = int(0.1 * df.shape[0])
train_val_indices, test_indices = train_test_split(
    np.arange(df.shape[0]),
    test_size=test_size_count,
    random_state=42,
    stratify=df.label
)
print(f"Number of independent test set samples: {len(test_indices)}，Number of training + validation set samples: {len(train_val_indices)}")

U50_train_val = [U50_embeddings_list[i] for i in train_val_indices]
U50_test = [U50_embeddings_list[i] for i in test_indices]
df_train_val = df.iloc[train_val_indices]
df_test = df.iloc[test_indices]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameters
HIDDEN_UNITS = 16
NUM_HEADS = 4
NUM_LAYERS = 1
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
DROPOUT_RATE = 0.5
L2_REG = 1e-3


# Customize callback to monitor F1 score
class F1Callback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        self.val_data = val_data
        self.best_f1 = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        y_val_true = np.array(self.val_data[1])
        y_pred = self.model.predict(self.val_data[0]).flatten()
        y_pred_binary = (y_pred > 0.5).astype(int)
        f1 = f1_score(y_val_true, y_pred_binary)
        print(f"\nEpoch {epoch + 1} F1: {f1:.4f}")
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights:
            self.model.set_weights(self.best_weights)


results = []
best_val_f1 = 0
best_model_path = "./best_model.h5"
fold_model_paths = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(df_train_val, df_train_val.label), start=1):
    print(f"\n===== Fold {fold} training starts=====")

    train_indices = df_train_val.index[train_idx]
    valid_indices = df_train_val.index[valid_idx]

    U50_embeddings_train = [U50_train_val[i] for i in train_idx]
    U50_embeddings_valid = [U50_train_val[i] for i in valid_idx]

    x_train = graph_from_pdb(df.loc[train_indices], U50_embeddings_train)
    y_train = df.loc[train_indices].label
    x_valid = graph_from_pdb(df.loc[valid_indices], U50_embeddings_valid)
    y_valid = df.loc[valid_indices].label

    train_dataset = Dataset(x_train, y_train, batch_size=BATCH_SIZE)
    valid_dataset = Dataset(x_valid, y_valid, batch_size=BATCH_SIZE)

    model = GraphAttentionNetwork(
        atom_dim=1024,
        hidden_units=HIDDEN_UNITS,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        dropout_rate=DROPOUT_RATE,
        l2_reg=L2_REG
    )

    # Model compilation
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        # optimizer=AdamW(learning_rate=LEARNING_RATE, weight_decay=L2_REG),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy', AUC(name="auc"), Precision(name="precision"), Recall(name="recall")]
    )

    # callback
    model_checkpoint_path = f"./model_fold_{fold}.h5"
    fold_model_paths.append(model_checkpoint_path)

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_checkpoint_path, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    f1_callback = F1Callback((valid_dataset, y_valid))

    # train
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping, model_checkpoint, reduce_lr, f1_callback]
    )

    gc.collect()

    # Training set prediction results
    train_preds = model.predict(train_dataset).flatten()
    train_pred_binary = (train_preds > 0.5).astype(int)
    y_train_np = np.array(y_train)

    # Calculate various indicators of the training set
    train_f1 = f1_score(y_train_np, train_pred_binary)
    train_precision = precision_score(y_train_np, train_pred_binary)
    train_recall = recall_score(y_train_np, train_pred_binary)
    train_auc = roc_auc_score(y_train_np, train_preds)
    train_accuracy = accuracy_score(y_train_np, train_pred_binary)
    train_loss = history.history.get('loss', [None])[-1]

    # save_train_results_to_csv(fold, train_indices, train_preds, y_train, train_f1, train_precision, train_recall, train_auc, train_accuracy,    train_results_output_file)
    results.append({
        "fold": fold,
        "train_loss": history.history.get('train_loss', [None])[-1],
        "train_accuracy": train_accuracy,
        "train_auc": train_auc,
        "train_precision": train_precision,
        "train_recall": train_recall,
        "train_f1": train_f1
    })

    valid_pred = model.predict(valid_dataset).flatten()
    valid_pred_binary = (valid_pred > 0.5).astype(int)
    y_valid_np = np.array(y_valid)

    val_f1 = f1_score(y_valid_np, valid_pred_binary)
    val_precision = precision_score(y_valid_np, valid_pred_binary)
    val_recall = recall_score(y_valid_np, valid_pred_binary)
    val_auc = roc_auc_score(y_valid_np, valid_pred)
    val_accuracy = accuracy_score(y_valid_np, valid_pred_binary)
    val_loss = history.history.get('val_loss', [None])[-1]

    results.append({
        "fold": fold,
        "val_loss": history.history.get('val_loss', [None])[-1],
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1
    })

    print(f"第 {fold} 折 - Val F1: {val_f1:.4f}")
    
    val_pred_df = pd.DataFrame({
        "Fold": fold,
        "True Label": y_valid_np,
        "Predicted Label": valid_pred_binary,
        "Predicted Probability": valid_pred
    })

    val_pred_path = "./all_folds_val_predictions.csv"
    if fold == 1:
        val_pred_df.to_csv(val_pred_path, index=False, mode='w')
    else:
        val_pred_df.to_csv(val_pred_path, index=False, mode='a', header=False)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model.save_weights(best_model_path)

    gc.collect()

results_df = pd.DataFrame(results)
results_df.to_csv("./cross_validation_metrics.csv", index=False)

print("\n===== Independent test set evaluation =====")
x_test = graph_from_pdb(df_test, U50_test)
y_test = df_test.label
test_dataset = Dataset(x_test, y_test, batch_size=BATCH_SIZE)

all_test_preds = []
for path in fold_model_paths:
    model.load_weights(path)
    test_pred = model.predict(test_dataset).flatten()
    all_test_preds.append(test_pred)

avg_test_pred = np.mean(all_test_preds, axis=0)
test_pred_binary = (avg_test_pred > 0.5).astype(int)
y_test_np = np.array(y_test)

test_f1 = f1_score(y_test_np, test_pred_binary)
test_precision = precision_score(y_test_np, test_pred_binary)
test_recall = recall_score(y_test_np, test_pred_binary)
test_auc = roc_auc_score(y_test_np, avg_test_pred)
test_accuracy = accuracy_score(y_test_np, test_pred_binary)

print(f"Test F1: {test_f1:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

df_test_result = df_test.copy()
df_test_result['Predicted Label'] = test_pred_binary
df_test_result['Predicted Probability'] = avg_test_pred

if 'id' in df_test_result.columns:
    id_column = 'id'
elif 'name' in df_test_result.columns:
    id_column = 'name'
else:
    id_column = df_test_result.columns[0]  

# df_test_result[[id_column, 'LLPS_score']].to_csv("./independent_test_LLPS_scores.csv", index=False)

# 按顺序保存
df_test_result[[id_column, 'label', 'Predicted Label', 'Predicted Probability']].to_csv(
    "./independent_test_scores.csv", index=False
)
