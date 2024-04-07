import os
import pandas as pd
from sklearn.model_selection import train_test_split

directory = 'data/'
label = 'hello'
combined_data = []

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path, header=None)
        df['label'] = label
        combined_data.append(df)

combined_df = pd.concat(combined_data, ignore_index=True)

X = combined_df.iloc[:, :-1]
y = combined_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

os.makedirs(f"{label}_train", exist_ok=True)
os.makedirs(f"{label}_test", exist_ok=True)
os.makedirs(f"{label}_val", exist_ok=True)

for i in range(len(X_train)):
    filename = f"{label}_{i+1:05d}.csv"
    file_path = os.path.join(f"{label}_train", filename)
    X_train.iloc[i].to_csv(file_path, index=False, header=False)
    with open(os.path.join(f"{label}_train", f"{label}_train_labels.txt"), 'a') as f:
        f.write(f"{filename},{label}\n")

for i in range(len(X_test)):
    filename = f"{label}_{i+1:05d}.csv"
    file_path = os.path.join(f"{label}_test", filename)
    X_test.iloc[i].to_csv(file_path, index=False, header=False)
    with open(os.path.join(f"{label}_test", f"{label}_test_labels.txt"), 'a') as f:
        f.write(f"{filename},{label}\n")

for i in range(len(X_val)):
    filename = f"{label}_{i+1:05d}.csv"
    file_path = os.path.join(f"{label}_val", filename)
    X_val.iloc[i].to_csv(file_path, index=False, header=False)
    with open(os.path.join(f"{label}_val", f"{label}_val_labels.txt"), 'a') as f:
        f.write(f"{filename},{label}\n")