# STEP 3: DATA PREPARATION (IMPROVED VERSION)

import pandas as pd
from sklearn.utils import resample


# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("HAM10000_metadata.csv")

print("First 5 rows:")
print(df.head())


print("\nTotal rows:", len(df))


# -----------------------------
# CLASS DISTRIBUTION BEFORE
# -----------------------------
print("\nClass Distribution (Before Balancing):")
print(df['dx'].value_counts())


# -----------------------------
# TARGET SIZE PER CLASS
# -----------------------------
TARGET_PER_CLASS = 1200


# -----------------------------
# SMART BALANCING STRATEGY
# -----------------------------
balanced_dfs = []

for label in df['dx'].unique():

    class_df = df[df['dx'] == label]

    if len(class_df) >= TARGET_PER_CLASS:

        sampled = class_df.sample(
            TARGET_PER_CLASS,
            random_state=42
        )

    else:

        sampled = resample(
            class_df,
            replace=True,
            n_samples=TARGET_PER_CLASS,
            random_state=42
        )

    balanced_dfs.append(sampled)


# -----------------------------
# MERGE DATASETS
# -----------------------------
df_balanced = pd.concat(balanced_dfs)


# -----------------------------
# SHUFFLE DATASET
# -----------------------------
df_balanced = df_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# -----------------------------
# LABEL ENCODING
# -----------------------------
label_map = {

    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6

}

df_balanced['label'] = df_balanced['dx'].map(label_map)


# -----------------------------
# FINAL CHECK
# -----------------------------
print("\nClass Distribution (After Balancing):")
print(df_balanced['dx'].value_counts())


print("\nSample after labeling:")
print(df_balanced.head())


# -----------------------------
# SAVE DATASET
# -----------------------------
df_balanced.to_csv("balanced_data.csv", index=False)


print("\nSTEP 3 COMPLETED SUCCESSFULLY!")
print("Balanced dataset saved as balanced_data.csv")