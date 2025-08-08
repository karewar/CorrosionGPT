import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptors = []
    for desc_name, desc_func in Descriptors._descList:
        try:
            descriptors.append(desc_func(mol))
        except:
            descriptors.append(np.nan) # Handle cases where descriptor calculation fails
    return dict(zip(descriptor_names, descriptors))

df = pd.read_excel("Fe-HCl-317_dataset.xlsx")

original_smiles = df["canonical_SMILES"]

descriptor_data = []
for smiles in df["canonical_SMILES"]:
    desc = calculate_descriptors(smiles)
    if desc:
        descriptor_data.append(desc)
    else:
        descriptor_data.append({name: np.nan for name, _ in Descriptors._descList})

desc_df = pd.DataFrame(descriptor_data)
df = pd.concat([df, desc_df], axis=1)

# Replace infinite values with NaN and then drop columns with NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns that have all NaN values
df = df.dropna(axis=1, how='all')

# Drop rows that have any NaN values in the numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns
df = df.dropna(subset=numeric_cols)

# Keep only numeric columns for model training
X = df.select_dtypes(include=np.number).drop(columns=["mole_id", "IE_exp"])
y = df["IE_exp"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Predict for all data to find best inhibitors
df["IE_predicted"] = gbr.predict(X)

best_inhibitors = df.sort_values(by="IE_predicted", ascending=False).head(10)

print("\nTop 10 Predicted Corrosion Inhibitors:")
print(best_inhibitors[["canonical_SMILES", "IE_predicted"]])

df.to_csv("predicted_inhibitors.csv", index=False)

