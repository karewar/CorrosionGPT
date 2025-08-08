# CorrosionGPT

**CorrosionGPT** is an AI-powered tool that predicts the efficiency of chemical compounds as corrosion inhibitors and generates novel synthetic candidates. It uses advanced language models and AI agents to address the widespread issue of corrosion across industries.

---

## 🚀 Features

- Predict corrosion inhibition efficiency from molecular representations.
- Generate new potential inhibitors using generative AI.
- Analyze and visualize structured chemical datasets.
- Supports data in both JSON and Excel formats.

---

## 📊 Datasets

### 🧪 Experimental Dataset
- `Fe-HCl-317_dataset.xlsx`: Contains real-world measurements of corrosion rates and inhibitor efficiencies in Fe-HCl systems.

### 📁 Formatted JSON Datasets
- `CoInDataset1_formatted.json`
- `CoInDataset2_formatted.json`
- `Fe_HCl_dataset_formatted.json`

Each JSON entry follows this structure:

```json
{
  "compound_name": "ExampleCompound",
  "smiles": "CCN(CC)CC",
  "efficiency": 88.5,
  "temperature": 298,
  "environment": "Fe-HCl"
}

---```

🧠 Model Training
Use the provided script to train a model on the corrosion inhibitor dataset:
python train_model.py


This script:

Loads the JSON datasets.
Featurizes molecular data (e.g., via RDKit).
Trains a machine learning model (e.g., Random Forest or any specified regressor).
Outputs training results and performance metrics.


📈 Data Analysis
Run the following script to analyze and visualize dataset trends:
python analyze_dataset.py


This will:
Plot efficiency distributions.
Identify correlations (e.g., between temperature and efficiency).
Provide insights into experimental conditions and compound performance.


📦 Output
After training or generation:

predicted_inhibitors.csv will contain predicted or generated compounds along with their expected inhibition efficiency.


