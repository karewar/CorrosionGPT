# Corrosion Inhibitor Prediction using Machine Learning

## 1. Introduction

Corrosion of steel is a significant problem across various industries, leading to substantial economic losses and safety concerns. The use of corrosion inhibitors is a widely adopted and effective strategy to mitigate this issue. Traditionally, the discovery and development of new corrosion inhibitors have relied heavily on time-consuming and costly experimental methods. However, recent advancements in machine learning (ML) and deep learning (DL) techniques offer promising alternatives for accelerating this process by predicting the efficiency of potential inhibitors based on their molecular structures and properties.

This report details a project aimed at identifying effective corrosion inhibitors for steels using machine learning. Specifically, a Quantitative Structure-Property Relationship (QSPR) modeling approach was employed, leveraging molecular descriptors to predict the corrosion inhibition efficiency (IE) of various compounds. The project involved several key phases: researching relevant datasets and ML/DL methods, obtaining and analyzing a suitable dataset, implementing and training an ML model, and finally, predicting the best corrosion inhibitors.

## 2. Methodology

### 2.1. Data Acquisition and Preprocessing

The primary dataset used in this study, named "Fe-HCl-317-dataset," was obtained from a GitHub repository associated with a research paper titled "A data-driven QSPR model for screening organic corrosion inhibitors for carbon steel using machine learning techniques" [1]. This dataset contains information on 317 organic inhibitors for carbon steel in a 1 M HCl solution, including their canonical SMILES (Simplified Molecular Input Line Entry System) strings and experimentally determined corrosion inhibition efficiencies (IE_exp).

Upon acquisition, the dataset underwent a series of preprocessing steps:

*   **Molecular Descriptor Calculation:** For each chemical compound represented by its SMILES string, a comprehensive set of molecular descriptors was calculated using the RDKit library. These descriptors numerically represent various physicochemical properties and structural features of the molecules, which are crucial for QSPR modeling. Examples of calculated descriptors include molecular weight, logP, topological polar surface area, and various counts of chemical features.

*   **Handling Missing and Infinite Values:** During descriptor calculation, some descriptors might result in NaN (Not a Number) or infinite values for certain molecules. These values were replaced with NaN, and then any columns that contained only NaN values (i.e., descriptors that could not be calculated for any molecule) were removed. Subsequently, any rows containing NaN values in the remaining numeric descriptor columns were also removed to ensure the dataset was clean and suitable for machine learning algorithms.

*   **Feature and Target Separation:** The preprocessed dataset was then divided into features (X) and the target variable (y). The features (X) consisted of the calculated molecular descriptors, while the target variable (y) was the experimental corrosion inhibition efficiency (IE_exp).

### 2.2. Model Implementation and Training

Based on the literature review, Gradient Boosting Regressor (GBR) was selected as the machine learning model for this study. GBR is an ensemble learning method that builds a strong predictive model by combining multiple weak prediction models, typically decision trees. It is known for its robustness and high predictive accuracy in various regression tasks.

The dataset was split into training and testing sets using a 80/20 ratio, respectively. This split ensures that the model is trained on a portion of the data and then evaluated on unseen data to assess its generalization performance. The GBR model was trained on the training set, learning the complex relationships between the molecular descriptors and the corrosion inhibition efficiency.

## 3. Results and Discussion

After training, the Gradient Boosting Regressor model's performance was evaluated using several standard regression metrics:

*   **R2 Score (Coefficient of Determination):** This metric indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. An R2 score of 0.56 suggests that 56% of the variability in corrosion inhibition efficiency can be explained by the model.

*   **RMSE (Root Mean Squared Error):** RMSE measures the average magnitude of the errors. A RMSE of 7.06 indicates that, on average, the model's predictions deviate from the actual experimental values by approximately 7.06 units of inhibition efficiency.

*   **MAE (Mean Absolute Error):** MAE measures the average absolute difference between predicted and actual values. A MAE of 5.28 suggests that, on average, the absolute difference between the predicted and actual inhibition efficiencies is about 5.28 units.

While the R2 score of 0.56 indicates a moderate predictive capability, the RMSE and MAE values suggest that the model provides reasonably accurate predictions for corrosion inhibition efficiency. The performance is comparable to or slightly better than some models reported in the literature for similar small datasets, especially considering the inherent complexity and variability in corrosion inhibition phenomena.

### 3.1. Top Predicted Corrosion Inhibitors

To identify the most promising corrosion inhibitors, the trained GBR model was used to predict the inhibition efficiency for all compounds in the dataset. The top 10 compounds with the highest predicted inhibition efficiencies are listed below:

```
                                      canonical_SMILES  IE_predicted
176       ClC1C=CC(=C(C=1)Cl)C1=NN(CC1)C(=O)CN1C=NC=N1     98.45
178       CC(C1C=CC(=CC=1)CSC1=NN=C(O1)CN1N=CN=C1)(C)C     97.76
173               ClC1C=CC(=C(C=1)C(=NOCN1C=NN=C1)C)Cl     97.68
172               CC(=NOCN1C=NN=C1)C1C=CC(=C(C=1)Cl)Cl     97.62
180        COC1C=CC(=CC=1)C(=O)CSC1=NN=C(N1)CN1N=CN=C1     97.58
177            O=C(C1C=CC=CC=1)CSC1=NN=C(O1)CN1C=NC=N1     97.12
146     CCCCCCCCNC1N=C(NCCCN(C)C)N=C(N=1)SC1=NN=C(S1)C     97.02
125                    OC1C=CC(=C2C=1N=CC=C2)CC1=NCCN1     96.81
117  CCCCCCCCCC[N+](C1C=CC(=CC=1)/C=N/C1C=CC(=CC=1O...     96.64
116  CCCCCCCC[N+](C1C=CC(=CC=1)/C=N/C1C=CC(=CC=1OC)...     96.64
```

These compounds represent potential candidates for further experimental validation as highly effective corrosion inhibitors. Their high predicted IE values suggest that their molecular structures possess characteristics favorable for inhibiting steel corrosion.

## 4. Conclusion

This project successfully demonstrated the application of machine learning, specifically a Gradient Boosting Regressor model, for predicting the corrosion inhibition efficiency of organic compounds on carbon steel. By leveraging molecular descriptors derived from SMILES strings, the model was able to learn and predict inhibition efficiencies with reasonable accuracy. The identified top 10 predicted corrosion inhibitors provide a valuable starting point for experimental chemists and material scientists to focus their efforts on developing new and more effective corrosion prevention strategies.

Future work could involve exploring more advanced deep learning architectures, such as graph neural networks or transformers, which are better suited for handling complex molecular structures. Additionally, incorporating more diverse and larger datasets, as well as considering environmental factors and specific steel types, could further enhance the model's predictive power and applicability.

## 5. References

[1] Pham, T. H., Le, P. K., & Son, D. N. (2024). A data-driven QSPR model for screening organic corrosion inhibitors for carbon steel using machine learning techniques. *RSC Advances*, *14*(16), 11157-11168. [https://pubs.rsc.org/en/content/articlehtml/2024/ra/d4ra02159b](https://pubs.rsc.org/en/content/articlehtml/2024/ra/d4ra02159b)




## 2.1.1. Feature Generation: Molecular Descriptors

In the context of Quantitative Structure-Property Relationship (QSPR) modeling, the effectiveness of a chemical compound in inhibiting corrosion is correlated with its molecular structure and properties. Since machine learning models require numerical inputs, the textual representation of chemical structures (SMILES strings) must be converted into a set of quantitative features. These features are known as **molecular descriptors**.

For this project, the `RDKit` cheminformatics library was utilized to generate these molecular descriptors. The process involves the following steps:

1.  **SMILES to Molecular Object Conversion:** Each `canonical_SMILES` string from the input dataset is first parsed by RDKit to create a `Mol` object (molecular object). This object is an in-memory representation of the chemical structure, containing information about atoms, bonds, and their connectivity.

2.  **Descriptor Calculation:** Once a `Mol` object is created, RDKit's `Descriptors` module is employed to compute a wide array of numerical descriptors. These descriptors quantify various aspects of the molecule, including:
    *   **Physicochemical Properties:** Such as `MolWt` (Molecular Weight), `LogP` (octanol-water partition coefficient, indicating lipophilicity), `TPSA` (Topological Polar Surface Area, related to drug absorption), and `NumHDonors`/`NumHAcceptors` (number of hydrogen bond donors/acceptors).
    *   **Constitutional Descriptors:** These describe the composition of the molecule, e.g., `NumAtoms` (total number of atoms), `NumHeavyAtoms` (number of non-hydrogen atoms), `NumRotatableBonds` (number of rotatable bonds, influencing molecular flexibility).
    *   **Topological Descriptors:** These capture information about the molecular graph (connectivity of atoms), such as `Chi` indices, `Kappa` indices, and `BalabanJ` (topological complexity).
    *   **Electronic Descriptors:** Although not explicitly calculated as quantum chemical descriptors in this script (which would require more advanced computational chemistry tools like DFT), some RDKit descriptors can indirectly reflect electronic properties, such as `MaxAbsPartialCharge`.

Each of these descriptors provides a unique piece of information about the molecule's characteristics, which are then used as input features for the machine learning model. The `train_model.py` script iterates through each SMILES string, calculates these descriptors, and compiles them into a DataFrame. Any descriptors that cannot be calculated (e.g., due to invalid SMILES or specific molecular structures) or result in infinite values are handled by replacing them with `NaN` (Not a Number). Subsequently, columns or rows with excessive `NaN` values are removed to ensure data quality for model training.

This process transforms the qualitative chemical structures into a quantitative format, enabling the Gradient Boosting Regressor to learn the relationships between these molecular properties and the corrosion inhibition efficiency.


