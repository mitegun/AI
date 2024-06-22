import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from skrebate import ReliefF

# Load data
train_X = pd.read_excel('train_X.xlsx')
train_Y = pd.read_excel('train_y.xlsx')

# Drop the first unnamed column
train_X = train_X.iloc[:, 1:]

# Extract the target variable
y = train_Y['Deadin_D14']

# Define feature columns
feature_columns = [
    'Age', 'Days to receive antifungal therapy (treatment on index day=1)', 'Days after index(Follow on index day =1)',
    'Temperature', 'Systolic pressure', 'Diastolic pressure', 'Heart rate', 'Respiratory rate', 'Na', 'K', 'Scr', 'Hct', 
    'WBC', 'Glasgow Coma Score', 'FiO2', 'Platelet count(×10^3/uL)', 'Empirical duration', 'Acute Renal failure', 
    'Treatment with appropriate selection', 'Surgical ward', 'Medical ward', 'Hema', 'Myocardial infarction', 'Heart failure', 
    'Peripheral vascular disease', 'Cerebrovascular disease', 'Dementia', 'Chronic pulmonary disease', 'Connective tissue disease', 
    'Peptic ulcer', 'Mild liver disease', 'Diabetes', 'Diabetes with end organ damage', 'Hemiplegia', 'Moderatetosevererenaldisease', 
    'Anytumor', 'Leukaemia', 'Moderate to severe liver disease', 'Metastatic solid tumor', 'Septic shock (SBP < 90 or vassopressor)', 
    'Neutropenia (ANC<1500)', 'Thrombocytopenia ( < 50,000/uL)', 'ICU admission', 'Dialysis requirement', 'Mechanical ventilation', 
    'Urinary catheter', 'Parenteral nutrition', 'Systemic steroids (20mg prednisone for at least 1week or >= 0.3 mg/kg/day >= 3weeks)', 
    'Primary', 'Intravascular catheter-related', 'Abdominal (cIAI)', 'Urinary tract', 'Others (2 or more sources)', 'Source control other than CVC', 
    'No CVC', 'CVC removal', 'Susceptibility testing', 'Appropriate antifungal therapy', 'Empiric adequate antifungal agent', 'Empiric with Fluconazole', 
    'Empirical with Echinocandin', 'Treatment with appropriate dosing', 'Treatment with loading dose', 'Notreatment', 'Treatment with Fluconazole', 
    'Treatment with Posaconazole', 'Treatment with Echinocandin', 'Gender', 'Mentalstatus_Alert', 'Mentalstatus_Comatose', 'Mentalstatus_Disoriented', 
    'Mentalstatus_Stuporous', 'Candida Albicans', 'Candida Glabrata', 'Candida Others', 'Candida Parapsilosis', 'Candida Tropicalis'
]

# Set the column names
train_X.columns = feature_columns

# Step 1: Feature Cleaning - Remove features with missing rate ≥30%
missing_rates = train_X.isnull().mean()
features_to_keep = missing_rates[missing_rates < 0.3].index
train_X_cleaned = train_X[features_to_keep]

# Step 2: Missing Data Handling - Use MICE for imputation
imputer = IterativeImputer(random_state=42)
train_X_imputed = pd.DataFrame(imputer.fit_transform(train_X_cleaned), columns=train_X_cleaned.columns)
train_X_imputed_array = train_X_imputed.values

# Step 3: Feature Ranking - Use ReliefF to rank features
relief = ReliefF(n_neighbors=100000)
relief.fit(train_X_imputed_array, y)
feature_importances = pd.Series(relief.feature_importances_, index=train_X_imputed.columns)
feature_importances.sort_values(ascending=False, inplace=True)

# Step 4: Feature Selection - Select top 50 features
top_features = feature_importances.head(50).index
train_X_selected = train_X_imputed[top_features]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_X_selected, y, test_size=0.2, random_state=42)

# Base classifiers
classifiers = {
    'nb': GaussianNB(),
    'knn': KNeighborsClassifier(),
    'lr': LogisticRegression(max_iter=100000, random_state=42),
    'rf': RandomForestClassifier(random_state=42)
}

# Get predictions from base classifiers
meta_data = np.zeros((X_val.shape[0], len(classifiers)))
for i, (name, clf) in enumerate(classifiers.items()):
    clf.fit(X_train, y_train)
    meta_data[:, i] = clf.predict(X_val)

# Meta-classifier
meta_clf = SVC(probability=True, random_state=42)
meta_clf.fit(meta_data, y_val)

# Predict on validation set using meta-classifier
meta_data_train = np.zeros((X_train.shape[0], len(classifiers)))
for i, (name, clf) in enumerate(classifiers.items()):
    meta_data_train[:, i] = cross_val_predict(clf, X_train, y_train, cv=5, method='predict')

meta_clf.fit(meta_data_train, y_train)

# Evaluate the stacked model
y_pred = meta_clf.predict(meta_data)
y_prob = meta_clf.predict_proba(meta_data)[:, 1]

# Calculate metrics
f1 = f1_score(y_val, y_pred)
mcc = matthews_corrcoef(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_prob)

print(f'F1 Score: {f1}')
print(f'MCC: {mcc}')
print(f'ROC AUC: {roc_auc}')

# Load the test data
test_X = pd.read_excel('test_X.xlsx')

# Drop the first unnamed column in the test data
test_X = test_X.iloc[:, 1:]

# Set the column names
test_X.columns = feature_columns

# Select important features
test_X_cleaned = test_X[features_to_keep]
test_X_imputed = pd.DataFrame(imputer.transform(test_X_cleaned), columns=test_X_cleaned.columns)
test_X_selected = test_X_imputed[top_features]

# Predict using the base classifiers
meta_data_test = np.zeros((test_X_selected.shape[0], len(classifiers)))
for i, (name, clf) in enumerate(classifiers.items()):
    meta_data_test[:, i] = clf.predict(test_X_selected)

# Predict using the meta-classifier
test_predictions = meta_clf.predict(meta_data_test)
test_probabilities = meta_clf.predict_proba(meta_data_test)[:, 1]

# Create a DataFrame for the results
results = pd.DataFrame({
    'prediction': test_predictions,
    'probability': test_probabilities
})

# Save the results to an Excel file
results.to_excel('test_results.xlsx', index=False)

# Load the actual labels from train_y.xlsx
train_y = pd.read_excel('train_y.xlsx')

# Extract the actual labels


# Load the test results
test_results = pd.read_excel('test_results.xlsx')

# Compare the predicted labels with the actual labels
comparison = pd.DataFrame({
    'Predicted': test_results['prediction'],
    'Probability': test_results['probability']
})

# Print the comparison
print(comparison)


