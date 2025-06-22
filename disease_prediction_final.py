import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("generated_disease_prediction_data.csv")
df['Text'] = df['Symptoms'].astype(str) + ' ' + df['Habits'].astype(str)

# Encode Disease
le = LabelEncoder()
df['DiseaseEncoded'] = le.fit_transform(df['Disease'])

# Vectorize text
cv = CountVectorizer()
X_text = cv.fit_transform(df['Text']).toarray()

# Combine with Age
X = pd.concat([df[['Age']].reset_index(drop=True), pd.DataFrame(X_text)], axis=1)
X.columns = X.columns.astype(str)

# Targets
y_class = df['DiseaseEncoded']
y_reg = df['Risk']

# Classification model
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Decision tree model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Regression model
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
regressor = RandomForestRegressor()
regressor.fit(X_train_reg, y_train_reg)

# Function: check person by name
def check_person(name_query):
    person_row = df[df["Name"].str.lower() == name_query.strip().lower()]
    if person_row.empty:
        print(f"No record found for '{name_query}'")
        return

    age = person_row.iloc[0]["Age"]
    text = person_row.iloc[0]["Symptoms"] + " " + person_row.iloc[0]["Habits"]
    vector = cv.transform([text]).toarray()
    input_data = pd.concat([pd.DataFrame([[age]], columns=["Age"]), pd.DataFrame(vector)], axis=1)
    input_data.columns = input_data.columns.astype(str)

    disease_pred_encoded = clf.predict(input_data)[0]
    risk_pred = regressor.predict(input_data)[0]
    disease_pred = le.inverse_transform([disease_pred_encoded])[0]
    health_rank = max(0, min(10, 10 - round(risk_pred * 2)))

    print(f"\nName: {person_row.iloc[0]['Name']}")
    print(f"Health Rank: {health_rank}")
    print(f"Predicted Disease: {disease_pred}")

# Ask user for input
if __name__ == "__main__":
    name_input = input("Enter patient name to analyze: ")
    check_person(name_input)
