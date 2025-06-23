import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Giving dataset
df = pd.read_csv("diabetes1.csv")

#Data Visualisation
counts=df['Outcome'].value_counts()
counts.plot(kind='bar',color=['green','red'])
plt.title("Diabedit Prediction")
plt.xticks(ticks=[0,1],labels=['Non-Diabetic','Diabetic'],rotation=0)
plt.ylabel("No of people")
plt.show()

# Spliting dataset into X and Y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train Logistic Regression model
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# Step 4: Training model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy on test data
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
