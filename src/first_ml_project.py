import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create a sample dataset
df = pd.DataFrame({
    "hours_studied": [1, 2, 3, 4, 5, 6],
    "marks": [35, 42, 50, 58, 65, 75]
})

X = df[["hours_studied"]]
y = df["marks"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make a prediction
prediction = model.predict(
    pd.DataFrame({"hours_studied": [7]})
)

print("Dataset:")
print(df)
print(f"\nPredicted marks for 7 study hours: {prediction[0]:.2f}")
print(f"Model coefficient: {model.coef_[0]:.2f}")
print(f"Model intercept: {model.intercept_:.2f}")

# Visualize the fitted model
plt.scatter(X["hours_studied"], y, label="Actual data")
plt.plot(X["hours_studied"], model.predict(X), label="Regression line")
plt.xlabel("Hours studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.legend()
plt.savefig("study_hours_vs_marks.png")
print("\nPlot saved to study_hours_vs_marks.png")
