
import numpy as np
import pandas as pd

np.random.seed(42)

# 1. Generate reference (training) data
reference = pd.DataFrame({
    "area": np.random.normal(1200, 250, 1000).clip(400, 3000),
    "bedrooms": np.random.choice([1, 2, 3, 4], 1000, p=[0.1, 0.3, 0.4, 0.2]),
    "age": np.random.normal(15, 7, 1000).clip(0, 50),
    "location": np.random.choice(
        ["Urban", "Suburban", "Rural"],
        1000,
        p=[0.4, 0.4, 0.2]
    )
})

reference["price"] = (
    reference["area"] * 200
    + reference["bedrooms"] * 10000
    - reference["age"] * 500
    + np.random.normal(0, 10000, 1000)
)

# 2. Generate production (incoming) data with changed distributions
production = pd.DataFrame({
    "area": np.random.normal(2000, 400, 500).clip(400, 4000),
    "bedrooms": np.random.choice([1, 2, 3, 4], 500, p=[0.05, 0.15, 0.4, 0.4]),
    "age": np.random.normal(25, 10, 500).clip(0, 60),
    "location": np.random.choice(
        ["Urban", "Suburban", "Rural"],
        500,
        p=[0.2, 0.3, 0.5]
    )
})

production["price"] = (
    production["area"] * 200
    + production["bedrooms"] * 10000
    - production["age"] * 500
    + np.random.normal(0, 10000, 500)
)

# 3. Save both datasets
reference.to_csv("data/reference.csv", index=False)
production.to_csv("data/production.csv", index=False)

print("Datasets generated successfully!")
print("Reference dataset shape:", reference.shape)
print("Production dataset shape:", production.shape)

print("\nReference data preview:")
print(reference.head())

print("\nProduction data preview:")
print(production.head())
