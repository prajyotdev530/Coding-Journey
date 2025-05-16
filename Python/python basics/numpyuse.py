import numpy as np
import pandas as pd

# Generate a random dataset (5 rows, 3 columns)
data = np.random.randint(1, 100, (5, 3))

# Convert to Pandas DataFrame
df = pd.DataFrame(data, columns=["Math", "Science", "English"])

# Calculate the average score per subject
averages = df.mean()

# Print results
print("Generated DataFrame:\n", df)
print("\nAverage Scores:\n", averages)