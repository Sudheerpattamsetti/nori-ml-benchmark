import pandas as pd
import matplotlib.pyplot as plt

# Results from our completed experiments
results = {
    "Model": [
        "Original Random Forest",
        "Tuned Random Forest",
        "XGBoost",
        "LightGBM",
        "Nori 1.5"
    ],
    "CV_MAE": [45.56, 45.29, 45.67, 46.05, 44.43],
    "Test_MAE": [49.37, 49.34, 50.23, 51.07, 48.34],
    "Test_R2": [0.352, 0.349, 0.320, 0.280, 0.406]
}

# Convert results into a DataFrame
df = pd.DataFrame(results)

# Save results as CSV
df.to_csv("benchmark_results.csv", index=False)

print("\nBenchmark Results:")
print(df.to_string(index=False))

# Create comparison chart
plt.figure(figsize=(10, 6))

plt.bar(df["Model"], df["Test_MAE"])

plt.title("ML Model Benchmark - Diabetes Dataset")
plt.xlabel("Machine Learning Models")
plt.ylabel("Test Mean Absolute Error (MAE)")

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

# Save chart
plt.savefig("benchmark_comparison.png", dpi=300)

print("\nReport and chart saved successfully!")