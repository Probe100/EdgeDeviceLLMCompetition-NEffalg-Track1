import pandas as pd


columns = ['CommonsenseQA', 'BIG-Bench-Hard', 'GSM8K', 'HumanEval', 'CHID', 'TruthfulQA', 'Throughput', 'Memory-Usage']

# replace 0 with your evaluated results.
value = [26.510, 28.123, 21.377, 12.603, 11.590, 0.173, 23.097, 8573.243]
data = {columns[i]: value[i] for i in range(len(columns))}


df = pd.DataFrame(data, index=[0])

csv_filename = 'Results.csv'
df.to_csv(csv_filename, index=False)

print(f"DataFrame saved to {csv_filename}")
