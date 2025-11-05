import pandas as pd

df = pd.read_json('outputs/mc1_results.jsonl', lines=True)

total = df.count()
print(f'Total Count: {total["hallucination"]}')
hallucination = df[df['hallucination'] == True].count()
print(f'Hallucination Count: {hallucination["hallucination"]}')
print(f'Correct Count: {total["hallucination"] - hallucination["hallucination"]}')
print(f'Hallucination Rate: {hallucination["hallucination"] / total["hallucination"]:.2%}')