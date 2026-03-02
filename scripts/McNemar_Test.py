import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

df = pd.read_csv("results/splunk_model_predictions.csv")

MODEL_A = "dl_cnn"
MODEL_B = "random_forest"

a = df[df["model"] == MODEL_A].reset_index(drop=True)
b = df[df["model"] == MODEL_B].reset_index(drop=True)

assert len(a) == len(b), "Test set sizes are different!"

a_correct = a["pred_label"] == a["true_label"]
b_correct = b["pred_label"] == b["true_label"]

n00 = (a_correct & b_correct).sum()      
n01 = (a_correct & ~b_correct).sum()     
n10 = (~a_correct & b_correct).sum()     
n11 = (~a_correct & ~b_correct).sum()   

table = [[int(n00), int(n01)],
         [int(n10), int(n11)]]

print("Contingency Table:")
print(table)

result = mcnemar(table, exact=False, correction=True)

print("Statistic:", result.statistic)
print("p-value:", result.pvalue)
