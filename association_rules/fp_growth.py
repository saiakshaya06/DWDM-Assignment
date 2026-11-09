
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Step 1: Load dataset
df = pd.read_csv("dataset/mushrooms.csv")

# Step 2: Convert rows into transactions
transactions = []

for index, row in df.iterrows():
    items = []

    for column in df.columns:
        items.append(column + "=" + str(row[column]))

    transactions.append(items)

# Step 3: Convert transactions into Boolean format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

transaction_df = pd.DataFrame(
    te_array,
    columns=te.columns_
)

# Step 4: Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(
    transaction_df,
    min_support=0.05,
    use_colnames=True,
    max_len=3
)

# Step 5: Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

# Step 6: Display results
print("\nFP-GROWTH FREQUENT ITEMSETS:")
print(frequent_itemsets.head(10))

print("\nFP-GROWTH ASSOCIATION RULES:")
print(rules[["antecedents", "consequents",
             "support", "confidence", "lift"]].head(10))

# Step 7: Save results
frequent_itemsets.to_csv(
    "association_rules/fpgrowth_itemsets.csv",
    index=False
)

rules.to_csv(
    "association_rules/fpgrowth_rules.csv",
    index=False
)

print("\nFP-Growth algorithm completed successfully!")
