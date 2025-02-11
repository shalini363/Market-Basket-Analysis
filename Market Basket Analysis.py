#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install mlxtend


# In[3]:


# Libraries
import pandas as pd  # For data handling
import time  # For time tracking
import itertools  # For combinations
from mlxtend.preprocessing import TransactionEncoder  # For transaction encoding
from mlxtend.frequent_patterns import apriori, association_rules  # For Apriori algorithm and association rules
import pyfpgrowth  # For FP-Growth algorithm


# ### Part 1: Generate Transactions and Save to CSV

# In[4]:


# Read CSV File and to convert it into a List of Transactions
def read_transactions(file_path):
    transactions = []
    with open(file_path, 'r') as file:
        csv_reader = pd.read_csv(file)
        for index, row in csv_reader.iterrows():
            transactions.append(row['Transaction'].split(', '))
    return transactions


# In[5]:


# To convert List of Transactions into MLxtend-Compatible Format
def convert_to_mlxtend_format(transactions):
    transaction_encoder = TransactionEncoder()
    transaction_array = transaction_encoder.fit(transactions).transform(transactions)
    dataframe = pd.DataFrame(transaction_array, columns=transaction_encoder.columns_)
    return dataframe


# ### Part 2: Brute Force Method for Generating Frequent Itemsets and Association Rules

# In[6]:


# Brute Force Method
def generate_frequent_itemsets_brute_force(transactions, support_threshold):
    items = set(item for transaction in transactions for item in transaction)
    frequent_itemsets = {}
    
    # Created every 1-itemset that could be imagined and checked the frequency
    for item in items:
        frequency = sum(1 for transaction in transactions if item in transaction)
        if frequency / len(transactions) >= support_threshold:
            frequent_itemsets[(item,)] = frequency
    
    # created every 2-itemset that could be imagined and checked the frequency
    for itemset in itertools.combinations(items, 2):
        frequency = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
        if frequency / len(transactions) >= support_threshold:
            frequent_itemsets[itemset] = frequency
    
    # Until no more frequent itemsets could be found, all potential k-itemsets were generated.
    k = 3
    while True:
        itemsets = itertools.combinations(items, k)
        found_frequent_itemsets = False
        
        for itemset in itemsets:
            frequency = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
            if frequency / len(transactions) >= support_threshold:
                frequent_itemsets[itemset] = frequency
                found_frequent_itemsets = True
                
        if not found_frequent_itemsets:
            break
        
        k += 1
    
    return frequent_itemsets


# In[7]:


# Apriori Algorithm
def run_apriori_algorithm(transactions, support_threshold):
    dataframe = convert_to_mlxtend_format(transactions)
    frequent_itemsets = apriori(dataframe, min_support=support_threshold, use_colnames=True)
    return frequent_itemsets


# In[8]:


pip install mlxtend pyfpgrowth


# In[9]:


# FP-Growth Algorithm
def run_fp_growth(transactions, support_threshold):
    patterns = pyfpgrowth.find_frequent_patterns(transactions, support_threshold * len(transactions))
    frequent_itemsets = pd.DataFrame(list(patterns.items()), columns=['itemsets', 'support'])
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(frozenset)
    return frequent_itemsets


# In[10]:


# Generate Association Rules Function from Frequently Occurring Itemsets
def generate_association_rules(frequent_itemsets, transactions, confidence_threshold):
    dataframe = convert_to_mlxtend_format(transactions)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_threshold, support_only=True)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
    return rules


# In[11]:


# A function to compare each of the three algorithms' performance times and results
def compare_algorithms(file_path, support_threshold, confidence_threshold):
    transactions = read_transactions(file_path)
    
    # Brute Force Method
    start_time = time.time()
    brute_force_itemsets = generate_frequent_itemsets_brute_force(transactions, support_threshold)
    brute_force_time = time.time() - start_time
    print(f"Brute Force Method: {len(brute_force_itemsets)} frequent itemsets found in {brute_force_time:.5f} seconds.")
    #print(brute_force_itemsets)
    
    # Apriori Algorithm
    start_time = time.time()
    apriori_itemsets = run_apriori_algorithm(transactions, support_threshold)
    apriori_rules = generate_association_rules(apriori_itemsets, transactions, confidence_threshold)
    apriori_time = time.time() - start_time
    print(f"Apriori Algorithm: {len(apriori_itemsets)} frequent itemsets and {len(apriori_rules)} rules found in {apriori_time:.5f} seconds.")
    #print(apriori_itemsets)
    
    # FP-Growth Algorithm
    start_time = time.time()
    fpgrowth_itemsets = run_fp_growth(transactions, support_threshold)
    fpgrowth_rules = generate_association_rules(fpgrowth_itemsets, transactions, confidence_threshold)
    fpgrowth_time = time.time() - start_time
    print(f"FP-Growth Algorithm: {len(fpgrowth_itemsets)} frequent itemsets and {len(fpgrowth_rules)} rules found in {fpgrowth_time:.5f} seconds.")
    #print(fpgrowth_itemsets)
    
    # Fastest algorithm
    fastest_time = min(brute_force_time, apriori_time, fpgrowth_time)
    fastest_algo = 'Brute Force' if fastest_time == brute_force_time else ('Apriori' if fastest_time == apriori_time else 'FP-Growth')
    print(f"The fastest algorithm for {file_path} is {fastest_algo} with a time of {fastest_time:.5f} sec.\n")
    
    return brute_force_itemsets, apriori_rules, fpgrowth_rules


# In[12]:


# Main function
def main():
    support_threshold = float(input("Enter the fractional support threshold: "))
    confidence_threshold = float(input("Enter the fractional confidence threshold: "))
    file_paths = ['Amazon.csv','Best_Buy.csv','Generic.csv','K-Mart.csv','Nike.csv']
    for file_path in file_paths:
        print(f"Working through {file_path}")
        brute_force_itemsets, apriori_rules, fpgrowth_rules = compare_algorithms(file_path, support_threshold, confidence_threshold)
        print(f"The number of rules formed by Brute Force: {len(brute_force_itemsets)}")
        print(f"The number of rules formed by Apriori: {len(apriori_rules)}")
        print(f"The number of rules formed by FP-Growth: {len(fpgrowth_rules)}")
        print("\n")

if __name__ == "__main__":
    main()


# In[ ]:




