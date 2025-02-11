# **Market Basket Analysis Using Association Rule Mining**

This project performs **Market Basket Analysis** using **three different algorithms**: **Brute Force, Apriori, and FP-Growth**. It identifies frequently purchased itemsets and generates association rules to uncover patterns in customer transactions. The study evaluates algorithm performance in terms of efficiency and accuracy, helping businesses optimize sales strategies.

---

## **Features**

### **1. Association Rule Mining Techniques**
- **Brute Force Method**: Iteratively checks all item combinations, ensuring completeness but at a high computational cost.
- **Apriori Algorithm**: Uses a level-wise approach to generate frequent itemsets, improving efficiency by pruning unnecessary calculations.
- **FP-Growth Algorithm**: Constructs a compact tree structure to optimize performance and reduce memory usage.

### **2. Performance Evaluation**
- **Execution time**: Measures algorithm speed for different dataset sizes.
- **Number of frequent itemsets**: Determines how well each algorithm identifies meaningful patterns.
- **Association rule generation**: Compares the quality and quantity of discovered rules.

### **3. Visualization & Insights**
- **Support, Confidence, and Lift metrics**: Evaluate the strength of generated rules.
- **Heatmaps and Graphs**: Represent itemset correlations and rule distributions.

---

## **Getting Started**

### **Prerequisites**
- **Python 3.8 or higher**
- **Required Libraries**: NumPy, Pandas, Matplotlib, Seaborn, MLxtend
---
Install dependencies using: pip install -r requirements.txt


---

## **Dataset**
- **Source**: Transactional data collected from multiple retail stores.
- **Format**: Each row represents a transaction containing a list of purchased items.
- **Preprocessing**:
  - Converted raw transactions into structured CSV format.
  - Applied one-hot encoding to prepare the data for MLxtend-based analysis.

---

## **How to Run**

1. **Clone this repository:**
  
   git clone https://github.com/shalini363/market-basket-analysis.git

   cd data_mining_midterm
  

3. **Launch the Jupyter Notebook:**
   
   jupyter notebook Market_Basket_Analysis.ipynb
  

4. **Follow the notebook to:**
   - Load and preprocess transaction data.
   - Apply **Brute Force, Apriori, and FP-Growth algorithms**.
   - Generate and visualize **association rules**.

---

## **Results**

### **Algorithm Performance**
| Algorithm   | Execution Time | Frequent Itemsets | Strongest Rules |
|------------|---------------|-------------------|----------------|
| Brute Force | Slow (High Complexity) | Most Complete | Computationally Expensive |
| Apriori | Moderate Speed | Efficient | Strong Rules with High Confidence |
| FP-Growth | Fastest | Optimized for Large Datasets | High Scalability |

### **Association Rules**
- **Support**: Frequency of itemsets in transactions.
- **Confidence**: Likelihood of one item being purchased when another is bought.
- **Lift**: Strength of association beyond random chance.

---

## **Analysis & Key Insights**
- **Apriori vs. FP-Growth**: FP-Growth outperforms Apriori on larger datasets due to efficient tree-based storage.
- **Business Applications**:
  - **Cross-Selling Opportunities**: Identify frequently co-purchased products.
  - **Customer Purchase Patterns**: Personalize recommendations.
  - **Inventory Optimization**: Stock high-demand item combinations.

---

## **Future Work**
- Test on larger datasets with millions of transactions.
- Implement parallel computing for faster rule mining.
- Apply deep learning techniques to enhance association rule accuracy.

---

## **Acknowledgments**
- **Data Source**: Public retail transaction datasets
- **Reference Libraries**: MLxtend, Pandas, Scikit-learn

