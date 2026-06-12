import pandas as pd
from scipy import stats

# ==========================================
# HEART DISEASE T-TEST
# ==========================================

heart = pd.read_csv(
    "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
)

group0 = heart[
    heart['target']==0
]['chol']

group1 = heart[
    heart['target']==1
]['chol']

t_stat, p_val = stats.ttest_ind(
    group0,
    group1
)

print("T-Test")
print("Statistic:", t_stat)
print("P-Value:", p_val)

decision1 = (
    "Reject H0"
    if p_val < 0.05
    else "Fail to Reject H0"
)

# ==========================================
# HAPPINESS PEARSON CORRELATION
# ==========================================

happy = pd.read_csv("2023.csv")

corr_stat, corr_p = stats.pearsonr(
    happy['Logged GDP per capita'],
    happy['Ladder score']
)

print("\nPearson Correlation")
print("Statistic:", corr_stat)
print("P-Value:", corr_p)

decision2 = (
    "Reject H0"
    if corr_p < 0.05
    else "Fail to Reject H0"
)

# ==========================================
# TITANIC CHI-SQUARE
# ==========================================

titanic = pd.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)

table = pd.crosstab(
    titanic['Survived'],
    titanic['Pclass']
)

chi2, chi_p, dof, expected = stats.chi2_contingency(table)

print("\nChi Square")
print("Statistic:", chi2)
print("P-Value:", chi_p)

decision3 = (
    "Reject H0"
    if chi_p < 0.05
    else "Fail to Reject H0"
)

# ==========================================
# RESULTS TABLE
# ==========================================

results = pd.DataFrame({
    "Test Scenario":[
        "Cholesterol & Heart Disease",
        "GDP vs Happiness Score",
        "Survival vs Passenger Class"
    ],
    "Test Used":[
        "Independent t-test",
        "Pearson Correlation",
        "Chi-Square Test"
    ],
    "Statistic":[
        t_stat,
        corr_stat,
        chi2
    ],
    "P-Value":[
        p_val,
        corr_p,
        chi_p
    ],
    "Decision":[
        decision1,
        decision2,
        decision3
    ]
})

print("\nRESULTS TABLE")
print(results)
