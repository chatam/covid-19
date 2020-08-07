#!/usr/bin/env python
# coding: utf-8

# I imported my project into Jupyter from BQ.

# In[396]:


# Packages

from datetime import datetime
from fbprophet import Prophet
from google.cloud import bigquery 
from google.cloud import bigquery_storage_v1beta1
from google.oauth2 import service_account
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

credentials = service_account.Credentials.from_service_account_file("/Users/cyrus.hatam/Desktop/Syntasa-Demo-40-9757e0bf8a9f.json")
project_id = "syntasa-demo-40"
client = bigquery.Client(credentials = credentials, project = project_id)

import math
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import statsmodels.formula.api as smf


# I chose to take the log of CONFIRMED_CASES, DEATHS, E_TOTPOP, and MEDIAN_INCOME because they each had significant right skews. Since OLS regression assumes a normal distribution, I took the log of these variables in order to fit them to such a model more accurately. The following is my SQL code from BQ:

# In[433]:


# BQ Dataset

sql_query = "SELECT DATE, COUNTY, FIPS, ROUND(LN(CONFIRMED_CASES)) AS CONFIRMED_CASES, ROUND(LN(DEATHS)) AS DEATHS, CONFIRMED_CASES / E_TOTPOP AS CC_PER_CAPITA, DEATHS / E_TOTPOP AS D_PER_CAPITA, DEATHS / CONFIRMED_CASES AS D_PER_CC, ROUND(LN(E_TOTPOP)) AS E_TOTPOP, DIABETES_RATE, SMOKING_RATE, ROUND(LN(MEDIAN_INCOME)) AS MEDIAN_INCOME, PERCENT_DEM, PERCENT_REP, RPL_THEMES, LAG(ROUND(LN(CONFIRMED_CASES)),1) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS CCT_1, LAG(ROUND(LN(CONFIRMED_CASES)),2) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS CCT_2, LAG(ROUND(LN(CONFIRMED_CASES)),3) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS CCT_3, LAG(ROUND(LN(CONFIRMED_CASES)),7) OVER (PARTITION BY COUNTY ORDER BY DATE) AS CCT_7, LAG(ROUND(LN(CONFIRMED_CASES)),14) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS CCT_14, LAG(ROUND(LN(CONFIRMED_CASES)),21) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS CCT_21, LAG(ROUND(LN(CONFIRMED_CASES)),31) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS CCT_31, LAG(DEATHS,1) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS DT_1, LAG(DEATHS,2) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS DT_2, LAG(DEATHS,3) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS DT_3, LAG(DEATHS,7) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS DT_7, LAG(DEATHS,14) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS DT_14, LAG(DEATHS,21) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS DT_21, LAG(DEATHS,31) OVER (PARTITION BY COUNTY ORDER BY DATE ASC) AS DT_31, ((MALE_UNDER_5 / E_TOTPOP) + (MALE_5_TO_9 / E_TOTPOP) + (MALE_10_TO_14 / E_TOTPOP) + (MALE_15_TO_17 / E_TOTPOP)) AS MALE_0_TO_17, ((MALE_18_TO_19 / E_TOTPOP) + (MALE_20 / E_TOTPOP) + (MALE_21 / E_TOTPOP) + (MALE_22_TO_24 / E_TOTPOP) + (MALE_25_TO_29 / E_TOTPOP)) AS MALE_18_TO_29, (MALE_30_TO_34 / E_TOTPOP) + (MALE_35_TO_39 / E_TOTPOP) + (MALE_40_TO_44 / E_TOTPOP) + (MALE_45_TO_49 / E_TOTPOP) AS MALE_30_TO_49, (MALE_50_TO_54 / E_TOTPOP) + (MALE_55_TO_59 / E_TOTPOP) + (MALE_60_TO_61 / E_TOTPOP) + (MALE_62_TO_64 / E_TOTPOP) + (MALE_65_TO_66 / E_TOTPOP) + (MALE_67_TO_69 / E_TOTPOP) + (MALE_70_TO_74 / E_TOTPOP) + (MALE_75_TO_79 / E_TOTPOP) + (MALE_80_TO_84 / E_TOTPOP) + MALE_85_AND_OVER / E_TOTPOP AS MALE_OVER_50, (FEMALE_UNDER_5 / E_TOTPOP) + (FEMALE_5_TO_9 / E_TOTPOP) + (FEMALE_10_TO_14 / E_TOTPOP) + (FEMALE_15_TO_17 / E_TOTPOP) AS FEMALE_0_TO_17, (FEMALE_18_TO_19 / E_TOTPOP) + (FEMALE_20 / E_TOTPOP) + (FEMALE_21 / E_TOTPOP) + (FEMALE_22_TO_24 / E_TOTPOP) + (FEMALE_25_TO_29 / E_TOTPOP) AS FEMALE_18_TO_29, (FEMALE_30_TO_34 / E_TOTPOP) + (FEMALE_35_TO_39 / E_TOTPOP) + (FEMALE_40_TO_44 / E_TOTPOP) + (FEMALE_45_TO_49 / E_TOTPOP) AS FEMALE_30_TO_49, (FEMALE_50_TO_54 / E_TOTPOP) + (FEMALE_55_TO_59 / E_TOTPOP) + (FEMALE_60_TO_61 / E_TOTPOP) + (FEMALE_62_TO_64 / E_TOTPOP) + (FEMALE_65_TO_66 / E_TOTPOP) + (FEMALE_67_TO_69 / E_TOTPOP) + (FEMALE_70_TO_74 / E_TOTPOP) + (FEMALE_75_TO_79 / E_TOTPOP) + (FEMALE_80_TO_84 / E_TOTPOP) + FEMALE_85_AND_OVER / E_TOTPOP AS FEMALE_OVER_50, NOT_US_CITIZEN_POP / E_TOTPOP AS PERCENT_NONCITIZEN, WHITE_POP / E_TOTPOP AS PERCENT_WHITE, BLACK_POP / E_TOTPOP AS PERCENT_BLACK, ASIAN_POP / E_TOTPOP AS PERCENT_ASIAN, HISPANIC_POP / E_TOTPOP AS PERCENT_HISPANIC, AMERINDIAN_POP / E_TOTPOP AS PERCENT_AMERINDIAN, OTHER_RACE_POP / E_TOTPOP AS PERCENT_OTHER_RACE, TWO_OR_MORE_RACES_POP / E_TOTPOP AS PERCENT_MULTIRACIAL, HISPANIC_ANY_RACE / E_TOTPOP AS PERCENT_HISPANIC_ANY_RACE, GINI_INDEX, EP_DISABL, EP_MINRTY, EP_CROWD, EP_GROUPQ, EP_UNINSUR, EP_NOHSDP, CAST(DATE AS STRING) AS ds, CONFIRMED_CASES AS y FROM prod_nyt_covid19.covid_aggregate_output WHERE RPL_THEMES > 0 AND DEATHS > 0"


# I created a filtered version of the DataFrame because I wanted to eliminate all values that were not floats or ints before conducting correlations and regressions.

# In[434]:


# Filtered Dataset

df = client.query(sql_query).to_dataframe()
df_filtered = df.drop(columns=['DATE', 'COUNTY', 'FIPS'])


# I also decided to use MinMaxScaler() to make all variables range from zero to one because not all variables initially did.

# In[435]:


# Normalized Dataset

scaler = MinMaxScaler()
df_filtered[['CONFIRMED_CASES', 'DEATHS', 'CC_PER_CAPITA', 'D_PER_CAPITA', 'D_PER_CC', 'E_TOTPOP', 'DIABETES_RATE', 'SMOKING_RATE', 'MEDIAN_INCOME', 'PERCENT_DEM', 'RPL_THEMES', 'CCT_1', 'CCT_2', 'CCT_3', 'CCT_7', 'CCT_14', 'CCT_21', 'CCT_31', 'DT_1', 'DT_2', 'DT_3', 'DT_7', 'DT_14', 'DT_21', 'DT_31', 'MALE_0_TO_17', 'MALE_18_TO_29', 'MALE_30_TO_49', 'MALE_OVER_50', 'FEMALE_0_TO_17', 'FEMALE_18_TO_29', 'FEMALE_30_TO_49', 'FEMALE_OVER_50', 'PERCENT_NONCITIZEN', 'PERCENT_BLACK', 'PERCENT_ASIAN', 'PERCENT_HISPANIC', 'PERCENT_AMERINDIAN', 'PERCENT_OTHER_RACE', 'PERCENT_MULTIRACIAL', 'PERCENT_HISPANIC_ANY_RACE', 'GINI_INDEX', 'EP_DISABL', 'EP_MINRTY', 'EP_CROWD', 'EP_GROUPQ', 'EP_UNINSUR', 'EP_NOHSDP', 'y']] = scaler.fit_transform(df_filtered[['CONFIRMED_CASES', 'DEATHS', 'CC_PER_CAPITA', 'D_PER_CAPITA', 'D_PER_CC', 'E_TOTPOP', 'DIABETES_RATE', 'SMOKING_RATE', 'MEDIAN_INCOME', 'PERCENT_DEM', 'RPL_THEMES', 'CCT_1', 'CCT_2', 'CCT_3', 'CCT_7', 'CCT_14', 'CCT_21', 'CCT_31', 'DT_1', 'DT_2', 'DT_3', 'DT_7', 'DT_14', 'DT_21', 'DT_31', 'MALE_0_TO_17', 'MALE_18_TO_29', 'MALE_30_TO_49', 'MALE_OVER_50', 'FEMALE_0_TO_17', 'FEMALE_18_TO_29', 'FEMALE_30_TO_49', 'FEMALE_OVER_50', 'PERCENT_NONCITIZEN', 'PERCENT_BLACK', 'PERCENT_ASIAN', 'PERCENT_HISPANIC', 'PERCENT_AMERINDIAN', 'PERCENT_OTHER_RACE', 'PERCENT_MULTIRACIAL', 'PERCENT_HISPANIC_ANY_RACE', 'GINI_INDEX', 'EP_DISABL', 'EP_MINRTY', 'EP_CROWD', 'EP_GROUPQ', 'EP_UNINSUR', 'EP_NOHSDP', 'y']])
df2 = df_filtered


# The following displays a summary of CONFIRMED_CASES (no log) and DEATHS (no log) to demonstrate the aforementioned right skew. The 75th percentiles for both confirmed cases and deaths do not even amount to 1% of the cases and deaths (respectively) that the max counties have. The means are significantly higher than the 50th percentiles for both metrics.

# In[412]:


# COVID-19 Summary

df_ccpc[["CONFIRMED_CASES", "DEATHS"]].describe()


# I regrouped the age metrics into larger classifications to make them more digestible. I made this change after running OLS regressions on the original distributions and saw that not all the groupings were statistically significant. Here is a breakdown of counties' age distributions:

# In[422]:


# Age Summary

df[["MALE_0_TO_17", "FEMALE_0_TO_17", "MALE_18_TO_29", "FEMALE_18_TO_29", "MALE_30_TO_49", "FEMALE_30_TO_49", "MALE_OVER_50", "FEMALE_OVER_50"]].describe()


# Perhaps I could have also tried taking the log of the racial variables (since they almost all have a right skew), but due to limited time, I will save that as something to consider moving forward. Here is the racial breakdown:

# In[432]:


# Race Summary

df[["PERCENT_WHITE", "PERCENT_HISPANIC", "PERCENT_BLACK", "PERCENT_ASIAN", "PERCENT_AMERINDIAN", "PERCENT_OTHER_RACE", "PERCENT_MULTIRACIAL"]].describe()


# Here is another summary of the remaining variables, which also display right skews in most cases (though less significant ones than in the previous tables):

# In[438]:


# Other Summary

df_ccpc[["E_TOTPOP", "EP_PCI", "EP_POV", "EP_CROWD", "EP_DISABL", "EP_GROUPQ", "EP_MINRTY", "EP_NOHSDP", "EP_UNINSUR", "RPL_THEMES", "PERCENT_DEM", "PERCENT_REP", "PERCENT_NONCITIZEN", "DIABETES_RATE", "SMOKING_RATE"]].describe()


# I conducted correlations between CONFIRMED_CASES and DEATHS and all other variables.
# 
# CONFIRMED_CASES and DEATHS have a correlation of 83.5%, a number which was previously ~ 96% (supporting the theory that people are now more prepared to handle COVID-19 cases and prevent deaths).
# 
# E_TOTPOP has a correlation of 67.9% with CONFIRMED_CASES and 61.8% with DEATHS. This high correlation is why it is important to consider CC_PER_CAPITA (which I do later in descriptive modeling).
# 
# EP_DISABL has a correlation of -41.3% with CONFIRMED_CASES and -33.8% with DEATHS. These correlations go against my Data Studio results, which state that in counties with populations over 1,000,000 people, there is a positive correlation between EP_DISABL and death rate. Looking at all counties regardless of population changes this relationship.
# 
# EP_MINRTY has a correlation of 31.9% with CONFIRMED_CASES and 25.4% with DEATHS. Looking at the racial breakdowns, racial percentages for all races except white and Native American are positively correlated with CONFIRMED_CASES and DEATHS. The strongest correlations are between CONFIRMED_CASES/DEATHS and PERCENT_ASIAN (40.2% and 37.8% respectively) and between CONFIRMED_CASES/DEATHS and PERCENT_NONCITIZEN (43.1% and 33.0% respectively). Perhaps a cause for this trend is that PERCENT_ASIAN and PERCENT_NONCITIZEN are higher in more populated counties, which naturally have higher CONFIRMED_CASES and DEATHS.
# 
# According to the table below, PERCENT_DEM is positively correlated with CONFIRMED_CASES and DEATHS, whereas PERCENT_REP is negatively correlated with these variables. In Data Studio, looking at CC_PER_CAPITA revealed that red counties were experiencing higher case counts relative to their populations than blue counties. The reason for this phenomenon might also be attributed to population, which is typically higher in blue counties and thus makes them more prone to cases and deaths.
# 
# RPL_THEMES surprisingly had a low correlation with both variables, directly contrasting the Data Studio results; perhaps there is an unknown error here that requires further investigation.
# 
# Contrary to my original hypothesis, the rates for diabetes and smoking in counties had negative correlations with cases and deaths.
# 
# Looking at the age breakdowns, MALE_OVER_50 and FEMALE_OVER_50 were both negatively correlated with both CONFIRMED_CASES and DEATHS. This result is somewhat surprising because it is known that older people have a higher chance of dying from COVID-19. It is possible that counties higher populations of old people have lower total populations and have not been exposed to the virus as much; also, it could be that counties with smaller populations of old people experience the most elderly deaths because the younger majority spreads the disease to them (maybe older communities are more careful).

# In[443]:


# Correlation

corr_df = df[["CONFIRMED_CASES", "DEATHS", "E_TOTPOP", "MEDIAN_INCOME", "EP_CROWD", "EP_DISABL", "EP_GROUPQ", "EP_MINRTY", "EP_NOHSDP", "EP_UNINSUR", "RPL_THEMES", "PERCENT_DEM", "PERCENT_REP", "DIABETES_RATE", "SMOKING_RATE", "PERCENT_WHITE", "PERCENT_HISPANIC", "PERCENT_BLACK", "PERCENT_ASIAN", "PERCENT_AMERINDIAN", "PERCENT_OTHER_RACE", "PERCENT_MULTIRACIAL", "PERCENT_NONCITIZEN", "MALE_0_TO_17", "FEMALE_0_TO_17", "MALE_18_TO_29", "FEMALE_18_TO_29", "MALE_30_TO_49", "FEMALE_30_TO_49", "MALE_OVER_50", "FEMALE_OVER_50"]].corr()
corr_df[["CONFIRMED_CASES", "DEATHS"]].head(100)


# I also conducted autocorrelations to better understand the relationships between CONFIRMED_CASES and DEATHS over time.
# 
# Cases on a given day are still the most highly correlated with deaths on that day.
# 
# CCT_1 (cases yesterday) are most correlated with CCT_3 and CCT_7 (with correlations of 88.2% and 88.4% respectively).
# 
# Similarly, DT_1 (deaths yesterday) are most correlated with DT_3 and DT_7 (with correlations of 94.2% and 94.1% respectively).
# 
# The autocorrelations for deaths against deaths are stronger than for cases against cases, meaning that more accurate predictions can be made about deaths using death past data than about cases using past case data.

# In[410]:


# Autocorrelation

sql_query_lag  = "select lag(CONFIRMED_CASES,1) over (partition by COUNTY order by DATE asc) as CCT_1, lag(CONFIRMED_CASES,2) over (partition by COUNTY order by DATE asc) as CCT_2, lag(CONFIRMED_CASES,3) over (partition by COUNTY order by DATE asc) as CCT_3, lag(CONFIRMED_CASES,7) over (partition by COUNTY order by DATE) as CCT_7, lag(CONFIRMED_CASES,14) over (partition by COUNTY order by DATE asc) as CCT_14, lag(CONFIRMED_CASES,21) over (partition by COUNTY order by DATE asc) as CCT_21, lag(CONFIRMED_CASES,31) over (partition by COUNTY order by DATE asc) as CCT_31, lag(deaths,1) over (partition by COUNTY order by DATE asc) as DT_1, lag(deaths,2) over (partition by COUNTY order by DATE asc) as DT_2, lag(deaths,3) over (partition by COUNTY order by DATE asc) as DT_3, lag(deaths,7) over (partition by COUNTY order by DATE asc) as DT_7, lag(DEATHS,14) over (partition by COUNTY order by DATE asc) as DT_14, lag(DEATHS,21) over (partition by COUNTY order by DATE asc) as DT_21, lag(DEATHS,31) over (partition by COUNTY order by DATE asc) as DT_31 from prod_nyt_covid19.covid_aggregate_output"
df_lag = client.query(sql_query_lag).to_dataframe()
corr_df_lag = df_lag.corr()
corr_df_lag[["CCT_1", "CCT_2", "CCT_3", "CCT_7", "CCT_14", "CCT_21", "CCT_31", "DT_1", "DT_2", "DT_3", "DT_7","DT_14", "DT_21", "DT_31"]].head(100)


# # Descriptive Modeling

# I did a series of ordinary least squares (OLS) regressions using the following as dependent variables: CONFIRMED_CASES, DEATHS, CC_PER_CAPITA, and error. I chose this regression because it clearly showed me the relationships amongst variables.
# 
# I initially wanted to use the forward selection code below to automatically select the variables that would optimize the model, but the factorial runtime would have had me waiting years for the algorithm to run through the hundreds of input variables in my SQL code. I decided to run the regression on all the variables to begin with and removed the statistically insignificant variables thereafter.

# In[ ]:


# Forward Selection

def forward_selected(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model


# The first regression I did was looking at CONFIRMED_CASES as the dependent variable and had an adjusted R-squared value (R-squared accounting for the number of variables) of 0.769, meaning that 76.9% of the data fits the regression model.
# 
# The skew of the dataset is well between -0.5 and 0.5, meaning that it is essentially symmetric. Previously, before I took the log of the highly skewed variables, this value had a magnitude of ~ 3.
# 
# The largest positive coefficient value was for total population (0.42), which makes sense because holding everything else constant, counties with higher populations will have more exposure to and higher spread of the disease.
# 
# PERCENT_AMERINDIAN in a county has the highest impact on CONFIRMED_CASES out of all the racial demographics (the news mostly talks about black and Hispanic). This result contrasts with my correlations, which suggested that PERCENT_ASIAN was the most highly correlated racial demographic with CONFIRMED_CASES (most likely from omitted variable bias); however, similarly to my correlation results, PERCENT_NONCITIZEN is more impactful than all other race-related variables (with a coefficient of 0.27).
# 
# MEDIAN_INCOME and RPL_THEMES surprisingly did not have much of an effect (the latter is possibly due to an unkown error).
# 
# All age groups in the model had a lower impact than males 30 to 49 (the base) on CONFIRMED_CASES.
# 
# PERCENT_DEM has a very minimal impact on CONFIRMED_CASES (contrary to my observations in Data Studio).

# In[445]:


# OLS Regression (CONFIRMED_CASES)

results_cc = smf.ols(formula = "CONFIRMED_CASES ~ CCT_1 + CCT_2 + CCT_3 + CCT_7 + CCT_14 + CCT_31 + E_TOTPOP + MEDIAN_INCOME + EP_CROWD + EP_DISABL + EP_GROUPQ + EP_NOHSDP + EP_UNINSUR + RPL_THEMES + PERCENT_DEM + DIABETES_RATE + SMOKING_RATE + PERCENT_HISPANIC + PERCENT_BLACK + PERCENT_ASIAN + PERCENT_AMERINDIAN + PERCENT_OTHER_RACE + PERCENT_MULTIRACIAL + PERCENT_NONCITIZEN + MALE_0_TO_17 + FEMALE_0_TO_17 + MALE_18_TO_29 + FEMALE_18_TO_29 + FEMALE_30_TO_49 + MALE_OVER_50 + FEMALE_OVER_50", data=df2).fit() 
print(results_cc.summary())


# I then did a regression looking at DEATHS as the dependent variable and had a slightly lower adjusted R-squared value of 0.647, meaning that 64.7% of the data fits the regression model.
# 
# Variables regarding deaths over time are included in this model even though they were excluded in the previous one (because the dead cannot affect the living).
# 
# Again, the largest positive coefficient value was for total population (0.62), supporting the fact that deaths occur disproportionately in urban areas. Population is the most likely culprit for the discrepancies between my correlations and regressions (caused by omitted variable bias).
# 
# PERCENT_AMERINDIAN in a county also has the highest impact on DEATHS out of all the racial demographics.
# 
# Overall, this regression was very similar to the previous model (since CONFIRMED_CASES and DEATHS are so highly related).

# In[402]:


# OLS Regression (DEATHS)

results_d = smf.ols(formula = "DEATHS ~ CCT_1 + CCT_2 + CCT_3 + CCT_7 + CCT_14 + CCT_31 + DT_1 + DT_2 + DT_3 + DT_7 + DT_14 + DT_31 + E_TOTPOP + MEDIAN_INCOME + EP_CROWD + EP_DISABL + EP_GROUPQ + EP_NOHSDP + EP_UNINSUR + RPL_THEMES + PERCENT_DEM + DIABETES_RATE + SMOKING_RATE + PERCENT_HISPANIC + PERCENT_BLACK + PERCENT_ASIAN + PERCENT_AMERINDIAN + PERCENT_OTHER_RACE + PERCENT_MULTIRACIAL + PERCENT_NONCITIZEN + MALE_0_TO_17 + FEMALE_0_TO_17 + MALE_18_TO_29 + FEMALE_18_TO_29 + FEMALE_30_TO_49 + MALE_OVER_50 + FEMALE_OVER_50", data=df2).fit()
print(results_d.summary())


# In order to eliminate the impacts of population on the model, I used CC_PER_CAPITA as the dependent variable and had an adjusted R-squared value of 0.669, meaning that 66.9% of the data fits the regression model.
# 
# The most impactful variables in the model after eliminating population were cases from previous days. CCT_3 (cases three days ago) is the variable with the highest coefficient value in the equation for determining CC_PER_CAPITA.
# 
# An issue with this model is that it is highly skewed as a result of none of the variables being logged. With more time, I would experiment logging the most highly skewed variables to see how that effects the model.
# 
# In part because the variables describing cases from previous days were the most profound in determining CC_PER_CAPITA, I decided to solely focus on CONFIRMED_CASES over time for my forecasting model (which itself reflects the demographic intricacies of counties).

# In[444]:


# OLS Regression (CC_PER_CAPITA)

sql_query_ccpc = "SELECT DATE, COUNTY, FIPS, CONFIRMED_CASES, DEATHS, CONFIRMED_CASES / E_TOTPOP AS CC_PER_CAPITA, DEATHS / E_TOTPOP AS D_PER_CAPITA, DEATHS / CONFIRMED_CASES AS D_PER_CC, E_TOTPOP, DIABETES_RATE, SMOKING_RATE, ROUND(LN(MEDIAN_INCOME)) AS MEDIAN_INCOME, PERCENT_DEM, PERCENT_REP, RPL_THEMES, lag(CONFIRMED_CASES / E_TOTPOP, 1) over (partition by COUNTY order by DATE asc) as CCT_1, lag(CONFIRMED_CASES / E_TOTPOP, 2) over (partition by COUNTY order by DATE asc) as CCT_2, lag(CONFIRMED_CASES / E_TOTPOP, 3) over (partition by COUNTY order by DATE asc) as CCT_3, lag(CONFIRMED_CASES / E_TOTPOP, 7) over (partition by COUNTY order by DATE) as CCT_7, lag(CONFIRMED_CASES / E_TOTPOP, 14) over (partition by COUNTY order by DATE asc) as CCT_14, lag(CONFIRMED_CASES / E_TOTPOP, 21) over (partition by COUNTY order by DATE asc) as CCT_21, lag(CONFIRMED_CASES / E_TOTPOP, 31) over (partition by COUNTY order by DATE asc) as CCT_31, lag(DEATHS / E_TOTPOP, 1) over (partition by COUNTY order by DATE asc) as DT_1, lag(DEATHS / E_TOTPOP, 2) over (partition by COUNTY order by DATE asc) as DT_2, lag(DEATHS / E_TOTPOP, 3) over (partition by COUNTY order by DATE asc) as DT_3, lag(DEATHS / E_TOTPOP, 7) over (partition by COUNTY order by DATE asc) as DT_7, lag(DEATHS / E_TOTPOP, 14) over (partition by COUNTY order by DATE asc) as DT_14, lag(DEATHS / E_TOTPOP, 21) over (partition by COUNTY order by DATE asc) as DT_21, lag(DEATHS / E_TOTPOP, 31) over (partition by COUNTY order by DATE asc) as DT_31, ((MALE_UNDER_5 / E_TOTPOP) + (MALE_5_TO_9 / E_TOTPOP) + (MALE_10_TO_14 / E_TOTPOP) + (MALE_15_TO_17 / E_TOTPOP)) as MALE_0_TO_17, ((MALE_18_TO_19 / E_TOTPOP) + (MALE_20 / E_TOTPOP) + (male_21 / E_TOTPOP) + (male_22_to_24 / E_TOTPOP) + (male_25_to_29 / E_TOTPOP)) as MALE_18_TO_29, (male_30_to_34 / E_TOTPOP) + (male_35_to_39 / E_TOTPOP) + (male_40_to_44 / E_TOTPOP) + (male_45_to_49 / E_TOTPOP) as MALE_30_TO_49, (male_50_to_54 / E_TOTPOP) + (male_55_to_59 / E_TOTPOP) + (male_60_to_61 / E_TOTPOP) + (male_62_to_64 / E_TOTPOP) + (male_65_to_66 / E_TOTPOP) + (male_67_to_69 / E_TOTPOP) + (male_70_to_74 / E_TOTPOP) + (male_75_to_79 / E_TOTPOP) + (male_80_to_84 / E_TOTPOP) + male_85_and_over / E_TOTPOP as MALE_OVER_50, (female_under_5 / E_TOTPOP) + (female_5_to_9 / E_TOTPOP) + (female_10_to_14 / E_TOTPOP) + (female_15_to_17 / E_TOTPOP) as FEMALE_0_TO_17, (female_18_to_19 / E_TOTPOP) + (female_20 / E_TOTPOP) + (female_21 / E_TOTPOP) + (female_22_to_24 / E_TOTPOP) + (female_25_to_29 / E_TOTPOP) as FEMALE_18_TO_29, (female_30_to_34 / E_TOTPOP) + (female_35_to_39 / E_TOTPOP) + (female_40_to_44 / E_TOTPOP) + (female_45_to_49 / E_TOTPOP) as FEMALE_30_TO_49, (female_50_to_54 / E_TOTPOP) + (female_55_to_59 / E_TOTPOP) + (female_60_to_61 / E_TOTPOP) + (female_62_to_64 / E_TOTPOP) + (female_65_to_66 / E_TOTPOP) + (female_67_to_69 / E_TOTPOP) + (female_70_to_74 / E_TOTPOP) + (female_75_to_79 / E_TOTPOP) + (female_80_to_84 / E_TOTPOP) + female_85_and_over / E_TOTPOP as FEMALE_OVER_50, not_us_citizen_pop / E_TOTPOP as PERCENT_NONCITIZEN, black_pop / E_TOTPOP as PERCENT_BLACK, asian_pop / E_TOTPOP as PERCENT_ASIAN, hispanic_pop / E_TOTPOP as PERCENT_HISPANIC, amerindian_pop / E_TOTPOP as PERCENT_AMERINDIAN, other_race_pop / E_TOTPOP as PERCENT_OTHER_RACE, two_or_more_races_pop / E_TOTPOP as PERCENT_MULTIRACIAL, hispanic_any_race / E_TOTPOP as PERCENT_HISPANIC_ANY_RACE, GINI_INDEX, EP_DISABL, EP_MINRTY, EP_CROWD, EP_GROUPQ, EP_UNINSUR, EP_NOHSDP, EP_PCI, EP_POV, CAST(DATE AS STRING) AS ds, CONFIRMED_CASES AS y FROM prod_nyt_covid19.covid_aggregate_output WHERE RPL_THEMES > 0 AND DEATHS > 0"
df_ccpc = client.query(sql_query_ccpc).to_dataframe()
results_ccpc = smf.ols(formula = "CC_PER_CAPITA ~ CCT_1 + CCT_2 + CCT_3 + CCT_7 + CCT_14 + CCT_31 + MEDIAN_INCOME + EP_CROWD + EP_DISABL + EP_GROUPQ + EP_NOHSDP + EP_UNINSUR + RPL_THEMES + PERCENT_DEM + DIABETES_RATE + SMOKING_RATE + PERCENT_HISPANIC + PERCENT_BLACK + PERCENT_ASIAN + PERCENT_AMERINDIAN + PERCENT_OTHER_RACE + PERCENT_MULTIRACIAL + PERCENT_NONCITIZEN + MALE_0_TO_17 + FEMALE_0_TO_17 + MALE_18_TO_29 + FEMALE_18_TO_29 + FEMALE_30_TO_49 + MALE_OVER_50 + FEMALE_OVER_50", data=df_ccpc).fit() 
print(results_ccpc.summary())


# # Forecasting

# For forecasting, I initially selected a few demographically and geographically diverse counties to focus on. The counties I selected were the following: Prince George's, Miami-Dade, Westchester, Los Angeles, and Bexar.
# 
# First, I had to recursively join COUNTY, ds, and y for every county into one aggregated dataset.
# 
# The training period I used was from whenever the county's first case was to 05/31/20 (since this date approximately marked somewhat of a shift in the communities the virus was affecting). The models each predict two weeks ahead.

# In[ ]:


# Prince George's Join

m_pg = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_pg = df[(df.COUNTY == "Prince George's") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_pg.fit(df_pg)
future_pg = m.make_future_dataframe(periods=14)
train_performance_pg = m_pg.predict(df_pg[["ds"]])
train_performance_pg['ds'] = train_performance_pg['ds'].dt.strftime('%Y-%m-%d')
df_pg_join = pd.merge(df_pg, train_performance_pg, on = 'ds', how = 'left')

# Miami-Dade Join

m_miami = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_miami = df[(df.COUNTY == "Miami-Dade") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_miami.fit(df_miami)
future_miami = m_miami.make_future_dataframe(periods=14)
train_performance_miami = m_miami.predict(df_miami[["ds"]])
train_performance_miami['ds'] = train_performance_miami['ds'].dt.strftime('%Y-%m-%d')
df_miami_join = pd.merge(df_miami, train_performance_miami, on = 'ds', how = 'left')
df_pg_miami = pd.concat([df_pg_join, df_miami_join])

# Westchester Join

m_wc = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_wc = df[(df.COUNTY == "Westchester") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_wc.fit(df_wc)
future_wc = m_wc.make_future_dataframe(periods=14)
train_performance_wc = m_wc.predict(df_wc[["ds"]])
train_performance_wc['ds'] = train_performance_wc['ds'].dt.strftime('%Y-%m-%d')
df_wc_join = pd.merge(df_wc, train_performance_wc, on = 'ds', how = 'left')
df_pg_miami_wc = pd.concat([df_pg_miami, df_wc_join])

# Los Angeles Join

m_la = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_la = df[(df.COUNTY == "Los Angeles") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_la.fit(df_la)
future_la = m_la.make_future_dataframe(periods=14)
train_performance_la = m_la.predict(df_la[["ds"]])
train_performance_la['ds'] = train_performance_la['ds'].dt.strftime('%Y-%m-%d')
df_la_join = pd.merge(df_la, train_performance_la, on = 'ds', how = 'left')
df_pg_miami_wc_la = pd.concat([df_pg_miami_wc, df_la_join])

# Bexar Join

m_bx = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_bx = df[(df.COUNTY == "Bexar") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_bx.fit(df_bx)
future_bx = m_bx.make_future_dataframe(periods=14)
train_performance_bx = m_bx.predict(df_bx[["ds"]])
train_performance_bx['ds'] = train_performance_bx['ds'].dt.strftime('%Y-%m-%d')
df_bx_join = pd.merge(df_bx, train_performance_bx, on = 'ds', how = 'left')
df_pg_miami_wc_la_bx = pd.concat([df_pg_miami_wc_la, df_bx_join])

""" # Wayne

m_wy = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_wy = df[(df.COUNTY == "Wayne") & (df.FIPS == "26163") & (df.DIABETES_RATE == 10.6) & (df.ds <= '2020-05-31')][["COUNTY", "FIPS", "ds", "y"]]
m_wy.fit(df_wy)
future_bx = m_bx.make_future_dataframe(periods=14)
train_performance_wy = m_wy.predict(df_bx[["ds"]])
train_performance_wy['ds'] = train_performance_wy['ds'].dt.strftime('%Y-%m-%d')
df_wy_join = pd.merge(df_wy, train_performance_wy, on = 'ds', how = 'left')
df_pg_miami_wc_la_bx_wy = pd.concat([df_pg_miami_wc_la_bx, df_wy_join]) """

m = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
m.fit(df_pg_miami_wc_la_bx_wy)
future = m.make_future_dataframe(periods=14)
train_performance = m.predict(df_pg_miami_wc_la_bx_wy[["ds"]])
train_performance['ds'] = train_performance['ds'].dt.strftime('%Y-%m-%d')


# I then combined the prior aggregated dataset with all my demographic data, adding an error column to keep track of actual - predicted values for cases.
# 
# I calculated the root mean square error (RMSE) for each county to gain a sense of the prediction accuracy of each model. Los Angeles had the largest RMSE value (317.48), which makes sense because cases are only starting to pick up now in that region (when they surprisingly did not in March and April).

# In[404]:


# Aggregate Forecasting Dataset for Selected Counties

df_agg = pd.merge(df_pg_miami_wc_la_bx_wy, df, on = ['COUNTY', 'FIPS', 'ds'], how = 'left')
df_agg["error"] = df_agg.apply(lambda x: x["y_y"] - x["yhat"], axis=1)

# Root Mean Square Error for Selected Counties

print("Root Mean Square Error (RMSE) by County")
print()

rmse_pg = math.sqrt(sum((df_pg_join.y - train_performance_pg.yhat)**2) / len(df_pg_join.index))
print("Prince George's:", rmse_pg)

rmse_miami = math.sqrt(sum((df_miami_join.y - train_performance_miami.yhat)**2) / len(df_miami_join.index))
print("Miami-Dade:", rmse_miami)

rmse_wc = math.sqrt(sum((df_wc_join.y - train_performance_wc.yhat)**2) / len(df_wc_join.index))
print("Westchester:", rmse_wc)

rmse_la = math.sqrt(sum((df_la_join.y - train_performance_la.yhat)**2) / len(df_la_join.index))
print("Los Angeles:", rmse_la)

rmse_bx = math.sqrt(sum((df_bx_join.y - train_performance_bx.yhat)**2) / len(df_bx_join.index))
print("Bexar:", rmse_bx)


# I tried to run an OLS regression on the error values, but because the static data does not change on the level of individual counties, the model was useless, comparing dynamic COVID-19 data with county data that remained constant despite the changes in CONFIRMED_CASES or DEATHS.

# In[446]:


# OLS Regression (error)

results_fc = smf.ols(formula = "error ~ E_TOTPOP + MEDIAN_INCOME + EP_CROWD + EP_DISABL + EP_GROUPQ + EP_NOHSDP + EP_UNINSUR + RPL_THEMES + PERCENT_DEM + DIABETES_RATE + SMOKING_RATE + PERCENT_HISPANIC + PERCENT_BLACK + PERCENT_ASIAN + PERCENT_AMERINDIAN + PERCENT_OTHER_RACE + PERCENT_MULTIRACIAL + PERCENT_NONCITIZEN + MALE_0_TO_17 + FEMALE_0_TO_17 + MALE_18_TO_29 + FEMALE_18_TO_29 + FEMALE_30_TO_49 + MALE_OVER_50 + FEMALE_OVER_50", data=df_agg).fit()
print(results_fc.summary())


# I then created visualizations of each model's predictions. In the plots, the black points represent actual data, the red line represents the trendline based on that data, and the blue space represents the uncertainty of the prediction.
# 
# Interestingly, many of the counties displayed have cases reaching a minimum on Wednesdays.
# 
# There seem to be different archetypes of counties: generally, those whose plots are concave and those whose are convex. Westchester, for example, is concave while Prince George's and Los Angeles are both convex.
# 
# It would be interesting to try to approximate the trendlines on these models to functions and to subsequently take their derivatives to determine the functions' concavities, which could be an interesting way to forecast.

# In[405]:


# Prince George's Forecast

m_pg = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_pg = df[(df.COUNTY == "Prince George's") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_pg.fit(df_pg)
future_pg = m_pg.make_future_dataframe(periods=14)
test_performance_pg = m_pg.predict(future_pg)
fig_pg = m_pg.plot(test_performance_pg)
a_pg = add_changepoints_to_plot(fig_pg.gca(), m_pg, test_performance_pg)
plt.title("Prince George's Confirmed Cases")
fig_pg_2 = m_pg.plot_components(test_performance_pg)


# In[368]:


# Miami-Dade Forecast

m_miami = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_miami = df[(df.COUNTY == "Miami-Dade") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_miami.fit(df_miami)
future_miami = m_miami.make_future_dataframe(periods=14)
test_performance_miami = m_miami.predict(future_miami)
fig_miami = m_miami.plot(test_performance_miami)
a_miami = add_changepoints_to_plot(fig_miami.gca(), m_miami, test_performance_miami)
plt.title("Miami-Dade's Confirmed Cases")
fig_miami_2 = m_miami.plot_components(test_performance_miami)


# In[369]:


# Westchester Forecast

m_wc = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_wc = df[(df.COUNTY == "Westchester") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_wc.fit(df_wc)
future_wc = m_wc.make_future_dataframe(periods=14)
test_performance_wc = m_wc.predict(future_wc)
fig_wc = m_wc.plot(test_performance_wc)
a_wc = add_changepoints_to_plot(fig_wc.gca(), m_wc, test_performance_wc)
plt.title("Westchester's Confirmed Cases")
fig_wc_2 = m_wc.plot_components(test_performance_wc)


# In[370]:


# Los Angeles Forecast

m_la = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_la = df[(df.COUNTY == "Los Angeles") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_la.fit(df_la)
future_la = m_la.make_future_dataframe(periods=14)
test_performance_la = m_la.predict(future_la)
fig_la = m_la.plot(test_performance_la)
a_la = add_changepoints_to_plot(fig_la.gca(), m_la, test_performance_la)
plt.title("Los Angeles' Confirmed Cases")
fig_la_2 = m_la.plot_components(test_performance_la)


# In[372]:


# Bexar Forecast

m_bx = Prophet(changepoint_prior_scale = 0.5, interval_width = 0.95)
df_bx = df[(df.COUNTY == "Bexar") & (df.ds <= '2020-05-31')][["COUNTY", "ds", "y"]]
m_bx.fit(df_bx)
future_bx = m_bx.make_future_dataframe(periods=14)
test_performance_bx = m_bx.predict(future_bx)
fig_bx = m_bx.plot(test_performance_bx)
a_bx = add_changepoints_to_plot(fig_bx.gca(), m_bx, test_performance_bx)
plt.title("Bexar's Confirmed Cases")
fig_bx_2 = m_bx.plot_components(test_performance_bx)

