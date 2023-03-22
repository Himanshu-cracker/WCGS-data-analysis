#!/usr/bin/env python
# coding: utf-8

# #                                               ANALYSIS OF WCGS DATA SET 

# 
# Q1-
# Create a frequency distribution characterising the age of the sample and summarise the measures of central tendency. Describe the results emerging from the analysis.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.offline import init_notebook_mode
import plotly.io as pio
pio.renderers.default = "notebook_connected"


# In[2]:


data=pd.read_excel("E:\DATA\wcgs.xls")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()/data.shape[0]


# In[7]:


data.dropna(inplace=True)


# In[8]:


data.isnull().sum()/data.shape[0]


# In[9]:


data.columns


# In[10]:


ages=data["age"]


# In[11]:


age_series=pd.Series(ages)


# In[12]:


freq_dist_age = age_series.value_counts()


# In[13]:


freq_dist_age


# # FREQUENCY DISTRIBUSTION OF AGE-

# In[14]:


plt.bar(freq_dist_age.index, freq_dist_age.values)

plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title("Frequency distribution of Age")
plt.show()


# In[15]:


sns.distplot(x=data['age'],kde=True,color='black',bins=20)


# In[16]:


data['age'].describe()


# In[17]:


import statistics
median=statistics.median(data['age'])
mode=statistics.mode(data['age'])
print("Median:", median)
print("Mode:", mode)


# ANALYSIS-
# We can infer from the above figure that the age frequency distribution is not regularly distributed(normal distribution). Also, we can deduce that the data points are right-skewed and that the majority of them are scattered on the left side.Thus, the mode(40) is less than both mean(46) and median(45), and the mean(46) is slightly greater than the median(45). Positively skewed data, sometimes referred to as right-skewed data, is a sort of distribution in which the majority of the data is concentrated on the left side of the distribution while the tail of the data extends to the right. This indicates that the left side of the distribution has a higher concentration of values while the right side has a lower concentration of data points. 

# Q2-
# Does the variable Systolic Blood Pressure (sbp) represent the characteristics of a normally 
# distributed variable? If yes, please justify how do you arrive at this conclusion and if No, 
# how it can be transformed to a normally distributed variable.

#  To check sbp is normally distributed-
#  1-boxplot
#  2-histogram 
#  3-density plots 

# In[18]:


import statsmodels.api as sm


# In[19]:


sbp=data['sbp']


# In[20]:


sbp


# In[21]:


sns.histplot(sbp)


# In[22]:


sns.boxplot(x=sbp)


# In[23]:


sns.distplot(x=data['sbp'],kde=True,color='black',bins=20)


# ANALYSIS-
# We may infer that the data points are not normally distributed from the three charts above. We must perform the Shapiro (p value) test in order to further examine the distribution.

# In[24]:


from scipy.stats import shapiro

stat, p = shapiro(sbp)

print('Test statistic:', stat)
print('P-value:', p)

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# ANALYSIS-
# We can infer from the preceding finding that the data points are not regularly distributed (normally distributed) and that the p value is less than 0.05.

# four methods to make data points normally distributed-. 
# 1-log transformation
# 2-square root transformation
# 3-z value transformation
# 4-box-cox transformation

# # Log transformation-

# In[25]:


log_sbp = np.log(sbp)


# In[26]:


from scipy.stats import shapiro     ### (for testing normal distribution)

stat, p = shapiro(log_sbp)

print('Test statistic:', stat)
print('P-value:' + str(p))

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# # square root-

# In[27]:


sbp_sqrt = np.sqrt(sbp)


# In[28]:


from scipy.stats import shapiro   ### (for testing normal distribution)

stat, p = shapiro(sbp_sqrt)

print('Test statistic:', stat)
print('P-value:', p)

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# # Z transformation-

# In[29]:


from scipy import stats


# In[30]:


z_sbp = stats.zscore(sbp)


# In[31]:


z_sbp


# In[32]:


from scipy.stats import shapiro   ### (for testing normal distribution)

stat, p = shapiro(z_sbp)

print('Test statistic:', stat)
print('P-value:', p)

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# # Box-cox transformation-

# In[33]:


sbp_boxcox,lambda_boxcox = stats.boxcox(sbp)


# In[34]:


from scipy.stats import shapiro    ### (for testing normal distribution)

stat, p = shapiro(sbp_boxcox)

print('Test statistic:', stat)
print('P-value:',p)

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# ANALYSIS-
# The four modifications all imply that the data points are not distributed regularly(normally distributed).
# That is primarily due to the fact that the data contains outliers. In order to get a normal distribution, we must deal with the outliers. Check to see if it is normal by substituting lower and upper whiskers for the lower and upper outliers, respectively.
# 

# In[35]:


'''treating outliers '''
'''
upper outliers treatment - replacing with upper whisker
'''
for x in ['sbp']:
    b = plt.boxplot(data[x])
    values = [item.get_ydata()[1] for item in b['whiskers']]
    upper_whisker = values[1]
    data[x][data[x] > upper_whisker] = upper_whisker

'''
lower outliers treatment - replacing with lower whisker
'''

for x in ['sbp']:
    b = plt.boxplot(data[x])
    values = [item.get_ydata()[1] for item in b['whiskers']]
    lower_whisker = values[0]
    data[x][data[x] < lower_whisker] = lower_whisker


# In[36]:


data.head()


# In[37]:


data.describe()


# In[38]:


sns.boxplot(x=data['sbp'])


# In[39]:


from scipy.stats import shapiro   ### (testing for normal distribution)

stat, p = shapiro(data['sbp'])

print('Test statistic:', stat)
print('P-value:',p)

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# -Taking log of the new sbp values ( after removing outliers) and checking whether it is normally distributed or not.

# In[40]:


log_newsbp=np.log(data['sbp'])


# In[41]:


from scipy.stats import shapiro      ### (testing for normal distribution)

stat, p = shapiro(log_newsbp)

print('Test statistic:', stat)
print('P-value:',p)

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# In[42]:


log_sbp_boxcox,log_lambda_boxcox = stats.boxcox(log_newsbp)


# In[43]:


from scipy.stats import shapiro             ### (testing for normal distribution)

stat, p = shapiro(log_sbp_boxcox)

print('Test statistic:', stat)
print('P-value:',p)

if p > 0.05:
    print('The data is normally distributed')
else:
    print('The data is not normally distributed')


# ANALYSIS-
# After removing outliers, i took log and then transformed it but results are showing - the data is not normally distributed.
# 

# Q3-
# It is assumed that Systolic Blood Pressure (sbp) varies positively with the Body Mass Index (bmi). Examine this hypothesis and create a graph to display the relationship.

# In[44]:


df=pd.read_excel("E:\DATA\wcgs.xls")


# In[45]:


import numpy as np
from scipy.stats import pearsonr

corr, p_val = pearsonr(df['bmi'], df['sbp'])

print("Correlation coefficient:", round(corr,3))
print("p-value:", round(p_val,3))

if p_val < 0.05:
    print("The correlation is significant")
else:
    print("The correlation is not significant")


# In[46]:


sns.scatterplot(data=df, x='sbp',y='bmi',alpha=0.6,size=40)


# In[47]:


sns.regplot(data=df, x='sbp',y='bmi')


# ANALYSIS-
# 
# Null Hypothesis      :The correlation between bmi and sbp is not positive.
# Alternate Hypothesis :The correlation between bmi and sbp is positive.
# From p value ,we can see that null hypothesis is false . Thus there is positive correlation between bmi and sbp .
# visually it can be seen from regression plot that slop is positive.

# Q4-
# Draw box-plots for the variables Systolic Blood Pressure (sbp) and Diastolic Blood 
# Pressure (dbp). From these graphs what do you conclude about the following: (i) Location 
# and Spread, (ii) Interquartile Range, and (iii) Range of Observations and outliers.

# # BOX PLOT FOR sbp

# In[48]:


sns.boxplot(x=data['sbp'])


# In[49]:


location = np.median(data['sbp'])
spread = np.std(data['sbp'])
q1, q3 = np.percentile(data['sbp'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_sbp = data[(data['sbp'] < lower_bound) | (data['sbp'] > upper_bound)]
range_of_observations = np.ptp(data['sbp'])


# In[50]:


print("iqr=",iqr,' ',"spread=",spread,' ',"location=", location)


# IQR(Inter-Quartile Range): It is a measure of the spread of the middle 50% of observations. In this case, the IQR is 16.0.
# 
# Spread: This is a measure of how spread out the data is. standard deviation is a common measure of spread. Here, std. dev. is 15.0502
# 
# Location: This refers to the central tendency of the data, which is where the data tends to cluster. In this case, the location is 126.0, which is the median of the data variable(sbp)

# In[51]:


print("range_of_observation=",range_of_observations,' ',"upper_bound=",upper_bound ,' ',"lower_bound=", lower_bound)


# Range of Observation:It is the difference between the maximum and minimum values of the observations. In this case, the range of observations for systolic blood pressure is 132.0.
# 
# Upper Bound:This is the highest value in a specified range for a variable. In this case, the upper bound is 160.0, which means that any systolic blood pressure value above 160.0 would be considered an outlier or beyond the upper limit.
# 
# Lower Bound:This is the lowest value in a specified range for a variable. In this case, the lower bound is 96.0, which means that any systolic blood pressure value below 96.0 would be considered an outlier or beyond the lower limit.

# In[52]:


print(outliers_sbp)


# Systolic blood pressure should not be higher than 120 mm/hg (115-125 mm/hg).
# Systolic blood pressure has upper and lower bounds of 160mm/Hg and 96mm/Hg, respectively. Any sbp values outside of these ranges will be classified as outliers.

# # BOX PLOT FOR dbp

# In[53]:


import plotly.express as px
fig = px.box(data, x=(data['sbp']), points="all")
fig.show()


# In[54]:


location = np.median(data['dbp'])
spread = np.std(data['dbp'])
q1, q3 = np.percentile(data['dbp'], [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers_dbp = data[(data['dbp'] < lower_bound) | (data['dbp'] > upper_bound)]
range_of_observations = np.ptp(data['dbp'])


# In[55]:


print("iqr=",iqr,' ',"spread=",spread,' ',"location=", location)


# In[56]:


print("range_of_observation=",range_of_observations,' ',"upper_bound=",upper_bound ,' ',"lower_bound=", lower_bound)


# IQR(Inter-Quartile Range)_dbp: It is a measure of the spread of the middle 50% of observations. In this case, the IQR is 10.0.
# 
# Spread_dbp: This is a measure of how spread out the data is. standard deviation is a common measure of spread. Here, std. dev. is 9.635
# 
# Location_dbp: This refers to the central tendency of the data, which is where the data tends to cluster. In this case, the location is 80.0, which is the median of the data variable(dbp)
# 
# Range of Observation_dbp:It is the difference between the maximum and minimum values of the observations. In this case, the range of observations for systolic blood pressure is 78.0
# 
# Upper Bound_dbp:This is the highest value in a specified range for a variable. In this case, the upper bound is 101.0, which means that any systolic blood pressure value above 160.0 would be considered an outlier or beyond the upper limit
# 
# Lower Bound_dbp:This is the lowest value in a specified range for a variable. In this case, the lower bound is 61.0, which means that any diastolic blood pressure value below 61.0 would be considered an outlier or beyond the lower limit.

# In[57]:


print(outliers_dbp)


# Systolic blood pressure should not be higher than 80 mm/hg (75-85 mm/hg). Any diastolic blood pressure results that are greater than 101mm/hg or lower than 61mm/hg will be considered outliers. The upper and lower bounds of diastolic blood pressure are 101mm/hg and 61mm/hg, respectively.

# Q5-
# The data consists of a categorical variable Current Smoking (smoke) that is coded in two categories (Yes and No). Create a box-plots for Total Cholesterol (chol), Diastolic Blood Pressure (dbp) and Systolic Blood Pressure (sbp) by the categorical variable. Explain the results that emerge from the analysis.

# In[58]:


plotly.offline.init_notebook_mode()


# In[59]:


variables = ['dbp', 'sbp', 'chol']
for var in variables:
    fig = px.box(data, x='smoke', y=var, points="all", title=f"{var.capitalize()} by smoking status")
    fig.update_layout(xaxis_title="Smoking status", yaxis_title=f"{var.capitalize()}")
    fig.show()


# Smoke and DBP
# 
# The maximum and lower bounds for the categorical variable "smoking" on a DBP plot are 100 and 62 mm/hg for smokers, respectively, while they are 106 and 60 mm/hg for non-smokers. The median blood pressure for people who smoke and those who don't is 80 mm/hg. We can therefore conclude that smoking has some impact on a person's dbp. The dbp of smokers is lower than that of non-smokers.

# BP and smoke
# 
# The range of the SBP for smokers and non-smokers is between 100 and 160, with outliers going above 160, according to the plotting of the categorical variable "smoking" with SBP. The median SBP value is 126 for both smokers and non-smokers. Hence, we may say that smoking has little to no impact on a person's dbp.

# SMOKE & CHOL
# 
# Smokers' chol varies from 123 to 339 mg/dl, while non-smokers' ranges are 113-330 mg/dl, according to a plot of the categorical variable "smoking" with CHOL. The median cholesterol level for smokers is 228 mg/dl, while the median cholesterol level for non-smokers is 219 mg/dl. So, we can infer that smoking causes the person's cholesterol level to rise.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# 
# 
# 
# 
# 
# 
# Q6-
# It is hypothesised that smoking has a positive and linear impact on both Diastolic Blood Pressure (dbp) and Systolic Blood Pressure (sbp). How will you test this hypothesis? What do the results emerging from the test suggest about the nature of the relationship in the wcgs dataset? Hint: Thinks about the variable ncigs in the dataset that depicts the number of cigarettes smoked per day.

# CONDUCTING LINEAR REGRESSION-

# In[60]:


import statsmodels.api as sm


# In[61]:


dbp_model = sm.OLS(data['dbp'], sm.add_constant(data['ncigs'])).fit()
print(dbp_model.summary())


# RESULTS- For a unit increase in ncigs, dbp reduces by a value of -0.0417.

# ANALYSIS-
# A statistically significant inverse link between smoking (as assessed by the number of cigarettes smoked per day) and diastolic blood pressure can be seen when the coefficient for ncigs is negative and the p-value is significant (less than 0.05). (dbp). In other words, dbp falls as daily cigarette consumption increases.

# In[62]:


sbp_model = sm.OLS(data['sbp'], sm.add_constant(data['ncigs'])).fit()
print(sbp_model.summary())


# RESULT- For a unit increase in ncigs, sbp increase by a value of 0.0304

# Smoking (as defined by the number of cigarettes smoked per day) and diastolic blood pressure (dbp) are not statistically associated with one another in the sample, according to the coefficient for ncigs, but the p-value is greater than 0.05, which suggests there is no statistically significant association between the two variables.

#  ANALYSIS-here we are accepting null hypothesis and rejecting the alternate hypothesis.
