# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 20:00:15 2019

@author: Kathryn
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sb
import matplotlib.pyplot as plt 

#Set Pandas to show all columns in a dataframe
pd.set_option('display.max_columns', None)
#Set Pandas to show all rows in a dataframe
pd.set_option('display.max_rows', None)

nesarc = pd.read_csv('NESARC Data.csv', low_memory=False)


#To avoid runtime errors
pd.set_option('display.float_format', lambda x:'%f'%x)

#Is major depression associated with smoking?
#Presence/absence of major depression is the explanatory variable (categorical)
#Number of cigarettes smoked is the response variable (quantitative)

nesarc['S3AQ3B1']=pd.to_numeric(nesarc['S3AQ3B1'])
nesarc['S3AQ3C1']=pd.to_numeric(nesarc['S3AQ3C1'])
nesarc['CHECK321']=pd.to_numeric(nesarc['CHECK321'])

sub1 = nesarc[(nesarc["AGE"]>=18)&(nesarc["AGE"]<=25)&(nesarc["CHECK321"]==1)]

sub2=sub1.copy()

#replace appropriate values to missing
sub2['S3AQ3B1']=sub2['S3AQ3B1'].replace(9, np.nan)
sub2['S3AQ3C1']=sub2['S3AQ3C1'].replace(99, np.nan)

#recode
recode1 = {1:30, 2:22, 3:14, 4:5, 5:2.5, 6:1}
sub2["USFREQMO"]=sub2["S3AQ3B1"].map(recode1)

sub2["USFREQMO"]=pd.to_numeric(sub2["USFREQMO"])

#Create a secondary variable with math
sub2["NUMCIGMO_EST"]=sub2["USFREQMO"]*sub2["S3AQ3C1"]
sub2["NUMCIGMO_EST"]=pd.to_numeric(sub2["NUMCIGMO_EST"])

#Use ordinary lease squared (OLS) function to calculate P-value
#Test differences in mean number of cigarettes smoked in the last month
#Explanatory variable goes first, response variable second.
#The C() indicates that the response variable is categorical.
model1 = smf.ols(formula="NUMCIGMO_EST ~ C(MAJORDEPLIFE)", data=sub2)
results1=model1.fit()
print(results1.summary()) #Will giveyou quite a few statistics
#Warning message here tells us that standard error estimates are valid as 
#long as underlying assumptions about the errors in the OLS regression are met.

sub3 = sub2[["NUMCIGMO_EST", "MAJORDEPLIFE"]].dropna()

print("means for numcigmo_est by major depression")
m1=sub3.groupby("MAJORDEPLIFE").mean()
print(m1)

print("standard dev for numcigmo_est by major depression")
sd1=sub3.groupby("MAJORDEPLIFE").std()
print(sd1)

sub4=sub2[["NUMCIGMO_EST", "ETHRACE2A"]].dropna()

model2=smf.ols(formula="NUMCIGMO_EST ~ C(ETHRACE2A)", data=sub4).fit()
print(model2.summary())

print("means for numcigmo_est by ethnicity")
m2=sub4.groupby("ETHRACE2A").mean()
print(m2)

print("standard dev for numcigmo_est by ethnicity")
sd2=sub4.groupby("ETHRACE2A").std()
print(sd2)
#The previous tests doesn't tell you which populations aren't equal, just that
#there are significant differences in some of them.

#A significant ANOVA doesn't tell which groups are different from the others.
#To determine which groups are different, you need to perform a post hoc test.
#Type 1 error is incorrectly rejecting the null hypothesis.
#Why can't you perform multiple ANOVAs to compare everything one at a time?
#If you perform multiple tests, the overall chance that you perform a type 1 error
#increases.
#There are lots of different types of post hoc tests to use...don't worry about that.
#The important point right now is that you do one.
#We're going to use the Tukey Honestly Significant Difference Test
import statsmodels.stats.multicomp as multi

mc1 = multi.MultiComparison(sub4["NUMCIGMO_EST"], sub4["ETHRACE2A"])
res1 = mc1.tukeyhsd() #Request the test
print(res1.summary())

#ANOVA was categorical explanatory and quantitative response variable.
#Chi-squared test of independence is categorical explanatory and categorical 
#response variable.
#To do chi-squared test, you calculate the numbers you would expect to see if
#both categorical variables were truly independent (or null hypothesis is true)
#We have observed counts and expected counts. We base the decision on if the 
#the null hypothesis is true by comparing those counts.

#If probability of A and B are independent,
#P(A and B)=P(A)*P(B)
#Expected Count = (Column Total*Row Total)/Table Total
#Chi Squared = sum((Observed Count-Expected Count)^2/Expected Count)
#^This is summed for all cells in the table
#The chi-square number tells us how far away the number that is observed is 
#from the number that is expected.
#For a 2x2 table, the Chi-square is large if it's >3.84
#If your number is less than that, then you can't reject the null hypothesis
#You can then check the p-value for your conclusion to see if it suggests a signficant difference
#The p-value for the chi-square test of independence is the probability
#of getting counts like those observed, assuming that the two variables
#are not related.

#Are rates of nicotine dependence equal or not equal for different smoking categories?
#Explanatory variable is smoking frequency (categorical)
#Response variable is categorical with two levels--presence or absence of nicotine dependence
import scipy.stats as sst

ct1=pd.crosstab(sub2["TAB12MDX"], sub2["USFREQMO"]) #categorical variables
print(ct1) #get counts

#column percentages
colsum=ct1.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct=ct1/colsum
print(colpct) #percent of individuals with and without nicotine dependence at each level

print("chi-square value, p value, expected counts")
cs1=sst.chi2_contingency(ct1)
print(cs1) #shows chi-square value, p-value, expected counts

#If output is set with explanatory categories at the top and response categories
#down the side, you want to determine the column percent.

sub2["USFREQMO"]=sub2["USFREQMO"].astype("category")
sub2["TAB12MDX"]=pd.to_numeric(sub2["TAB12MDX"])

sb.factorplot(x="USFREQMO", y="TAB12MDX", data=sub2, kind="bar", ci=None)
plt.xlabel("Days smoked per month")
plt.ylabel("Proportion nicotine dependent")

#If explanatory variable has more than two levels, chi-square and p-value
#do not provide insight as to why the null hypothesis can be rejected.
#If you reject the null hypothesis, you need to perform comparisons between
#each group
#Use Bonferroni adjustment to protect against type 1 error. It controls
#against family-wise error rate/maximum overall type 1 erorr-rate
#Adjusted p-value is calculated by dividing p=.05 by the # of comparisons.
#Only reject the null hypothesis is the adjusted pvalue is less than p/#comparisons
#need to run a chi-square test for each 15 comparisons
#To do this, you need to select only two groups at a time
#Worth noting: Bonferroni adjustment p-value is applied to all the subsequent p-values.


recode2={1:1, 2.5:2.5} #keeping 1 and 2.5 value as is, but exclude other values in variable
sub2['COMP1v2']=sub2['USFREQMO'].map(recode2)

ct2=pd.crosstab(sub2['TAB12MDX'], sub2['COMP1v2'])
print(ct2)

colsum=ct2.sum(axis=0)
colpct=ct2/colsum
print(colpct)

print('chi-square value, p value, expcted counts')
cs2=sst.chi2_contingency(ct2)
print(cs2) #see that the p-value isn't where it should be

#you keep repeating this for each grouping


#Pearson Correlation, Week 3
#Looking at association between two quantitative variables (Pearson Correlation)
#Positive relationship is increase in one variable goes with increase in other
#Negative relationship means that increase in one variable associated with decrease in other
#Form of relationship is identifying the shape. Most common are:
    #Linear - scattered around a line
    #Curvalinear - points distributed around the same curved line
#r, the correlation coefficient, measures the linear relationship. It is the numerical
    #measure of a linear relationship between two quantitative variables
    #Ranges from -1 to +1. A negative is a negative relationship. Closer to zero, the weaker the relationship.
#Strength of the relationship is determined by how closely dataset follows the form

data_clean=data.dropna() #Need to do this because correlation coefficient cannot be calculated in presence of nas
print("Association between urbanrate and internetuserate")
print(sst.pearsonr(data_clean["urbanrate", data_clean["internetuserate"]]))

print("Association between incomeperperson and internetuserate")
print(sst.pearsonr(data_clean["incomeperperson", data_clean["internetuserate"]]))
#Correlation coefficient and p-value will be generated
#Post-hoc tests are not necessary. They are only necessary for categorical variables with more than two levels
#If we square the correlation coefficient, we get a value that helps our understanding between two quantaitive variables
#r^2 - the fraction of the variability of one variable that can be predicted by the other
    #The higher the better.

#Statistical interaction describes a relationship between two variables
#that is dependent upon a third variable. 
    #The third variable is the moderating variable. It determines the direction or
    #strength of the explanatory and response variable.

modelname=smf.ols(formula="QUANT_RESPONSE~C(CAT_EXPLANATORY)", data=dataframe.fit())
print(modelname.summary)
#Can get an association, but need to understand it.
#Use groupby function to get this
sub1=data[["Weightloss", "Diet"]].dropna()

print("means for Weightloss by Diet A vs. B")
m1=sub1.grouby("Diet").mean()
print(m1)

print("standard deviation for mean Weightloss by Diet A vs. B")
st1=sub1.groupby("Diet").std()
print(st1)

sb.factorplot(x="Diet", y="WeightLoss", data=data, kind="bar", ci=None)
plt.xlabel("Diet Type")
ply.ylabel("Mean Weight Loss in lbs")
#So, what about the 3rd variable?

#The stanard way to address a third variable is to move to a two-way analysis of variance
#instead of the one-way ANOVA
#We're going to ask if our explanatory variable is associated with our response
#variable for each population sub-group or each level of our third variable?
#To do this, we're running two separate ANOVAs for each level of a third variable

sub2=data[(data["Exercise"]=="Cardio")]
sub3=data[(data["Exercise"]=="Weights")]

print("association between diet and weight loss for those using Cardio")
model2 = smf.ols(formula="WeihtLoss~C(Diet)", data=sub2).fit()
print(model2.summary())

print("association between diet and weightloss for those using weights")
model3=smf.ols(formula="WeightLoss~C(Diet)", data=sub3).fit()
print(model3.summary())

print("means for WeightLoss by Diet A vs B for Cardio")
m3=sub2.groupby("Diet").mean()
print(m3) #Look at which diet has more effect for cardio

print("means for WeightLoss by Diet A vs B for weights")
m4=sub3.groupby("Diet").mean()
print(m4)

#Now looking at 3rd variables as moderators in chi-sequare test
#Is smoking associated with nicotine dependence? 
#Making secondary variable for how many cigarettes smoked per day?

recode2{1:30, 2:22, 3:14, 4:5, 5:2.5, 6:1}
sub2["USFREQMO"]=sub2["S3AQ3B1"].map(recode2)

def USQUAN(row):
    if row["S3AQ3B1"] != 1:
        return 0 #non daily smoker
    elif row["S3AQ3C1"] <=5:
        return 3 #range of cigarettes per day for the rest
    elif row["S3AQ3C1"] <=10:
        return 8
    elif row["S3AQ3C1"] <=15:
        return 13
    elif row["S3AQ3C1"] <=20:
        return 18
    elif row["S3AQ3C1"] >20:
        return 37

sub2["USQUAN"]=sub2.apply(lambda row: USQUAN(row), axis=1)

c5=sub2["USQUAN"].value_counts(sort=False, dropna=True)
print(c5)

#now request a chi-square test of independence
#contingency table of observed counts
ct1=pd.crosstab(sub2["TAB12MDX"], sub2["USQUAN"])
print(ct1)

#column percentages
colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

#chi-square
print("chi-square value, p value, expected counts")
cs1=sst.chi2_contingency(ct1)
print(cs1)

#To graph these proportions...
sub2["USQUAN"]=sub2["USQUAN"].astype("category")
sub2["TAB12MDX"]=pd.to_numeric(sub2["TAB12MDX"])

sb.factorplot(x="USQUAN", y="TAB12MDX", data=sub2, kind="bar", ci=None)
plt.xlabel("number of cigarettes smoked per day")
plt.ylabel("Proportion Nictoine Dependent")

#Could a third variable affect this? Going to evaluate major depressive disorder?
#Will evaluate if major depression affects either the strength or direction between smoking and nicotine dependence?

sub3=sub2[(sub2["MAJORDEPLIFE"]==0)]
sub4=sub2[(sub2["MAJORDEPLIFE"]==1)]

print("association between smoking quantity and nicotine dependence for those w/o depression")
ct2=pd.crosstab(sub3["TAB12MDX"], sub3["USQUAN"])
print(ct2)

colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

print("chi-square value, pvalue, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

print("association between smoking quantity and nicotine dependence for those w/ depression")
ct3=pd.crosstab(sub4["TAB12MDX"], sub4["USQUAN"])
print(ct3)

colsum=ct1.sum(axis=0)
colpct=ct1/colsum
print(colpct)

print("chi-square value, pvalue, expected counts")
cs3=sst.chi2_contingency(ct3)
print(cs3)

#To use a line graph to look at this relationship...
sub2["USQUAN"]=sub2["USQUAN"].astype("category")
sub2["TAB12MDX"]=pd.to_numeric(sub2["TAB12MDX"])

sb.factorplot(x="USQUAN", y="TAB12MDX", data=sub3, kind="point", ci=None)
plt.xlabel("number of cigarettes smoked per day")
plt.ylabel("Proportion Nicotine Dependent")

sb.factorplot(x="USQUAN", y="TAB12MDX", data=sub4, kind="point", ci=None)
plt.xlabel("number of cigarettes smoked per day")
plt.ylabel("Proportion Nicotine Dependent")

#While the trend is a little different for each, there's not enough data to support 
#that major depression moderates the effect.

#For a correlation coefficient relationship, you can also have a moderating variable.
data['urbanrate']=pd.to_numeric(data['urbanrate'])
data['incomeperperson']=pd.to_numeric(data['incomeperperson'])
data['internetuserate']=pd.to_numeric(data['internetuserate'])
data['incomperperson']=data['incomeperperson'].replace(' ', np.nan)

data_clean=data.dropna()

print(sst.pearsonr(data_clean['urbanrate'],data_clean['internetuserate']))

def incomegrp(row):
    if row['incomeperperson'] <= 744.239:
        return 1
    elif row['incomeperperson'] <= 9425.326:
        return 2
    elif row['incomeperperson'] > 9425.326:
        return 3
    
data_clean['incomegrp']=data_clean.apply(lambda row: incomegrp(row),axis=1)

chk1= data_clean['incomegrp'].value_counts(sort=False, dropna=False)
print(chk1)

sub1=data_clean[(data_clean['incomegrp']==1)]
sub2=data_clean[(data_clean['incomegrp']==2)]
sub3=data_clean[(data_clean['incomegrp']==3)]

print("association between urbanrate and internet use rate in low income countries")
print(sst.pearsonr(sub1['urbanrate'], sub1['internetuserate']))
print("association between urbanrate and internet use rate in medium income countries")
print(sst.pearsonr(sub2['urbanrate'], sub2['internetuserate']))
print("association between urbanrate and internet use rate in high income countries")
print(sst.pearsonr(sub3['urbanrate'], sub3['internetuserate']))

#Can see the p-value differs for each. Can see where it matters.
