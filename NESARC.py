# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

#Set Pandas to show all columns in a dataframe
pd.set_option('display.max_columns', None)
#Set Pandas to show all rows in a dataframe
pd.set_option('display.max_rows', None)

nesarc = pd.read_csv('NESARC Data.csv', low_memory=False)


#To avoid runtime errors
pd.set_option('display.float_format', lambda x:'%f'%x)

print(len(nesarc)) #number of rows
print(len(nesarc.columns)) #number of columns

#To just get the number of values for a particular variable (in this case "TAB12MDX"):

c1 = nesarc["TAB12MDX"].value_counts(sort=False)

#The normalize=True allows you to get percentages
p1 = nesarc["TAB12MDX"].value_counts(sort=False, normalize=True)

print("counts for TAB12MDX - nicotine dependence in the past 12 months")
print(c1)

print("percentages for TAB12MDX - nicotine dependence in the past 12 months")
print(p1)

ct1=nesarc.groupby('TAB12MDX').size()
pt1=nesarc.groupby('TAB12MDX').size()*100/len(nesarc)
print(ct1)


#If you have variables that go up higher than 9, then you need to covnert them
#to numeric if you want them to be displayed in the correct order.
#Actually, how the lesson told you to do it is outdated. I don't know if you 
#even need to convert to numeric to do this. Try just ".sort_index()" next time.
nesarc['S3AQ3C1'] = pd.to_numeric(nesarc['S3AQ3C1'])
nesarc['TAB12MDX'] = pd.to_numeric(nesarc['TAB12MDX'])
nesarc['S3AQ3B1'] = pd.to_numeric(nesarc['S3AQ3B1'])
nesarc['CHECK321'] = pd.to_numeric(nesarc['CHECK321'])

c2 = nesarc["S3AQ3C1"].value_counts(sort=False, dropna=False)
print("counts for S3AQ3C1 - nicotine dependence in the past 12 months")
print(c2.sort_index())

#Will need to subset your data for your study.
sub1=nesarc[(nesarc['AGE']>=18) & (nesarc['AGE']<=25) & (nesarc['CHECK321']==1)]

#Can be useful to make a copy of this data. Not necessary, but can get rid of copy warnings.
sub2=sub1.copy()

#Remove missing data
sub2['S3AQ3B1']=sub2['S3AQ3B1'].replace(9, np.nan)
print("counts for S3AQB1 with 9 set to NAN")
c2 = sub2["S3AQ3B1"].value_counts(sort=False, dropna=False)
print(c2)

#Want to set 99 as missing since the value is unknown
sub2['S3AQ3C1'] = pd.to_numeric(sub2['S3AQ3C1'])
sub2['S3AQ3C1']=sub2['S3AQ3C1'].replace(99, np.nan)
print("counts for S3AQB1 with 99 set to NAN")
c3 = sub2["S3AQ3C1"].value_counts(sort=False, dropna=False)
print(c3.sort_index())

#Correct for skip patterns:
#Adding a category with code of 11 to indicate participants who didn't drink alcohol
#in the past 12 months...the code they gave me did not work. I have # it out.
#I included the code that did work for me.
#nesarc.loc[(nesarc['S2AQ3']!=9) & (nesarc['S2AQ8A'].isnull()), 'S2AQ8A']=11
nesarc['S2AQ8A'] = pd.to_numeric(nesarc['S2AQ8A'],errors='coerce').fillna(0).astype(int)
nesarc.loc[(nesarc["S2AQ3"]!=9) & (nesarc["S2AQ8A"]==0), "S2AQ8A"]=11
nesarc['S2AQ8A'] = pd.to_numeric(nesarc['S2AQ8A'])
test = nesarc["S2AQ8A"].value_counts(sort=False, dropna=False)
print(test.sort_index())

#recoding values
recode1 = {1:6, 2:5, 3:4, 4:3, 5:2, 6:1}
#Asking this recoding to be applied to the S3AQ3B1, but naming it something new.
sub2["USFREQ"]=sub2["S3AQ3B1"].map(recode1)
sub2["USFREQ"].head(n=5)

#Alternately, you can change this to a quantitative variable since it makes more sense.
#CHoose values that correspond to how much an individual smokes in a given month.
#Everyday, 30. Every other day, 15. Etc.
recode2 = {1:30, 2:22, 3:14, 4:5, 5:2.5, 6:1}
sub2["USFREQMO"]=sub2["S3AQ3B1"].map(recode2)
c4 = sub2["USFREQMO"].value_counts(sort=False, dropna=False)
print(c4.sort_index())

#Want to known #cigs smoke/month. Know the number of days, and can get #cigs from the
#USFREQMO that we made earlier.
#Let's subset our dataframe to the only varaibles we care about.
sub2["NUMCIGMO_EST"]=sub2["USFREQMO"] *sub2["S3AQ3C1"]
sub3=sub2[["IDNUM", "S3AQ3C1", "USFREQMO", "NUMCIGMO_EST"]]
sub3.head(25)

#If you want to combine a group of responses...
data=pd.read_csv("addhealth.csv")
#Need to turn everything into a number since you're going to sum all the variables.
data['HIGI4'] = pd.to_numeric(data['HIGI4'])
data['HIGI6A'] = pd.to_numeric(data['HIGI6A'])
data['HIGI6B'] = pd.to_numeric(data['HIGI6B'])
data['HIGI6C'] = pd.to_numeric(data['HIGI6C'])
data['HIGI6D'] = pd.to_numeric(data['HIGI6D'])
#Set aside missing data in 6 and 8
data['HIGI4'] = data['HIGI4'].replace([6,8], numpy.nan)
data['HIGI6A'] = data['HIGI6A'].replace([6,8], numpy.nan)
data['HIGI6B'] = data['HIGI6B'].replace([6,8], numpy.nan)
data['HIGI6C'] = data['HIGI6C'].replace([6,8], numpy.nan)
data['HIGI6D'] = data['HIGI6D'].replace([6,8], numpy.nan)

data["NUMETHNIC"]=data["HIGI4"]+data["HIGI6A"]+data["HIGI6B"]+data["HIGI6C"]+data["HIGI6D"]
print("counts for NUMETHNIC value")
chk8=data["NUMETHNIC"].value_counts(sort=False, dropna=False)
print(chk8)

#assigning variables to these values
def ETHNICITY(row):
    if row["NUMETHNIC"] >1:
        return 1
    if row["HIGI4"] ==1:
        return 2
    if row["HIGI6A"] ==1:
        return 3
    if row["HIGI6B"] ==1:
        return 4
    if row["HIGI6C"] ==1:
        return 5

#axis=1 tells Python to apply the function to each row.
#lambda operator is a way to make throwaway functions that just function where they're
#created
data["ETHNICITY"] = data.apply(lambda row: ETHNICITY(row), axis=1)
#So if NUMETHNIC is greater than 1, it will return 1 meaning multiple levels of ethnicity.
#Will need to print a description of this and probably also write it in the code.
#Will want to print out a data frame just to show that this is working correctly.
sub2 = data[["AID", "HIGI4", "HIGI6A", "HIGI6B", "HIGI6C", "HIGI6D", "NUMETNIC", "ETHNICITY"]]
a=sub2.head(n=25)
print(a)

#Further grouping observations.
print("AGE - 4 categories - quartiles")
sub2["AGEGROUP4"]=pd.qcut(sub2.AGE, 4, labels=["1=25%tile", "2=50%tile", "3=75%tile", "4=100%tile"])
c9 = sub2["AGEGROUP4"].value_counts(sort=False, dropna=True)
print(c9)

sub2["AGEGROUP4"]=pd.cut(sub2.AGE, [17,20,22,25])
print(pd.crosstab(sub2["AGEGROUP4"], sub2["AGE"]))

#VIsualizing variables as graphs
#To visualize data, you need to import additional libraries.

import seaborn as sb
import matplotlib.pyplot as plt #seaborn is dependent upon this to create graphs

#To order categorical variables properly, convert numerical variables into a format
#that python recognizes as categorical.

sub2["TAB12MDX"] = sub2["TAB12MDX"].astype('category')
#Basic code for univariable graph
sb.countplot(x="TAB12MDX", data=sub2)
plt.xlabel("Nicotine dependence past 12 months")
plt.title("Nicotine dependence in the past 12 months among young adult smokers in the NESARC Study")

#Now let's try that for a quantitative variable
sb.distplot(sub2["NUMCIGMO_EST"].dropna(), kde=False);
plt.xlabel("Number of Cigarettes per Month")
plt.title("Estimated # of Cigarettes/Month in NESARC Young Adult Study")

#Calculating st dev in python
print("describe number of cigarettes smoked per month")
descr1 = sub2["NUMCIGMO_EST"].describe()
print(descr1)
print("mean")
mean1=sub2["NUMCIGMO_EST"].mean()
print(mean1)
print("std")
std1=sub2["NUMCIGMO_EST"].std()
print(std1)
print("min")
min1=sub2["NUMCIGMO_EST"].min()
print(min1)
print("max")
max1=sub2["NUMCIGMO_EST"].max()
print(max1)
print("median")
med1=sub2["NUMCIGMO_EST"].median()
print(med1)
print("mean")
mode1=sub2["NUMCIGMO_EST"].mode()
print(mode1)

#For categorical data, the describe function will give you a description that's appropriate.          
print("describe nicotine dependence in the past 12 months in young adult smokers")
descr2 = sub2["TAB12MDX"].describe()
print(descr2)

#Quantitative variables are best described as histograms
#Then measure with shape, center, and spread

#Categorical variables are best as frequency distributions
#Need bar charts


#Going to make a new variable with a slightly more usable quant
sub2['PACKSPERMONTH']=sub2['NUMCIGMO_EST']/20
c2=sub2.groupby('PACKSPERMONTH').size()
print(c2)

sub2['PACKCATEGORY']=pd.cut(sub2.PACKSPERMONTH, [0, 5, 10, 20, 30, 147])

sub2['PACKCATEGORY']=sub2['PACKCATEGORY'].astype('category')

print('describe nicotine dependence')
desc3=sub2['PACKCATEGORY'].describe()
print(desc3)

print('pack category')
c7=sub2['PACKCATEGORY'].value_counts(sort=False, dropna=True)
print(c7)

sub2["TAB12MDX"]=pd.to_numeric(sub2["TAB12MDX"])
sb.factorplot(x="PACKCATEGORY", y="TAB12MDX", data=sub2, kind="bar", ci=None)#ci = None suppresses error bars
plt.xlabel("Packs per month")
plt.ylabel("Proportion Nicotine Dependent")

def SMOKEGRP(row):
    if row['TAB12MDX'] ==1:
        return 1
    elif row['USFREQMO']==30:
        return 2
    else:
        return 3
    
sub2['SMOKEGRP']=sub2.apply(lambda row: SMOKEGRP(row), axis=1)

#Need to simplify this to two choices or else it's too confusing to plot.
def DAILY(row):
    if row['USFREQMO']==30:
        return 1
    elif row['USFREQMO']!=30:
        return 0

sub2['DAILY']=sub2.apply(lambda row: DAILY(row), axis=1)

c4=sub2.groupby('DAILY').size()
print(c4)

#You can rename categorical variables to something that's not a number by first
#chaning the variable to a "category" and then renaming the categories
sub2["ETHRACE2A"]=sub2["ETHRACE2A"].astype("category")
sub2["ETHRACE2A"]=sub2["ETHRACE2A"].cat.rename_categories(["White", "Black", "NatAm", "Asian", "Hispanic"])

sb.factorplot(x='ETHRACE2A', y='DAILY', data=sub2, kind="bar", ci=None)
plt.xlabel('Ethnic Group')
plt.ylabel('Proportion of Daily Smokers')

#For Quantitative variables...
#This example is using Gapminder, so won't be running anything.

data['internetuserate']=pd.to_numeric(data['internetuserate'])
data['urbanrate']=pd.to_numeric(data['urbanrate'])
desc1=data['urbanrate'].describe()
print(desc1)
desc2=data['internetuserate'].describe()
print(desc2)

#Find if there's a relationship between the two variables using a scatterplot.
scat1=sb.regplot(x="urbanrate", y="internetuserate", fit_reg=False, data=data)#Fit_reg=False suppresses line of best fit
plt.xlabel('Urban Rate')
plt.ylabel=('Internet Use Rate')
plt.title=('Scatterplot for the Association Between Urban Rate and Internet Use Rate')

#The closer the points follow the linear pattern, the closer the relationship.
#there are many possible forms a scatterplot can take.

#Sometimes, the correlation seems sort of weak, and it could benefit from grouping.
#Try moving things into quartiles and look at distribution.
#You can now use this to make a bar chart.

data['INCOMEGROUP4']=pd.qcut(data.incomeperperson, 4, labels=["1=25%", "2=50%", "3=75%", "4=100%"])
sb.factorplot(x="INCOMEGROUP4", y="hivrate", data=data, kind="bar", ci=None)
plt.xlabel("Income group")
ply.ylabel("mean HIV rate")