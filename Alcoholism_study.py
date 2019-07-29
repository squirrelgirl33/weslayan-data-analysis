# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:08:03 2019

@author: Kathryn
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy.stats as sst
import seaborn as sb
import matplotlib.pyplot as plt #seaborn is dependent upon this to create graphs#Create univariate graphs to show center and spread.



nesarc = pd.read_csv('NESARC Data.csv', low_memory=False)


#To avoid runtime errors
pd.set_option('display.float_format', lambda x:'%f'%x)

#Need three tables as frequency variables.

#For my data, I want to look only at people who have no history of alcohol abuse
#or alcohol dependence. Then, I want to look at two groups.
#Group 1: Those with a reported family history of alcoholism.
#Group 2: Those with no reported family history of alcoholism.

#	Alcohol abuse/dependence in last 12 months / 3648-3648  ALCABDEP12DX
print("0. No alcohol diagnosis" + "\n" 
      "1. Alcohol abuse only" + "\n" 
      "2. Alcohol dependence only" + "\n" 
      "3. Alcohol abuse and dependence")
print("Percentage of Alcohol abuse/dependence in last 12 months")
aa12_p = nesarc['ALCABDEP12DX'].value_counts(sort=False, normalize=True)
print(aa12_p)

#	Alcohol abuse/dependence prior to last 12 months / 3649-3649  ALCABDEPP12DX
print("Percentage of Alcohol abuse/dependence prior to last 12 months")
aa0_p = nesarc['ALCABDEPP12DX'].value_counts(sort=False, normalize=True)
print(aa0_p)

#If I want to only look at those that had no alcohol dependence in the last 12 
#months to see what the distribution is for alcohol dependence prior...
#sub1 = nesarc[(nesarc['ALCABDEP12DX']==0)]
#sub2 = sub1.copy()
#print(sub2['ALCABDEPP12DX'].value_counts(sort=False, normalize=True))
#The percentage goes up for those with no alcohol dependence, suggesting that 
#some people who have alcohol dependence currently had alcohol dependence prior.

#Subset my data to obtain people with no alcohol abuse/dependence
subaa = nesarc[(nesarc['ALCABDEP12DX']==0) & (nesarc['ALCABDEPP12DX']==0)]

subaa1=subaa.copy()

#Now, I want to look at people with a reported family history of alcoholism.

#First, I need to subset people who answered yes to a question of whether they 
#had an alcoholic father/mother/aunt/uncle/granmother/grandfather. I will use 
#the | (or) operator since they just need to have answered yes to one condition.
#subaafam = nesarc[(nesarc['S2DQ1']==1) | (nesarc['S2DQ2']==1) | 
#        (nesarc['S2DQ7C2']==1) | (nesarc['S2DQ9C2']==1) | (nesarc['S2DQ8C2']==1)
#         | (nesarc['S2DQ10C2']==1) | (nesarc['S2DQ11']==1) | (nesarc['S2DQ13A']==1)
#          | (nesarc['S2DQ12']==1) | (nesarc['S2DQ13B']==1)]

#subaafam1=subaafam.copy()


#Then, I will compare the length of this data set to the length of the total
#data set to get a percentage.
#print("The percentage of all tested with a known, elder relative with alcoholism:")
#print(len(subaafam1)/len(nesarc)*100)

#Now, I need to repeat this for my previously subsetted dataset of people
#who have not been diagnosed with alcohol dependence/abuse.
subaafam20 = subaa1[(subaa1['S2DQ1']==1) | (subaa1['S2DQ2']==1) | 
        (subaa1['S2DQ7C2']==1) | (subaa1['S2DQ9C2']==1) | (subaa1['S2DQ8C2']==1)
         | (subaa1['S2DQ10C2']==1) | (subaa1['S2DQ11']==1) | (subaa1['S2DQ13A']==1)
          | (subaa1['S2DQ12']==1) | (subaa1['S2DQ13B']==1)]

subaafam2=subaafam20.copy()


#Then, I will compare the length of this data set to the length of the non-
#diagnosed data set to get a percentage.
print("The percentage of those with no history of alcohol dependence/abuse with a known, elder relative with alcoholism:")
print(len(subaafam2)/len(subaa1)*100)

#For actual analysis, I also need a "no family history" subset. This needs to
#be "and" stringed, as I want no family history.
#First, compared with the overall dataset.
#subaafam30 = nesarc[(nesarc['S2DQ1']==2) & (nesarc['S2DQ2']==2) & 
#        (nesarc['S2DQ7C2']==2) & (nesarc['S2DQ9C2']==2) & (nesarc['S2DQ8C2']==2)
#         & (nesarc['S2DQ10C2']==2) & (nesarc['S2DQ11']==2) & (nesarc['S2DQ13A']==2)
#          & (nesarc['S2DQ12']==2) & (nesarc['S2DQ13B']==2)]

#subaafam3=subaafam30.copy()


#Then, I will compare the length of this data set to the length of the total
#data set to get a percentage.
#print("The percentage of all tested with no family history of alcoholism:")
#print(len(subaafam3)/len(nesarc)*100)

#Now, I need to repeat this for my previously subsetted dataset of people
#who have not been diagnosed with alcohol dependence/abuse.
subaafam40 = subaa1[(subaa1['S2DQ1']==2) & (subaa1['S2DQ2']==2) & 
        (subaa1['S2DQ7C2']==2) & (subaa1['S2DQ9C2']==2) & (subaa1['S2DQ8C2']==2)
         & (subaa1['S2DQ10C2']==2) & (subaa1['S2DQ11']==2) & (subaa1['S2DQ13A']==2)
          & (subaa1['S2DQ12']==2) & (subaa1['S2DQ13B']==2)]

subaafam4=subaafam40.copy()

print("The percentage of those with no history of alcohol dependence/abuse with no family history of alcoholism:")
print(len(subaafam4)/len(subaa1)*100)

#For the no history of alcohol dependence/abuse with family history, let's
#look at the number of drinks of alcohol on days when drinking.
subaafam2['S2AQ8B'] = pd.to_numeric(subaafam2['S2AQ8B'],errors='coerce').fillna(0).astype(int)

#print("Count of # drinks on days when drinking for those with family history:")
#c1 = subaafam2['S2AQ8B'].value_counts(sort=False, dropna=False)
#print(c1.sort_index())

print("Percentage of # drinks on days when drinking for those with family history:")
p1 = subaafam2['S2AQ8B'].value_counts(sort=False, dropna=False, normalize=True)
print(p1.sort_index())

#For the no history of alcohol dependence/abuse with NO family history, let's
#look at the number of drinks of alcohol on days when drinking.
subaafam4['S2AQ8B'] = pd.to_numeric(subaafam4['S2AQ8B'],errors='coerce').fillna(0).astype(int)
print("Percentage of # drinks on days when drinking for those with NO family history:")
p2 = subaafam4['S2AQ8B'].value_counts(sort=False, dropna=False,normalize=True)
print(p2.sort_index())

print("Key:" + "\n"
    "0. NA, former drinker or lifetime abstainer" +"\n"
    "1. Every day" +"\n"
    "2. Nearly every day" +"\n"
    "3. 3 to 4 times a week" +"\n"
    "4. 2 times a week" +"\n"
    "5. Once a week" +"\n"
    "6. 2 to 3 times a month" +"\n"
    "7. Once a month" +"\n"
    "8. 7 to 11 times in the last year" +"\n"
    "9. 3 to 6 times in the last year" +"\n"
    "10. 1 or 2 times in the last year" +"\n"
    "99. Unknown")

print("Percentage of # drinks on days when drinking for those with NO family history:")
p2 = subaafam4['S2AQ8B'].value_counts(sort=False, dropna=False, normalize=True)
print(p2.sort_index())

#Let's also look at how often they drank
#subaafam2['S2AQ8A'] = pd.to_numeric(subaafam2['S2AQ8A'],errors='coerce').fillna(0).astype(int)

print("Percetage of alcohohol consumption frequency for those with family history of alcoholism:")
p3 = subaafam2['S2AQ8A'].value_counts(sort=False, dropna=False, normalize=True)
print(p3.sort_index())


#subaafam4['S2AQ8A'] = pd.to_numeric(subaafam4['S2AQ8A'],errors='coerce').fillna(0).astype(int)

print("Percentage of alcohol consumption frequency for those with NO family history of alcoholism:")
p4 = subaafam4['S2AQ8A'].value_counts(sort=False, dropna=False, normalize=True)
print(p4.sort_index())

#For Lesson 3, I need to re-do one thing, and then re-sort some data.
#First, I need to re-do the grouping for the abstainers vs. did not answer.
#I thought those were the same, but according to the video, they're not. Need to
#re-clarify that data.
#subaafam2 is for those that have an alcoholic family history.
#subaafam4 is for those with no alcoholic family history
print("This data is for individuals who, as reported by this study, have not experienced alcohol abuse/dependence.")

print("Key:" + "\n"
    "0. Did not answer." +"\n"
    "1. Every day" +"\n"
    "2. Nearly every day" +"\n"
    "3. 3 to 4 times a week" +"\n"
    "4. 2 times a week" +"\n"
    "5. Once a week" +"\n"
    "6. 2 to 3 times a month" +"\n"
    "7. Once a month" +"\n"
    "8. 7 to 11 times in the last year" +"\n"
    "9. 3 to 6 times in the last year" +"\n"
    "10. 1 or 2 times in the last year" +"\n"
    "11. Have no drinken in the past 12 months." + "\n"
    "99. Unknown")

subaafam4['S2AQ8A'] = pd.to_numeric(subaafam4['S2AQ8A'],errors='coerce').fillna(0).astype(int)
subaafam4.loc[(subaafam4["S2AQ3"]!=9) & (subaafam4["S2AQ8A"]==0), "S2AQ8A"]=11
print("Distribution of alcohol consumption frequency for those with NO family history of alcoholism:")
subaafam4['S2AQ8A'] = pd.to_numeric(subaafam4['S2AQ8A'])
print("Percentages")
p4 = subaafam4['S2AQ8A'].value_counts(sort=False, dropna=False, normalize=True)
print(p4.sort_index())
print("Counts")
c4 = subaafam4['S2AQ8A'].value_counts(sort=False, dropna=False)
print(c4.sort_index())


subaafam2['S2AQ8A'] = pd.to_numeric(subaafam2['S2AQ8A'],errors='coerce').fillna(0).astype(int)
subaafam2.loc[(subaafam2["S2AQ3"]!=9) & (subaafam2["S2AQ8A"]==0), "S2AQ8A"]=11
print("Percentage of alcohol consumption frequency for those with family history of alcoholism:")
subaafam2['S2AQ8A'] = pd.to_numeric(subaafam2['S2AQ8A'])
print("Percentages")
p5 = subaafam2['S2AQ8A'].value_counts(sort=False, dropna=False, normalize=True)
print(p5.sort_index())
print("Counts")
c5 = subaafam2['S2AQ8A'].value_counts(sort=False, dropna=False)
print(c5.sort_index())

#Second, I want to get an actual number of days that they drank alcohol.
recode1={1:365, 2:300, 3:180, 4:104, 5:52, 6:30, 7:12, 8:9, 9:4.5, 10:1.5, 11:0}
subaafam4["USFREQ"]=subaafam4["S2AQ8A"].map(recode1)
print("Approximate number of days alcohol was consumed in the last 12 months for those with NO family history of alcoholism:")
print("Percentages")
p6 = subaafam4['USFREQ'].value_counts(sort=False, dropna=False, normalize=True)
print(p6.sort_index())
print("Counts")
c6 = subaafam4['USFREQ'].value_counts(sort=False, dropna=False)
print(c6.sort_index())

print("Approximate number of days alcohol was consumed in the last 12 months for those with family history of alcoholism:")
subaafam2["USFREQ"]=subaafam2["S2AQ8A"].map(recode1)
print("Percentages")
p7 = subaafam2['USFREQ'].value_counts(sort=False, dropna=False, normalize=True)
print(p7.sort_index())
print("Counts")
c7 = subaafam2['USFREQ'].value_counts(sort=False, dropna=False)
print(c7.sort_index())


#Third, I need to multiply the result from USFREQ (number of days drinking) by 
#S2AQ8B (number of drinks usually consumed on days when drinking) to get total 
#number of drinks consumed in a year. Then, I’ll divide this by 12 to get # drinks 
#consumed/month (on average).
subaafam4['S2AQ8B'] = pd.to_numeric(subaafam4['S2AQ8B'],errors='coerce').fillna(0).astype(int)
subaafam4['S2AQ8B']=subaafam4['S2AQ8B'].replace(99, np.nan)
#print(subaafam4["S2AQ8B"].value_counts(sort=False, dropna=False))#Just to make sure this worked
subaafam4["DRINKMO"]=subaafam4["S2AQ8B"]*subaafam4["USFREQ"]/12
print("Max drinks/month")
print(max(subaafam4["DRINKMO"])) #Get the idea of what the highest number needs to be.
print("Average drinks per month for those with NO family history of alcoholism:")
print("NaN indicates the amout of drinks consumed was unknown")
#Split into categories (0-0.5), (0.5+-1), (1+-3), (3+-7.5), (7.5+-15), (15+-30), (30+-60), (60+)
subaafam4["DRINKMOCUT"]=pd.cut(subaafam4.DRINKMO, [-1,0.5,1,3,7.5,15,30,60,520])
print("Percentages")
p8=subaafam4["DRINKMOCUT"].value_counts(sort=False, dropna=False, normalize=True)
print(p8.sort_index())
print("Counts")
c8=subaafam4["DRINKMOCUT"].value_counts(sort=False, dropna=False)
print(c8.sort_index())

subaafam2['S2AQ8B'] = pd.to_numeric(subaafam2['S2AQ8B'],errors='coerce').fillna(0).astype(int)
subaafam2['S2AQ8B']=subaafam2['S2AQ8B'].replace(99, np.nan)
#print(subaafam2["S2AQ8B"].value_counts(sort=False, dropna=False))#Just to make sure this worked
subaafam2["DRINKMO"]=subaafam2["S2AQ8B"]*subaafam2["USFREQ"]/12
print("Max drinks/month")
print(max(subaafam2["DRINKMO"])) #Get the idea of what the highest number needs to be.
print("Average drinks per month for those with family history of alcoholism")
#Split into categories (0-0.5), (0.5+-1), (1+-3), (3+-7.5), (7.5+-15), (15+-30), (30+-60), (60+)
subaafam2["DRINKMOCUT"]=pd.cut(subaafam2.DRINKMO, [-1,0.5,1,3,7.5,15,30,60,520])
print("Percentages")
p9=subaafam2["DRINKMOCUT"].value_counts(sort=False, dropna=False, normalize=True)
print(p9.sort_index())
print("Counts")
c9=subaafam2["DRINKMOCUT"].value_counts(sort=False, dropna=False)
print(c9.sort_index())


#Fourth, if I have skills to do this, I will try to re-divide the data for the family relationship.
#I will have categories for both parents, parent + additional relatives, 
#more than one additional relative, just one parent, just one non-parent relative.
#Ideally, I would like to have distributions for that.
subaafam2['S2DQ1'] = pd.to_numeric(subaafam2['S2DQ1']) #Father
subaafam2['S2DQ2'] = pd.to_numeric(subaafam2['S2DQ2']) #Mother
subaafam2['S2DQ7C2'] = pd.to_numeric(subaafam2['S2DQ7C2']) #Dad's bro
subaafam2['S2DQ8C2'] = pd.to_numeric(subaafam2['S2DQ8C2']) #Dad's sis
subaafam2['S2DQ9C2'] = pd.to_numeric(subaafam2['S2DQ9C2']) #Mom's bro
subaafam2['S2DQ10C2'] = pd.to_numeric(subaafam2['S2DQ10C2']) #Mom's sis
subaafam2['S2DQ11'] = pd.to_numeric(subaafam2['S2DQ11']) #Dad's pa
subaafam2['S2DQ12'] = pd.to_numeric(subaafam2['S2DQ12']) #Dad's ma
subaafam2['S2DQ13A'] = pd.to_numeric(subaafam2['S2DQ13A']) #Mom's pa
subaafam2['S2DQ13B'] = pd.to_numeric(subaafam2['S2DQ13B']) #Mom's ma

#Replace all "nos" to "0"s
subaafam2['S2DQ1'] = subaafam2['S2DQ1'].replace([2], 0) #Dad
subaafam2['S2DQ2'] = subaafam2['S2DQ2'].replace([2], 0) #Mom
subaafam2['S2DQ7C2'] = subaafam2['S2DQ7C2'].replace([2], 0) #D-bro
subaafam2['S2DQ8C2'] = subaafam2['S2DQ8C2'].replace([2], 0) #D-sis
subaafam2['S2DQ9C2'] = subaafam2['S2DQ9C2'].replace([2], 0) #M-bro
subaafam2['S2DQ10C2'] = subaafam2['S2DQ10C2'].replace([2], 0) #M-sis
subaafam2['S2DQ11'] = subaafam2['S2DQ11'].replace([2], 0) #D-pa
subaafam2['S2DQ12'] = subaafam2['S2DQ12'].replace([2], 0) #D-ma
subaafam2['S2DQ13A'] = subaafam2['S2DQ13A'].replace([2], 0) #M-pa
subaafam2['S2DQ13B'] = subaafam2['S2DQ13B'].replace([2], 0) #M-ma


subaafam2['S2DQ1'] = subaafam2['S2DQ1'].replace([9], np.nan) #Dad
subaafam2['S2DQ2'] = subaafam2['S2DQ2'].replace([9], np.nan) #Mom
subaafam2['S2DQ7C2'] = subaafam2['S2DQ7C2'].replace([9], np.nan) #D-bro
subaafam2['S2DQ8C2'] = subaafam2['S2DQ8C2'].replace([9], np.nan) #D-sis
subaafam2['S2DQ9C2'] = subaafam2['S2DQ9C2'].replace([9], np.nan) #M-bro
subaafam2['S2DQ10C2'] = subaafam2['S2DQ10C2'].replace([9], np.nan) #M-sis
subaafam2['S2DQ11'] = subaafam2['S2DQ11'].replace([9], np.nan) #D-pa
subaafam2['S2DQ12'] = subaafam2['S2DQ12'].replace([9], np.nan) #D-ma
subaafam2['S2DQ13A'] = subaafam2['S2DQ13A'].replace([9], np.nan) #M-pa
subaafam2['S2DQ13B'] = subaafam2['S2DQ13B'].replace([9], np.nan) #M-ma

subaafam2["AAFAMPAR"]=subaafam2["S2DQ1"]+subaafam2["S2DQ2"]
subaafam2["AAFAMEXT"]=subaafam2["S2DQ7C2"]+subaafam2["S2DQ8C2"]+subaafam2["S2DQ9C2"]+subaafam2["S2DQ10C2"]+subaafam2["S2DQ11"]+subaafam2["S2DQ12"]+subaafam2["S2DQ13A"]+subaafam2["S2DQ13B"]


def AAFAM(row):
    if row["AAFAMPAR"]==1 and row["AAFAMEXT"]==0:
        return 1 #1 alcoholic parent
    if row["AAFAMPAR"]==2 and row["AAFAMEXT"]==0:
        return 2 #both alcoholic parents
    if row["AAFAMPAR"]==0 and row["AAFAMEXT"]==1:
        return 3 #one alcoholic relative
    if row["AAFAMPAR"]==0 and row["AAFAMEXT"]>1:
        return 4 #multiple alcoholic relatives
    if row["AAFAMPAR"]==1 and row["AAFAMEXT"]>0:
        return 5 #One alcoholic parent and at least 1 alcoholic relative
    if row["AAFAMPAR"]>1 and row["AAFAMEXT"]>0:
        return 6 #Both parents and at least 1 relative

print("For those with family history of alcohol abuse/dependece, but no personal history, I wanted to look at the distribution of family with alcoholism (parents vs extended family)" + "\n"
      "Number code:" + "\n"
      "1. A single alcoholic parent" + "\n"
      "2. Two alcoholic parents" + "\n"
      "3. One alcoholic extended relative" + "\n"
      "4. Multiple alcoholic extended relatives" + "\n"
      "5. One alcoholic parents and at least one alcoholic relative" + "\n"
      "6. Both parents and at least one extended relative" + "\n"
      "nan A parent/relative alcohol abuse/dependence was unknown")
subaafam2["AAFAM"] = subaafam2.apply(lambda row: AAFAM(row), axis=1)
c10=subaafam2["AAFAM"].value_counts(sort=False, dropna=False)
p10=subaafam2["AAFAM"].value_counts(sort=False, dropna=False, normalize=True)
print("Percentages")
print(p10.sort_index()) 
print("Counts")
print(c10.sort_index()) 

subaafam21 = subaafam2[["AAFAM", "S2DQ1", "S2DQ2", "S2DQ7C2", "S2DQ8C2", "S2DQ9C2", "S2DQ10C2", "S2DQ11", "S2DQ12", "S2DQ13A", "S2DQ13B"]]
a=subaafam21.head(n=25)
print("Categrization confirmation:")
print(a)

#Graph time
#This is just general counts.
#Will do number of people with no reported alcohol problem that have family history of alcoholism.
subaa = nesarc[(nesarc['ALCABDEP12DX']==0) & (nesarc['ALCABDEPP12DX']==0)] #No history of alcohol abuse subgroup

subaa1=subaa.copy()
#Now need to make a single variable for those with family history.
subaa1['S2DQ1'] = pd.to_numeric(subaa1['S2DQ1']) #Father
subaa1['S2DQ2'] = pd.to_numeric(subaa1['S2DQ2']) #Mother
subaa1['S2DQ7C2'] = pd.to_numeric(subaa1['S2DQ7C2']) #Dad's bro
subaa1['S2DQ8C2'] = pd.to_numeric(subaa1['S2DQ8C2']) #Dad's sis
subaa1['S2DQ9C2'] = pd.to_numeric(subaa1['S2DQ9C2']) #Mom's bro
subaa1['S2DQ10C2'] = pd.to_numeric(subaa1['S2DQ10C2']) #Mom's sis
subaa1['S2DQ11'] = pd.to_numeric(subaa1['S2DQ11']) #Dad's pa
subaa1['S2DQ12'] = pd.to_numeric(subaa1['S2DQ12']) #Dad's ma
subaa1['S2DQ13A'] = pd.to_numeric(subaa1['S2DQ13A']) #Mom's pa
subaa1['S2DQ13B'] = pd.to_numeric(subaa1['S2DQ13B']) #Mom's ma

#Replace all "nos" to "0"s
subaa1['S2DQ1'] = subaa1['S2DQ1'].replace([2], 0) #Dad
subaa1['S2DQ2'] = subaa1['S2DQ2'].replace([2], 0) #Mom
subaa1['S2DQ7C2'] = subaa1['S2DQ7C2'].replace([2], 0) #D-bro
subaa1['S2DQ8C2'] = subaa1['S2DQ8C2'].replace([2], 0) #D-sis
subaa1['S2DQ9C2'] = subaa1['S2DQ9C2'].replace([2], 0) #M-bro
subaa1['S2DQ10C2'] = subaa1['S2DQ10C2'].replace([2], 0) #M-sis
subaa1['S2DQ11'] = subaa1['S2DQ11'].replace([2], 0) #D-pa
subaa1['S2DQ12'] = subaa1['S2DQ12'].replace([2], 0) #D-ma
subaa1['S2DQ13A'] = subaa1['S2DQ13A'].replace([2], 0) #M-pa
subaa1['S2DQ13B'] = subaa1['S2DQ13B'].replace([2], 0) #M-ma


subaa1['S2DQ1'] = subaa1['S2DQ1'].replace([9], np.nan) #Dad
subaa1['S2DQ2'] = subaa1['S2DQ2'].replace([9], np.nan) #Mom
subaa1['S2DQ7C2'] = subaa1['S2DQ7C2'].replace([9], np.nan) #D-bro
subaa1['S2DQ8C2'] = subaa1['S2DQ8C2'].replace([9], np.nan) #D-sis
subaa1['S2DQ9C2'] = subaa1['S2DQ9C2'].replace([9], np.nan) #M-bro
subaa1['S2DQ10C2'] = subaa1['S2DQ10C2'].replace([9], np.nan) #M-sis
subaa1['S2DQ11'] = subaa1['S2DQ11'].replace([9], np.nan) #D-pa
subaa1['S2DQ12'] = subaa1['S2DQ12'].replace([9], np.nan) #D-ma
subaa1['S2DQ13A'] = subaa1['S2DQ13A'].replace([9], np.nan) #M-pa
subaa1['S2DQ13B'] = subaa1['S2DQ13B'].replace([9], np.nan) #M-ma


def FAM(row):
    if row["S2DQ1"]>0 or row["S2DQ2"]>0 or row["S2DQ7C2"]>0 or row["S2DQ8C2"]>0 or row["S2DQ9C2"]>0 or row["S2DQ10C2"]>0 or row["S2DQ11"]>0 or row["S2DQ12"]>0 or row["S2DQ13A"]>0 or row["S2DQ13B"]>0:
        return 1
    else:
        return 0

subaa1["FAM"] = subaa1.apply(lambda row: FAM(row), axis=1)
subaa12 = subaa1[["FAM", "S2DQ1", "S2DQ2", "S2DQ7C2", "S2DQ8C2", "S2DQ9C2", "S2DQ10C2", "S2DQ11", "S2DQ12", "S2DQ13A", "S2DQ13B"]]
subaa12.head(n=25)

c11=subaa1["FAM"].value_counts(sort=False, dropna=False)
p11=subaa1["FAM"].value_counts(sort=False, dropna=False, normalize=True)
print("Family history of alcohol abuse or dependence--1 is yes, 0 is no")
print(c11)

subaa1["FAM"] = subaa1["FAM"].astype('category')
sb.countplot(x="FAM", data=subaa1)
plt.xlabel("Presence of alcohol abuse or dependence in previous generation")
plt.title("Alcoholism in family history for individuals with no personal history of alcohol abuse or dependence")

#Will do drinks/month for those that have/do not have alcoholic family history.
sb.distplot(subaafam2["DRINKMO"].dropna(), kde=False);
plt.xlabel("Number of Drinks Per Month")
plt.title("Estimated # of Drinks/Month for those with Family History of Alcoholism")


sb.distplot(subaafam4["DRINKMO"].dropna(), kde=False);
plt.xlabel("Number of Drinks Per Month")
plt.title("Estimated # of Drinks/Month for those with NO Family History of Alcoholism")

#Will do closeness of relative for those with family history
#Might only post one or two to the actual blog, though.





#Create a graph to show the association between the explanantory variables and the response variables.
#Drinks/month is the response variable. With and w/o familial alcoholism is the explanatory variable.
#So need to do a bar chart for that (response is quantiative, explanatory is categorical). Will have to go back to the original dataset before I split it.
subaa1['S2AQ8A'] = pd.to_numeric(subaa1['S2AQ8A'],errors='coerce').fillna(0).astype(int)
subaa1.loc[(subaa1["S2AQ3"]!=9) & (subaa1["S2AQ8A"]==0), "S2AQ8A"]=11
subaa1["USFREQ"]=subaa1["S2AQ8A"].map(recode1)
subaa1['S2AQ8B'] = pd.to_numeric(subaa1['S2AQ8B'],errors='coerce').fillna(0).astype(int)
subaa1['S2AQ8B']=subaa1['S2AQ8B'].replace(99, np.nan)
#print(subaafam4["S2AQ8B"].value_counts(sort=False, dropna=False))#Just to make sure this worked
subaa1["DRINKMO"]=subaa1["S2AQ8B"]*subaa1["USFREQ"]/12
print(max(subaa1["DRINKMO"]))
subaa1['DRINKMO10']=pd.qcut(subaa1.DRINKMO, 10, duplicates="drop", labels=["1=50%tile", "2=60%tile", "3=70%tile", "4=80%tile", "5=90%tile", "6=100%tile"])
subaa1["DRINKMO10"].head(n=25)

subaa1["FAM"] = subaa1["FAM"].astype('category')
subaa1["FAM2"]=subaa1["FAM"]

subaa1["FAM2"]=subaa1["FAM2"].cat.rename_categories(["No Family History", "Family History"])

sb.factorplot(x='FAM2', y='DRINKMO', data=subaa1, kind="bar")
plt.xlabel('Family Alcoholism')
plt.ylabel('Average of Drinks/Month')

sb.countplot(x="DRINKMO10", hue="FAM2", data=subaa1)
plt.xlabel("Percentiles of Alcohol Consumption")
plt.ylabel("Counts of individuals in the Percentiles")
plt.xticks(rotation = 45)

          
          
#Secondary question is closeness of relationship of those with alcoholism (categorical, explanatory) to alcohol consumption (quantitive, response). This is also a bar chart.
#Will probably need to clean up that dataset that I made earlier.


#The below is a clean-up of the family categorization that I did earlier where
#I had to remove the nan categorization. 
parents = np.dstack((subaafam2["S2DQ1"],subaafam2["S2DQ2"])) #making variables of interest into a single array
print(parents.shape) #double checking
parents2 = np.nansum(parents,2) #adding the two elements together on the correct dimension
print(parents2.T.shape) #checking that if I transpose it again it's now in a column once more
subaafam2["AAFAMPAR2"]=parents2.T #adding the column into the dataset as a new variable
extfam = np.dstack((subaafam2["S2DQ7C2"],subaafam2["S2DQ8C2"],subaafam2["S2DQ9C2"],subaafam2["S2DQ10C2"],subaafam2["S2DQ11"],subaafam2["S2DQ12"],subaafam2["S2DQ13A"],subaafam2["S2DQ13B"]))
print(extfam.shape)
extfam2=np.nansum(extfam, 2)
print(extfam2.shape)
subaafam2["AAFAMEXT2"]=extfam2.T

def AAFAM2(row):
    if row["AAFAMPAR2"]==1 and row["AAFAMEXT2"]==0:
        return 1 #1 alcoholic parent
    if row["AAFAMPAR2"]==2 and row["AAFAMEXT2"]==0:
        return 2 #both alcoholic parents
    if row["AAFAMPAR2"]==0 and row["AAFAMEXT2"]==1:
        return 3 #one alcoholic relative
    if row["AAFAMPAR2"]==0 and row["AAFAMEXT2"]>1:
        return 4 #multiple alcoholic relatives
    if row["AAFAMPAR2"]==1 and row["AAFAMEXT2"]>0:
        return 5 #One alcoholic parent and at least 1 alcoholic relative
    if row["AAFAMPAR2"]>1 and row["AAFAMEXT2"]>0:
        return 6 #Both parents and at least 1 relative


print("For those with family history of alcohol abuse/dependece, but no personal history, I wanted to look at the distribution of family with alcoholism (parents vs extended family)" + "\n"
      "Number code:" + "\n"
      "1. A single alcoholic parent" + "\n"
      "2. Two alcoholic parents" + "\n"
      "3. One alcoholic extended relative" + "\n"
      "4. Multiple alcoholic extended relatives" + "\n"
      "5. One alcoholic parents and at least one alcoholic relative" + "\n"
      "6. Both parents and at least one extended relative")
subaafam2["AAFAM2"] = subaafam2.apply(lambda row: AAFAM2(row), axis=1)
c12=subaafam2["AAFAM2"].value_counts(sort=False, dropna=False)
p12=subaafam2["AAFAM2"].value_counts(sort=False, dropna=False, normalize=True)
print("Percentages")
print(p12.sort_index()) 
print("Counts")
print(c12.sort_index()) 

subaafam2["AAFAM2"]=subaafam2["AAFAM2"].astype("category")
#subaafam2["AAFAM2"]=subaafam2["AAFAM2"].cat.rename_categories(["Single Alcoholic Parent", "Two Alcoholic Parents", "One Alcoholic Extended Relative", "Multiple Alcoholic Extended Relatives", "One Alcoholic Parent and at least one alcoholic extended relative", "Both Alcoholic parents and at least one extended relative"])

sb.factorplot(x='AAFAM2', y='DRINKMO', data=subaafam2, kind="bar")
plt.xlabel('Family Alcoholism')
plt.ylabel('Proportion of Drinks/Month')

#Now re-do with with 0s
parents = np.dstack((subaa1["S2DQ1"],subaa1["S2DQ2"])) #making variables of interest into a single array
print(parents.shape) #double checking
parents2 = np.nansum(parents,2) #adding the two elements together on the correct dimension
print(parents2.T.shape) #checking that if I transpose it again it's now in a column once more
subaa1["AAFAMPAR2"]=parents2.T #adding the column into the dataset as a new variable
extfam = np.dstack((subaa1["S2DQ7C2"],subaa1["S2DQ8C2"],subaa1["S2DQ9C2"],subaa1["S2DQ10C2"],subaa1["S2DQ11"],subaa1["S2DQ12"],subaa1["S2DQ13A"],subaa1["S2DQ13B"]))
print(extfam.shape)
extfam2=np.nansum(extfam, 2)
print(extfam2.shape)
subaa1["AAFAMEXT2"]=extfam2.T

def AAFAM2(row):
    if row["AAFAMPAR2"]==1 and row["AAFAMEXT2"]==0:
        return 1 #1 alcoholic parent
    if row["AAFAMPAR2"]==2 and row["AAFAMEXT2"]==0:
        return 2 #both alcoholic parents
    if row["AAFAMPAR2"]==0 and row["AAFAMEXT2"]==1:
        return 3 #one alcoholic relative
    if row["AAFAMPAR2"]==0 and row["AAFAMEXT2"]>1:
        return 4 #multiple alcoholic relatives
    if row["AAFAMPAR2"]==1 and row["AAFAMEXT2"]>0:
        return 5 #One alcoholic parent and at least 1 alcoholic relative
    if row["AAFAMPAR2"]>1 and row["AAFAMEXT2"]>0:
        return 6 #Both parents and at least 1 relative
    if row["AAFAMPAR2"]==0 and row["AAFAMEXT2"]==0:
        return 7 #No known alcoholic family history


print("For those with family history of alcohol abuse/dependece, but no personal history, I wanted to look at the distribution of family with alcoholism (parents vs extended family)" + "\n"
      "Number code:" + "\n"
      "1. A single alcoholic parent" + "\n"
      "2. Two alcoholic parents" + "\n"
      "3. One alcoholic extended relative" + "\n"
      "4. Multiple alcoholic extended relatives" + "\n"
      "5. One alcoholic parents and at least one alcoholic relative" + "\n"
      "6. Both parents and at least one extended relative" + "\n"
      "7. No known alcoholic family history")
subaa1["AAFAM2"] = subaa1.apply(lambda row: AAFAM2(row), axis=1)
c13=subaa1["AAFAM2"].value_counts(sort=False, dropna=False)
p13=subaa1["AAFAM2"].value_counts(sort=False, dropna=False, normalize=True)
print("Percentages")
print(p13.sort_index()) 
print("Counts")
print(c13.sort_index()) 

subaa1["AAFAM2"]=subaa1["AAFAM2"].astype("category")
subaa1["AAFAM2"]=subaa1["AAFAM2"].cat.rename_categories(["1 Par", "2 Par", "1 ExtRel", ">1 ExtRel", "1 Par, ExtRel", "2 Par, ExtRel", "None known"])

sb.factorplot(x='AAFAM2', y='DRINKMO', data=subaa1, kind="bar")
plt.xlabel('Family Alcoholism')
plt.ylabel('Avg of Drinks/Month')
plt.xticks(rotation = 45)

subaa1["DRINKMO"].head(n=10)
subaa1["USFREQ"].head(n=10)
subaa1["S2AQ8B"].head(n=10)


#Run ANOVA for Family Alcoholism and Avg Drinks/Month

subaa12=subaa1[["DRINKMO", "AAFAM2"]].dropna()

aafammodel = smf.ols(formula="DRINKMO ~ C(AAFAM2)", data=subaa12)
aafamresults=aafammodel.fit()
print(aafamresults.summary())
#F-statistic of 3.82e-06, so something is significantly different.

mc_aafam = multi.MultiComparison(subaa12["DRINKMO"], subaa12["AAFAM2"])
res_aafam = mc_aafam.tukeyhsd() #Request the test
print(res_aafam.summary())

#Why do the .dropna() matter?
len(subaa12)
subaa12["AAFAM2"].value_counts(sort=False, dropna=False)
len(subaa1)
subaa1["DRINKMO"].value_counts(sort=False, dropna=False).sort_index()
#It looks like I drop about 300 values with dropna between these two data sets.
#Must be in the DRINKMO category. Yup. Confirmed it--254 nan there.

#Chi-square test between two categorical variables.
#Will need to perform post hoc analysis
#Can ask if people who have no history of alcohol abuse/dependence were more
#likely to be abstain from alcohol (S2AQ1) if they had family history (FAM)

subaa1["S2AQ1"]=subaa1["S2AQ1"].astype("category")
subaa1["ABST"]=subaa1["S2AQ1"].cat.rename_categories(["Drinks", "Abstains"])
ct1=pd.crosstab(subaa1["ABST"], subaa1["FAM2"]) #categorical variables
print(ct1) #get counts
colsum=ct1.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct=ct1/colsum
print(colpct)

print("chi-square value, p value, expected counts")
cs1=sst.chi2_contingency(ct1)
print(cs1)

#If I need to do post hoc, I guess I can look for abstinence for those with different
#categories of family history
ct2=pd.crosstab(subaa1["ABST"], subaa1["AAFAM2"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
#7 degrees of freedom

print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)
print("Expected chi-square for 7 degrees of freedom is 14.07.")
print("Corrected p-value for 20 comparisons")
0.05/20

recode2={"1 Par":"1 Par", "2 Par":"2 Par"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv1']=subaa1['AAFAM2'].map(recode2)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv1"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode3={"1 Par":"1 Par", "1 ExtRel":"1 ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv2']=subaa1['AAFAM2'].map(recode3)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv2"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode4={"1 Par":"1 Par", ">1 ExtRel":">1 ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv3']=subaa1['AAFAM2'].map(recode4)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv3"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode5={"1 Par":"1 Par", "1 Par, ExtRel":"1 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv4']=subaa1['AAFAM2'].map(recode5)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv4"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode6={"1 Par":"1 Par", "2 Par, ExtRel":"2 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv5']=subaa1['AAFAM2'].map(recode6)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv5"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode7={"1 Par":"1 Par", "None known":"None known"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv6']=subaa1['AAFAM2'].map(recode7)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv6"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode8={"2 Par":"2 Par", "None known":"None known"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv7']=subaa1['AAFAM2'].map(recode8)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv7"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode9={"2 Par":"2 Par", "2 Par, ExtRel":"2 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv8']=subaa1['AAFAM2'].map(recode9)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv8"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode10={"2 Par":"2 Par", "1 Par, ExtRel":"1 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv9']=subaa1['AAFAM2'].map(recode10)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv9"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode11={"2 Par":"2 Par", ">1 ExtRel":">1 ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv10']=subaa1['AAFAM2'].map(recode11)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv10"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode12={"2 Par":"2 Par", "1 ExtRel":"1 ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv11']=subaa1['AAFAM2'].map(recode12)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv11"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode13={"1 ExtRel":"1 ExtRel", ">1 ExtRel":">1 ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv12']=subaa1['AAFAM2'].map(recode13)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv12"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode14={"1 ExtRel":"1 ExtRel", "1 Par, ExtRel":"1 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv13']=subaa1['AAFAM2'].map(recode14)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv13"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode15={"1 ExtRel":"1 ExtRel", "2 Par, ExtRel":"2 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv14']=subaa1['AAFAM2'].map(recode15)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv14"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode16={"1 ExtRel":"1 ExtRel", "None known":"None known"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv15']=subaa1['AAFAM2'].map(recode16)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv15"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)


recode16={">1 ExtRel":">1 ExtRel", "1 Par, ExtRel":"1 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv15']=subaa1['AAFAM2'].map(recode16)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv15"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode17={">1 ExtRel":">1 ExtRel", "2 Par, ExtRel":"2 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv16']=subaa1['AAFAM2'].map(recode17)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv16"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode18={">1 ExtRel":">1 ExtRel", "None known":"None known"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv17']=subaa1['AAFAM2'].map(recode18)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv17"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode19={"1 Par, ExtRel":"1 Par, ExtRel", "None known":"None known"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv18']=subaa1['AAFAM2'].map(recode19)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv18"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode20={"1 Par, ExtRel":"1 Par, ExtRel", "2 Par, ExtRel":"2 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv19']=subaa1['AAFAM2'].map(recode20)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv19"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

recode21={"None known":"None known", "2 Par, ExtRel":"2 Par, ExtRel"} #keeping 2 values but exclude other values in variable
subaa1['FAMCOMPv20']=subaa1['AAFAM2'].map(recode21)
ct2=pd.crosstab(subaa1["ABST"], subaa1["FAMCOMPv20"]) #categorical variables
print(ct2) #get counts
colsum2=ct2.sum(axis=0)#use counts from crosstab table. axis=0 says to sum all values in each column
#axis=0 means columns. axis=1 means rows
colpct2=ct2/colsum2
print(colpct2)
print("chi-square value, p value, expected counts")
cs2=sst.chi2_contingency(ct2)
print(cs2)

#My data's not great for correlation coefficient. I could try two things.
#Number of drinks/mo correlated to number of family members that have bee diagnosed with alcoholism
#However, I think that since so many have none, this will be heavily skewed.
#Instead, I might do numbers of drinks consumed to number of days drinking


allfam = np.dstack((subaafam2["S2DQ1"],subaafam2["S2DQ2"],subaafam2["S2DQ7C2"],subaafam2["S2DQ8C2"],subaafam2["S2DQ9C2"],subaafam2["S2DQ10C2"],subaafam2["S2DQ11"],subaafam2["S2DQ12"],subaafam2["S2DQ13A"],subaafam2["S2DQ13B"]))
allfam2 = np.nansum(allfam,2)
subaafam2["AAFAMTOT"]=allfam2.T

famdrink = subaafam2[["AAFAMTOT", "DRINKMO"]].dropna()

sb.regplot(y="DRINKMO", x="AAFAMTOT", data=famdrink)
plt.xlabel("Family members with alcoholic abuse or dependence")
plt.ylabel("Number of drinks/month")

print("Association between number of alcoholic family members and drinks/mo")
print(sst.pearsonr(famdrink["AAFAMTOT"], famdrink["DRINKMO"]))


subaa1_dd=subaa1[["S2AQ8B", "DOBY", "DRINKMO"]].dropna()

max(subaa1_dd["DOBY"])

subaa1_dd["AGE"]=2002-subaa1_dd["DOBY"] #Study was published in 2002

subaa1_dd["AGE"].head(n=25)


sb.regplot(y="DRINKMO", x="AGE", data=subaa1_dd)
plt.xlabel("Age")
plt.ylabel("Drinks/month")


print("Association between number of number of drinks when drinking and year born")
print(sst.pearsonr(subaa1_dd["AGE"], subaa1_dd["DRINKMO"]))

(-.025620117177638197)^2

#Look at whether alcoholic consumption is affected by family history but also account
#for age
#Will need to further subdivide data into age groups (under 25, 26-50, 50+)
subaa1["AGE"]=2002-subaa1["DOBY"]

subaa1["AGE"].head(n=10)

subaa13=subaa1[["DRINKMO", "AAFAM2", "AGE"]].dropna()

subaa13.head(n=25)

subaa1_25=subaa13[(subaa13["AGE"]<=25)]
subaa1_50=subaa13[(subaa13["AGE"]>25) & (subaa13["AGE"]<=50)]
subaa1_old=subaa13[(subaa13["AGE"]>50)]

#25 and under
print("association between family history of alcoholism and drinks/month for those over 50")
aafammod25 = smf.ols(formula="DRINKMO~C(AAFAM2)", data=subaa1_25).fit()
print(aafammod25.summary())

mc_aafam25 = multi.MultiComparison(subaa1_25["DRINKMO"], subaa1_25["AAFAM2"])
res_aafam25 = mc_aafam25.tukeyhsd() #Request the test
print(res_aafam25.summary())

sb.factorplot(x='AAFAM2', y='DRINKMO', data=subaa1_25, kind="bar")
plt.xlabel('Family Alcoholism')
plt.ylabel('Avg of Drinks/Month')
plt.xticks(rotation = 45)

print(subaa1_25.groupby("AAFAM2").mean())
print(subaa1_25.groupby("AAFAM2").std())

#26-50
print("association between family history of alcoholism and drinks/month for those from 26-50")
aafammod50 = smf.ols(formula="DRINKMO~C(AAFAM2)", data=subaa1_50).fit()
print(aafammod50.summary())

mc_aafam50 = multi.MultiComparison(subaa1_50["DRINKMO"], subaa1_50["AAFAM2"])
res_aafam50 = mc_aafam50.tukeyhsd() #Request the test
print(res_aafam50.summary())

sb.factorplot(x='AAFAM2', y='DRINKMO', data=subaa1_50, kind="bar")
plt.xlabel('Family Alcoholism')
plt.ylabel('Avg of Drinks/Month')
plt.xticks(rotation = 45)

print(subaa1_50.groupby("AAFAM2").mean())
print(subaa1_50.groupby("AAFAM2").std())

#50+
print("association between family history of alcoholism and drinks/month for those under 25")
aafammodold = smf.ols(formula="DRINKMO~C(AAFAM2)", data=subaa1_old).fit()
print(aafammodold.summary())

mc_aafamold = multi.MultiComparison(subaa1_old["DRINKMO"], subaa1_old["AAFAM2"])
res_aafamold = mc_aafamold.tukeyhsd() #Request the test
print(res_aafamold.summary())

sb.factorplot(x='AAFAM2', y='DRINKMO', data=subaa1_old, kind="bar")
plt.xlabel('Family Alcoholism')
plt.ylabel('Avg of Drinks/Month')
plt.xticks(rotation = 45)

print(subaa1_old.groupby("AAFAM2").mean())
print(subaa1_old.groupby("AAFAM2").std())


#For regression, I might try to develop a new category of those that most often
#drink hard alcohol vs less hard...though people might mix, and that'll create a third category.
#I can also look at sex and ethnicity and set one category to 0.
#I can also try to group by income and set the median income to 0. 


#I want to check that my DRINKMO numbers are accurate.
subaa13=subaa1[["DRINKMO", "PERSINCOME", "USFREQ", "S2AQ8B", "S2AQ8A", "AGEGROUP", "SEX", "ETH"]]

subaa13.nlargest(25, "DRINKMO")


#There are too many ethnicity categories, so I'm going to try to shrink them. Apologies if these categories suck, I'm going off of instinct and general geographical region.
def ETH(row):
    if row["S1Q1E"]==4 or row["S1Q1E"]==5 or row["S1Q1E"]==6 or row["S1Q1E"]==7 or row["S1Q1E"]==12 or row["S1Q1E"]==13 or row["S1Q1E"]==14 or row["S1Q1E"]==15 or row["S1Q1E"]==17 or row["S1Q1E"]==18 or row["S1Q1E"]==19 or row["S1Q1E"]==20 or row["S1Q1E"]==22 or row["S1Q1E"]==27 or row["S1Q1E"]==29 or row["S1Q1E"]==37 or row["S1Q1E"]==38 or row["S1Q1E"]==40 or row["S1Q1E"]==41 or row["S1Q1E"]==44 or row["S1Q1E"]==45 or row["S1Q1E"]==46 or row["S1Q1E"]==50 or row["S1Q1E"]==51 or row["S1Q1E"]==55:
        return 0 #Caucasion/European/Russian Descent
    if row["S1Q1E"]==1 or row["S1Q1E"]==2 or row["S1Q1E"]==25 or row["S1Q1E"]==26 or row["S1Q1E"]==28 or row["S1Q1E"]==31 or row["S1Q1E"]==33 or row["S1Q1E"]==48 or row["S1Q1E"]==53 or row["S1Q1E"]==54 or row["S1Q1E"]==56:
        return 1 #African/Middle Eastern/Caribbean Descent
    if row["S1Q1E"]==10 or row["S1Q1E"]==16 or row["S1Q1E"]==21 or row["S1Q1E"]==23 or row["S1Q1E"]==24 or row["S1Q1E"]==30 or row["S1Q1E"]==32 or row["S1Q1E"]==34 or row["S1Q1E"]==42 or row["S1Q1E"]==47 or row["S1Q1E"]==49 or row["S1Q1E"]==52 or row["S1Q1E"]==57:
        return 2 #South Asian/East Asian/Pacific Islander
    if row["S1Q1E"]==3 or row["S1Q1E"]==8 or row["S1Q1E"]==9 or row["S1Q1E"]==11 or row["S1Q1E"]==35 or row["S1Q1E"]==36 or row["S1Q1E"]==39 or row["S1Q1E"]==43 or row["S1Q1E"]==58:
        return 3 #Central American/South American/General Latino/Native American
    else:
        return 4 #Other or unknown


print("For those with family history of alcohol abuse/dependece, but no personal history, I wanted to look at the distribution of family with alcoholism (parents vs extended family)" + "\n"
      "Number code:" + "\n"
      "0. Caucasion/European/Russian Descent" + "\n"
      "1. African/Middle Eastern/Caribbean Descent" + "\n"
      "2. South Asian/East Asian/Pacific Islander" + "\n"
      "3. Central American/South American/General Latino/Native American" + "\n"
      "4. Other or unknown")
subaa1["ETH"] = subaa1.apply(lambda row: ETH(row), axis=1)

recode_x={1:0, 2:1} #Male=1, female=2
subaa1["SEX"]=subaa1["SEX"].map(recode_x)

#Making age categories. once agian, everyone will probably hate this grouping.
def AGEGROUP(row):
    if row["AGE"]<=30:
        return 0 #Young
    if row["AGE"]>=31 and row["AGE"]<=59:
        return 1 #Middle
    if row["AGE"]>59:
        return 2 #Old
    
subaa1["AGEGROUP"] = subaa1.apply(lambda row: AGEGROUP(row), axis=1)

subaa1["AGE_c"] = (subaa1["AGE"] - subaa1["AGE"].mean()) 
subaa1["AGE_c"].mean() #Mean is near 0

#Simplified income categories
def PERSINCOME(row):
    if row["S1Q10A"]<=30000:
        return 0 #Less than or equal to 30000/year
    if row["S1Q10A"]>30000 and row["S1Q10A"]<70000:
        return 1 #Greater than 30,000, less than 70,000/year
    if row["S1Q10A"]>=70000:
        return 2 #Greater than 70,000/year
    
subaa1["PERSINCOME"] = subaa1.apply(lambda row: PERSINCOME(row), axis=1)

#Standardize personal income to mean=0

subaa1["ADJPERSINCOME"]=(subaa1["S1Q10A"]-subaa1["S1Q10A"].mean())

subaa1["ADJPERSINCOME"].mean()

subaa1["S2AQ4A"]=pd.to_numeric(subaa1["S2AQ4A"],errors='coerce').fillna(0).astype(int)
subaa1["S2AQ5A"]=pd.to_numeric(subaa1["S2AQ5A"],errors='coerce').fillna(0).astype(int)
subaa1["S2AQ6A"]=pd.to_numeric(subaa1["S2AQ6A"],errors='coerce').fillna(0).astype(int)
subaa1["S2AQ7A"]=pd.to_numeric(subaa1["S2AQ7A"],errors='coerce').fillna(0).astype(int)



#Whether drank hard/non-hard alcohol or both
def ALCCHOICE(row):
    if (row["S2AQ4A"]==1 or row["S2AQ5A"]==1 or row["S2AQ6A"]==1) and row["S2AQ7A"]!=1:
        return 0 #nonhard alcohol
    if row["S2AQ4A"]!=1 and row["S2AQ5A"]!=1 and row["S2AQ6A"]!=1 and row["S2AQ7A"]==1:
        return 1 #only hard alcohol
    if (row["S2AQ4A"]==1 or row["S2AQ5A"]==1 or row["S2AQ6A"]==1) and row["S2AQ7A"]==1:
        return 2 #both non-hard and hard alcohol
    if row["S2AQ4A"]!=1 and row["S2AQ5A"]!=1 and row["S2AQ6A"]!=1 and row["S2AQ7A"]!=1:
        return 3 #does not drink

    
subaa1["ALCCHOICE"] = subaa1.apply(lambda row: ALCCHOICE(row), axis=1)

subaa13=subaa1[["DRINKMO", "ADJPERSINCOME", "S2AQ7A", "ALCCHOICE", "AGEGROUP", "AAFAM"]]

subaa13.nlargest(25, "DRINKMO")
#Looking for how income affects Drinks/month. Then would want to look at how it correlates to presence or absense of family history of alcoholism?
subaa14=subaa1[["DRINKMO","ADJPERSINCOME"]].dropna()
subaa14["DRINKMO"]=pd.to_numeric(subaa14["DRINKMO"])
subaa14["ADJPERSINCOME"]=pd.to_numeric(subaa14["ADJPERSINCOME"])

modeldrinkincome = smf.ols(formula="DRINKMO ~ ADJPERSINCOME", data=subaa14).fit()
print(modeldrinkincome.summary())
sb.regplot(y="DRINKMO", x="ADJPERSINCOME", data=subaa14)
plt.xlabel("Income Adjusted Around Mean")
plt.ylabel("Number of drinks/month")

subaa14_1 = subaa14[(subaa14["ADJPERSINCOME"]<=50000)&(subaa14["DRINKMO"]>=2)]
subaa14_1["ADJPERSINCOME"].max()
subaa14_1["DRINKMO"].min()
modeldrinkincome = smf.ols(formula="DRINKMO ~ ADJPERSINCOME", data=subaa14_1).fit()
print(modeldrinkincome.summary())
sb.regplot(y="DRINKMO", x="ADJPERSINCOME", data=subaa14_1)
plt.xlabel("Income Adjusted Around Mean")
plt.ylabel("Number of drinks/month")

def AAFAM(row):
    if row["AAFAMPAR2"]>=1 and row["AAFAMEXT2"]>=1:
        return 1 #alcoholic family
    else:
        return 0 #No known alcoholic family history
    
subaa1["AAFAM"] = subaa1.apply(lambda row: AAFAM(row), axis=1)

#I think I need to stratify my sample set for a smaller portion or something.
#Maybe limit age group and only to people who drink at least once a week?

subaa1_2=subaa1[["DRINKMO","ADJPERSINCOME","SEX","ETH", "AGE_c", "AAFAM", "ALCCHOICE"]].dropna()

modeltest1 = smf.ols(formula="DRINKMO ~ ADJPERSINCOME", data=subaa1_2).fit()
print(modeltest1.summary())
fig1=sm.qqplot(modeltest1.resid, line='r') #There is really high deviation at the upper end of the mean, somewhat at lower
stdres1 = pd.DataFrame(modeltest1.resid_pearson) #convert array of standardized residuals to dataframe. Reg3 has the results of the data analysis. resid tells python to use standardized residuals
fig1_1=plt.plot(stdres1, 'o', ls='None') #generate a plot of standardized residuals. 'o' tells python to use dots. ls='None' tells python not to connec the markers
l = plt.axhline(y=0, color='r') #draws horizontal line on the graph
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print(fig1_1)
#Tons are falling out of residual plot. So the current model is unacceptable. Does adding more help?

modeltest2 = smf.ols(formula="DRINKMO ~ ADJPERSINCOME + AGE_c", data=subaa1_2).fit()
print(modeltest2.summary())
fig2=sm.qqplot(modeltest2.resid, line='r')
stdres2 = pd.DataFrame(modeltest2.resid_pearson) #convert array of standardized residuals to dataframe. Reg3 has the results of the data analysis. resid tells python to use standardized residuals
fig2_1=plt.plot(stdres2, 'o', ls='None') #generate a plot of standardized residuals. 'o' tells python to use dots. ls='None' tells python not to connec the markers
l = plt.axhline(y=0, color='r') #draws horizontal line on the graph
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print(fig2_1)

#This is apparently wrong since you can't do categorical variables (ALCCHOICE) with more than two groups this way...
#modeltest3 = smf.ols(formula="DRINKMO ~ ADJPERSINCOME + AGE_c + ALCCHOICE", data=subaa1_2).fit()
#print(modeltest3.summary())
#fig3=sm.qqplot(modeltest3.resid, line='r')
#stdres3 = pd.DataFrame(modeltest3.resid_pearson) #convert array of standardized residuals to dataframe. Reg3 has the results of the data analysis. resid tells python to use standardized residuals
#fig3_1=plt.plot(stdres3, 'o', ls='None') #generate a plot of standardized residuals. 'o' tells python to use dots. ls='None' tells python not to connec the markers
#l = plt.axhline(y=0, color='r') #draws horizontal line on the graph
#plt.ylabel('Standardized Residual')
#plt.xlabel('Observation Number')
#print(fig3_1)
#when I add alcchoice in, age is no longer relevant.
#fig3_2 = plt.figure(figsize(12,8)) #numbers specify size of the plot image in pixels
#fig3_2 = sm.graphics.plot_regress_exog(modeltest3, "ADJPERSINCOME", fig=fig3_2) #put in residual results and the explanatory variable that you want to plot
#print(fig3_2)

modeltest3 = smf.ols(formula="DRINKMO ~ ADJPERSINCOME + AGE_c + C(ALCCHOICE)", data=subaa1_2).fit()
print(modeltest3.summary())
fig3=sm.qqplot(modeltest3.resid, line='r')
stdres3 = pd.DataFrame(modeltest3.resid_pearson) #convert array of standardized residuals to dataframe. Reg3 has the results of the data analysis. resid tells python to use standardized residuals
fig3_1=plt.plot(stdres3, 'o', ls='None') #generate a plot of standardized residuals. 'o' tells python to use dots. ls='None' tells python not to connec the markers
l = plt.axhline(y=0, color='r') #draws horizontal line on the graph
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print(fig3_1)
#when I add alcchoice in, age is no longer relevant.
fig3_2 = plt.figure(figsize(12,8)) #numbers specify size of the plot image in pixels
fig3_2 = sm.graphics.plot_regress_exog(modeltest3, "ADJPERSINCOME", fig=fig3_2) #put in residual results and the explanatory variable that you want to plot
print(fig3_2)


modeltest4 = smf.ols(formula="DRINKMO ~ ADJPERSINCOME + SEX + ETH + AGE_c + ALCCHOICE", data=subaa1_2).fit()
print(modeltest4.summary())
fig4=sm.qqplot(modeltest4.resid, line='r')
stdres4 = pd.DataFrame(modeltest4.resid_pearson) #convert array of standardized residuals to dataframe. Reg3 has the results of the data analysis. resid tells python to use standardized residuals
fig4_1=plt.plot(stdres4, 'o', ls='None') #generate a plot of standardized residuals. 'o' tells python to use dots. ls='None' tells python not to connec the markers
l = plt.axhline(y=0, color='r') #draws horizontal line on the graph
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print(fig4_1)
#when I add alcchoice in, age is no longer relevant.

modeltest5 = smf.ols(formula="DRINKMO ~ ADJPERSINCOME + SEX + ETH + AGE_c + ALCCHOICE +AAFAM", data=subaa1_2).fit()
print(modeltest5.summary())
fig5=sm.qqplot(modeltest5.resid, line='r')
stdres5 = pd.DataFrame(modeltest5.resid_pearson) #convert array of standardized residuals to dataframe. Reg3 has the results of the data analysis. resid tells python to use standardized residuals
fig5_1=plt.plot(stdres5, 'o', ls='None') #generate a plot of standardized residuals. 'o' tells python to use dots. ls='None' tells python not to connec the markers
l = plt.axhline(y=0, color='r') #draws horizontal line on the graph
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print(fig5_1)

sb.regplot(y="DRINKMO", x="ADJPERSINCOME", data=subaa14_1)

fig6 = sm.graphics.influence_plot(modeltest1, size=8)
print(fig6)

#Lesson 4
def DAYDRINKING(row):
    if row["DRINKMO"]>30:
        return 1 #drinks approximately every day or more
    else:
        return 0 #drinks less than every day
    
subaa1["DAYDRINKING"] = subaa1.apply(lambda row: DAYDRINKING(row), axis=1)

subaa1["AAFAM"].head(n=25)

reg1=smf.ols("DAYDRINKING ~ ADJPERSINCOME + SEX+ AGE_c + AAFAM", data=subaa1).fit()
print(reg1.summary())
print("Odds Ratio")
print(np.exp(reg1.params))
#Can also get a confidence interval
params1=reg1.params
conf1=reg1.conf_int()
conf1["OR"]=params1
conf1.columns=["Lower CI", "Upper CI", "OR"] #The confidence intervals for the response variables overlap. Cannot tell which one is more strongly associated.
print(np.exp(conf1))

reg2=smf.ols("DAYDRINKING ~ AAFAM + SEX +AGE_c", data=subaa1).fit()
print(reg2.summary()) #AAFAM appears to have no affect by itself

reg3=smf.ols("DAYDRINKING ~  AGE_c + AAFAM", data=subaa1).fit()
print(reg3.summary()) #SEX confounds income and alcoholic family


reg4=smf.ols("DAYDRINKING ~ AAFAM", data=subaa1).fit()
print(reg4.summary()) #AAFAM is still signifiant without income


