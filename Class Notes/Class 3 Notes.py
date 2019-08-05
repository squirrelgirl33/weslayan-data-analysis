# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:57:44 2019

@author: Kathryn
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm #This is new
import statsmodels.formula.api as smf
import seaborn as sb
import matplotlib.pyplot as plt 
import statsmodels.stats.multicomp as multi
import scipy.stats as sst


#Set Pandas to show all columns in a dataframe
pd.set_option('display.max_columns', None)
#Set Pandas to show all rows in a dataframe
pd.set_option('display.max_rows', None)

nesarc = pd.read_csv('NESARC Data.csv', low_memory=False)


#To avoid runtime errors
pd.set_option('display.float_format', lambda x:'%f'%x)

nesarc['S3AQ3B1']=pd.to_numeric(nesarc['S3AQ3B1'])
nesarc['S3AQ3C1']=pd.to_numeric(nesarc['S3AQ3C1'])
nesarc['CHECK321']=pd.to_numeric(nesarc['CHECK321'])

sub1 = nesarc[(nesarc["AGE"]>=18)&(nesarc["AGE"]<=25)&(nesarc["CHECK321"]==1)]

sub2=sub1.copy()

#replace appropriate values to missing
sub2['S3AQ3B1']=sub2['S3AQ3B1'].replace(9, np.nan)
sub2['S3AQ3C1']=sub2['S3AQ3C1'].replace(99, np.nan)
recode1 = {1:30, 2:22, 3:14, 4:5, 5:2.5, 6:1}
sub2["USFREQMO"]=sub2["S3AQ3B1"].map(recode1)

sub2["USFREQMO"]=pd.to_numeric(sub2["USFREQMO"])

#Create a secondary variable with math
sub2["NUMCIGMO_EST"]=sub2["USFREQMO"]*sub2["S3AQ3C1"]
sub2["NUMCIGMO_EST"]=pd.to_numeric(sub2["NUMCIGMO_EST"])

#Confounding variables
#I lost my notes from lesson 1...hopefully won't affect things too much.
#Yo, is anyone else confused by the outfit the girl is wearing in the first video.
#Seriously, what is it?
#Do we know something about the presence or absence of coronary heart disease as related
#to coffee drinking? Smoking is the confounding variable. Explanatory variable is
#Coffee drinking. Response variable is coronary heart disease.

#Multiple regression - quantitative response variable
#Logistic regression - binary response variable

#Previously, there was an association between urban rate and internet use rate.
#Find the slope of the correlation line (y=mx+b). Internet use rate is y. Urban rate x
#To find the equation of the best fit line use:
modelname = smf.ols(formula="QUANT_RESPONSE ~ QUANT_EXPLANATORY", data=dataframe).fit()
print(modelname.summary())
#We use this same function to test analysis of variance.

data["internetuserate"]=pd.to_numeric(data["internetuserate"])
data["urbanrate"]=pd.to_numeric(data["urbanrate"])

print("OLS regression model for the association between urban rate and internet use rate")
reg1 = smf.ols("internetuserate ~ urbanrate", data=data).fit()
print(reg.summary())
#Dep variable is the name of the response variable
#No. Observations is number of observations
#Prob (F-statistic) is the p-value
#Coef column gives you the coefficients for the intercept and urban rate.
#From the coef, you can get your equation for slope.
#P > |t| is the p-value for the explanatory variable in association with our response variables.
#Also get an R squared value which gives you the proportion of the variance of the response variable than can be explained by the explanatory variable.

#Can use this equation value to predict values for y.

#The previous is great for quantitative data...but not for categorical.

#A regression model is still informative for categorical variables.
#Looking at depression (explanatory) and # of nicotine dependence symptoms (response)

nesarc["IDNUM"]=pd.to_numeric(nesarc["IDNUM"], errors="coerce")
nesarc["TAB12MDX"]=pd.to_numeric(nesarc["TAB12MDX"], errors="coerce")
nesarc["MAJORDEPLIFE"]=pd.to_numeric(nesarc["MAJORDEPLIFE"], errors="coerce")
nesarc["NDSymptoms"]=pd.to_numeric(nesarc["NDSymptoms"], errors="coerce")
nesarc["DYSLIFE"]=pd.to_numeric(nesarc["DYSLIFE"], errors="coerce")
nesarc["SEX"]=pd.to_numeric(nesarc["SEX"], errors="coerce")
nesarc["AGE"]=pd.to_numeric(nesarc["AGE"], errors="coerce")
nesarc["SOCPDLIFE"]=pd.to_numeric(nesarc["SOCPDLIFE"], errors="coerce")

sub1 = nesarc[(nesarc["AGE"]>=18)&(nesarc["AGE"]<=25)&(nesarc["CHECK321"]==1)]

sub2=sub1.copy()

reg1=smf.ols("NDSymptoms ~ MAJORDEPLIFE", data=nesarc).fit()
print(reg1.summary())
#Get a similar output as before.
#For coef of MAJORDEPLIFE, get the parameter estimate
#Once again, the P > |t| tells you statistical significance
#Can form the equation based upon this.

sub3 = sub2[['NDSymptoms', 'MAJORDEPLIFE']].dropna()
print("Mean")
ds1=sub3.groupby("MAJORDEPLIFE").mean()
print (ds1)
print("Standard deviation")
ds2 = sub3.groupby("MAJORDEPLIFE").std()
print(ds2)

sb.factorplot(x="MAJORDEPLIFE", y="NDSymptoms", data=sub3, kind="bar", ci=None)
plt.xlabel("MAJORDEPLIFE")
plt.ylabel("NDSymptoms")

#Linear Regression Model Assumptions are:
#Normality - residuals from linear regression model are normally distributed.
#Linearity - assocations between explanatory and response variables are linear
#Homoscedasticity - variability in the response variable is the same at all levels of the response variable
#Independence - observations in dataset are not correlated with each other. Clustered data and repeated measures data have correlated observations
#You also need to contend with multicolinearity, which happens when explanatory variables are highly correlated
#Also need to content with outliers

#Violation of the assumption of independence is the most serious and most likely to make an impact
#You can't fix assumption of indepedence by modifying variables or excluding data. The structure of the data itself often results in the violation.
#If independence is a problem, you need to use an alternative regression method that can take into account the lack of independence in the data.

#Outliers are observations that have unusual or extreme values compared to the others.
#They can really fuck with the line of best fit.
#You need to decide whether to get rid of outliers:
#Check to see if the conclusion or assumption changes if you get rid of the outlier.
#If yes, it is a data cleaning issue?
#If yes, can you fix the data cleaning issue. If so, do it. If not, recode it as missing.
#If it's not a data cleaning issue, does the outlier come from the target population?
#If yes, transform it and re-analyze the data. if no, delete the observation.

#If you don't have a good reason to exclude the population, you can try transforming the variable.

#Multicollinearity is when your explanatory variables are highly correlated with each other
#Each explanatory variable has to have a lot of unique variability that will contribute to explaining
#variability in the response variable. That way, you can figure out the contribution
#of each variable to the response variable.
#If they all overlap too much, then having multiple won't really clear anything up since they're
#pretty much all the same.
#Signs of multicollinearity: highly associated explanatory variable not significant
#negative regression coefficient that should be positive, taking out an explanatory
#variable drastically changes the results.
#To correct for this, one of the best thing is to choose just one variable.
#You can also try to aggregate or combine the variables to create a single variable.
#There are also joint hypothesis tests that you can try to do to account for this, though this course won't cover them.

#Adding additional explanatory variables to linear regression creates a multiple regression
#model
#When there are multiple explanatory variables in a multiple regression model
#then you look at the relationship between an explanatory and response variable
#while holding the other explanatory variables constant.
#This is typically done with the other explanatory variables held at 0...
#...but that's not practical for some variables.
#Categorical variables can be recoded to one value as 0 if necessary
#FOr quantitative explanaotry variables to have a 0 value, you have to center them.
#Re-code the variable so the mean equals 0.
#You ONLY center explanatory variable. Not response variables.

#Week3
#center quantitative IVs for regression analysis
sub2["NUMCIGMO_EST_c"]=(sub2["NUMCIGMO_EST"]-sub2["NUMCIGMO_EST"].mean())

print(sub2["NUMCIGMO_EST_c"].mean())
#1= Depression, 0 = no depression. That's the other variable, so that's fine.

#Now go back to the regression model with the centered values.
#Just adding a + to add a second explanatory variable works.
reg2 = smf.ols("NDSymptoms ~ MAJORDEPLIFE + NUMCIGMO_EST_c", data=sub2).fit()
print(reg2.summary())

sb.regplot(x="NDSymptoms", y="NUMCIGMO_EST_c", data=sub2)
#look at the coefficient table in the output and check the p-values (P>|t|),
#the parameter estimates (t). The parameter estimate positive values tell you
#that these things still have positive values with major depression even when 
#you account for both of them.

#They give a second example where controlling for another variable does matter.
reg4 = smf.ols("NDSymptoms ~ DYSLIFE + MAJORDEPLIFE", data=sub2).fit()
print(reg4.summary())
#P-value of DYSLIFE is no longer statistically significant
#Major depression confounds relationship between dysthmia and number of nicotine dependent symptoms

#We can keep adding factors to this.
sub2["SEX"].head(n=10)
recode1 = {1:0, 2:1}
sub2["SEX"]=sub2["SEX"].map(recode1)

reg5 = smf.ols("NDSymptoms ~ DYSLIFE + MAJORDEPLIFE + NUMCIGMO_EST_c + SEX", data=sub2).fit()
print(reg5.summary())
#Can see what variables are significant and which aren't.

#Confidence intervals will tell you which values of the parameters estimates are
#plausible within the population.
sub2["AGE_c"] = (sub2["AGE"]-sub2["AGE"].mean())
print(sub2["AGE_c"].mean())

reg5 = smf.ols("NDSymptoms ~ DYSLIFE + MAJORDEPLIFE + NUMCIGMO_EST_c + SEX + AGE", data=sub2).fit()
print(reg5.summary())
#In the coefficient table, the last two columns provide the lower and upper limits
#for the 95% confidence interval for each parameter estimate.
#If you look at the coefficient parameter estimate for majordeplife, it's ~1.3
#This means that on average individuals with major depression have 1.3x nicotine depndent symptoms
#than those without major depression.
#If you look at the confidence interval, it ranges from about 1.1-1.5, meaning that
#we're 95% certain that the true population parameter for this association falls between
#that range.

#If you look at something with a not significant p-value and you look at the confidence
#interval range, then you'll see that it includes 0. This means that you can't rule out with
#95% confidence that the association is 0.

#If you have a non-significant p-value, the confidence interval will include 0 as no association.

#Polynomial Regression
#Not all regression lines are linear.
scat1 = sb.regplot(x="urbanrate", y="femaleemployrate", scatter=True, data=sub1)
plt.xlabel("Urbanization Rate")
plt.ylabel("Female Employment Rate")
#Straight line doesn't always work
#Add a polynomial term to make the line curve
scat1 = sb.regplot(x="urbanrate", y="femaleemployrate", scatter=True, order=2, data=sub1) #order=2 adds a second order polynomial
plt.xlabel("Urbanization Rate")
plt.ylabel("Female Employment Rate")
#The quadratic line can do a better job of capturing the rates.
#To be sure, see if adding a second order polynomial term gives you a better regression model.

sub1=data[["urbanrate", "femaleemployrate", "internetuserate"]].dropna()

scat1=sb.regplot(x="urbanrate", y="femaleemployrate", scatter=True, order=2, data=sub1)
plt.xlabel("Urbanization Rate")
plt.ylabel("Female Employment Rate")

sub1["urbanrate_c"] = (sub1["urbanrate"]-sub1["urbanrate"].mean())
sub1["internetuserate_c"] = (sub1["internetuserate"]-sub1["internetuserate"].mean())
reg1=smf.ols("femaleemployrate ~ urbanrate_c", data=sub1).fit()
print(reg1.summary())

#If the Rsquared is really low, but it seems like there's a correlation based
#on the other metrics, the Rsquared can be adjusted by changing the line to a 
#polynomial.

#Regression model with second order polynomial:
reg1=smf.ols("femaleemployrate ~ urbanrate_c + I(urbanrate_c**2)", data=sub1).fit()
print(reg1.summary())
#I() is the identity function that returns the input in the parenthesis. It just returns
#the value in the parenthesis.

#If you get a multicollinearity warning, then it means there are variables in the model that
#are highly correlated with one another. Multicollinearity can make thinks difficult to determine.
#If you have two highly correlated varaibles, you typically only want to put one in the model.

#Centering helps reduce the correlation between linear and quadratic variables in a polynomial regression model.

#You can also add higher order polynomial terms (cubic, or quartic)
#Modeling more complex curves in sample data improves the fit, but it can
#also cause OVERFITTING. Overfitting occurs when you get a model that fits really well
#on that sample data, but it ONLY applies to that sample data. It can be applied to other
#data, not similar data from that population.

#Try to establish balance with BIAS-VARIANCE TRADE-OFF--have a model that fits your
#sample well and also fits other samples in that population.

#Evaluate your regression models for misspecification
#Specification is the process of developing a regression model
#If your data don't meet the regression assumptions or the model misses important explanatory variables
#then you have misspecification error.
#Perform diagnostics to identify this and address the error.
#Look at the e in the regression formula. The e is the error, or residual measurement.
#Examine residual plots to evaluate misspecification error.

#Add the internetuserate variable
reg3=smf.ols("femaleemployrate ~ urbanrate_c + I(urbanrate_c**2) + internetuserate_c", data=sub1).fit()
print(reg3.summary())
#If you look at this, look at the intercept.
#The intercept is the value of the response variable when all explanatory variables
#are held constant at zero.
#Since you center everything, the intercept is the femaleemploymentrate when all other variables
#are at the mean value.

#Look at the residual variability to see how large the residuals are and if the 
#regression assumptions are met and if there any outliers unduly influencing this.

#look at the residuals with a qqplot for normality
fig1=sm.qqplot(reg3.resid, line='r') #reg3 has our regression results
#line = 'r' tells python to put a red line on the plot.

#When you see the qqplot, you can see if the residuals follow as straight line and where they deviate
#This indicates that the data aren't perfectly following a normal distribtuion
#and that the quadratic urban rate might not fully estimate things
#Might want to include additional variables to improve estimation of observed curvilinearity

#Standarized residuals are the residual values transformed to have a mean of zero
#and a standard deviation of 1. This is called normalizing so that they fit a 
#standard normal distribution
#In a standard normal distribution, 68% of observations fall within one stdev
#of the mean
#95% of observations expected to fall within two stdev of the mean

#Graph the residuals of each osbervation
stdres = pd.DataFrame(reg3.resid_pearson) #convert array of standardized residuals to dataframe. Reg3 has the results of the data analysis. resid tells python to use standardized residuals
fig2=plt.plot(stdres, 'o', ls='None') #generate a plot of standardized residuals. 'o' tells python to use dots. ls='None' tells python not to connec the markers
l = plt.axhline(y=0, color='r') #draws horizontal line on the graph
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')
print(fig2)

#The plot lets you visualize whether the residuals falls within st dev of the mean
#If more than 1% of residuals have an absolute value of >2.5 or 5% of residuals
#have an absolute value >2, then the level of error within the model is unacceptable

#The biggest contributor to poor model fit is leaving out important explanatory
#variables 

#add additional regression diagnostics
fig3 = plt.figure(figsize=(12,8)) #numbers specify size of the plot image in pixels
fig3 = sm.graphics.plot_regress_exog(reg3, "internetuserate_c", fig=fig3) #put in residual results and the explanatory variable that you want to plot
print(fig3)

#Lower left hand corner plot is a partial regression residual plot. It attempts to show affect of adding
#internetuserate in the model given that one or more explanatory variables are already
#in the model
#there was lots of information here, but it went really fast and I didn't undestand it all.

#Leverage plot to identify observations that have an unusually large influence on
#the estimation of the predicted value of the response variable, or that are outliers,
#or both
#The leverage of the observation is how much the predicted scores for other observations
#would differ if the osbervations in question were not included in the analysis
#The leverage is always between 0 and 1. 0 has no effect on regression model.

fig4 = sm.graphics.influence_plot(reg3, size=8)
print(fig4)
#Will show you outliers


#Week 4

#What do you do with a categorical explanatory variable with more than two categories?
#I....already did this by accident, and probably incorrectly.
#Dummy coding/parameterization - process of coding category explanatory variables.
#Might want to compare one group to the average of other groups. This is called
#effect coding or effect paramterization
#Reference group coding is the most basic paramterization method.
#This is similar to post hoc variance comparisons.
#Unlike post hoc, where comparisons are tested after ANOVA, the comparisons
#are part of the estimation in the multi regression model.

#Add a categorical variable that has more than two categories
def ETHRACE(row):
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


sub2["ETHRACE"] = sub2.apply(lambda row: ETHRACE(row), axis=1)


reg6=smf.ols("NDSymptoms ~DYSLIFE + MAJORDEPLIFE+NUMCIGMO_EST_c+AGE_c+SEX+C(ETHRACE)", data=sub2).fit()
print(reg6.summary())
#The reference treatment group is the group with the value equal to 0 (so for us, Caucasian)

#The T.# is the categorical variable code for the group.

#If you want to use something else as the reference group besides what was coded as 0...
#You'll need to specify the treatment group
reg7=smf.ols("NDSymptoms ~DYSLIFE + MAJORDEPLIFE+NUMCIGMO_EST_c+AGE_c+SEX+C(ETHRACE, Treatment(reference=1))", data=sub2).fit()
print(reg7.summary())

def NICOTINEDEP(row):
    if row["TAB12MDX"]==1:
        return 1
    else:
        return 0

sub2["NICOTINEDEP"]=sub2.apply(lambda row: NICOTINEDEP(row), axis=1)

#Do logistic regression with the logit function
lreg1 = smf.logit(formula="NICOTINEDEP~SOCPDLIFE", data=sub2).fit()
print(lreg1.summary())
#In a logistic regression, the response variable only takes on values of 0s and 1s,
#so you can't do a best fit line.

#It would be more helpful to talk about the probability of being nicotine dependent
#Want to quantify the probability of getting a 1 vs a 0
#Choose odds ratio, a probability of an event occuring in one group compared to another
#instead of coefficients
#Get an odds model of 1, that means there's an equal probability of nicotine dependence
#for those with and without social phobia. Or more generally, the model is statistically
#non-significant.
#An odds ratio of >1, then the probability of becoming nicotine dependent is greater
#in those with social phobia than those without. Generally, that as the explanatory variable increases,
#the response variable more likely.
#If the odds ratio <1, then the probability of becoming nicotine dependent is lower
#among those with social phobia than in those without. Generally, as the explanatory variable
#increases, then the response is less likely.

print("Odds Ratio")
print(np.exp(lreg1.params))
#Can also get a confidence interval
params=lreg1.params
conf=lreg1.conf_int()
conf["OR"]=params
conf.columns=["Lower CI", "Upper CI", "OR"]
print(np.exp(conf))
#Looking at the confidence interval, you can get an idea of how the odds ratio
#would change for a different sample drawn from the population

#Control for major depression
lreg2 = smf.logit(formula="NICOTINEDEP~SOCPDLIFE + MAJORDEPLIFE", data=sub2).fit()
print(lreg2.summary()) #Both are independently associated
print("Odds Ratio")
print(np.exp(lreg2.params))
#Can also get a confidence interval
params2=lreg2.params
conf2=lreg2.conf_int()
conf2["OR"]=params2
conf2.columns=["Lower CI", "Upper CI", "OR"] #The confidence intervals for the response variables overlap. Cannot tell which one is more strongly associated.
print(np.exp(conf2))
#Can continue to add variables to the model to evaluate multiple predictors of binary response

lreg3 = smf.logit(formula="NICOTINEDEP~PANIC", data=sub2).fit()
print(lreg3.summary()) #Both are independently associated
print("Odds Ratio")
print(np.exp(lreg3.params))
#Can also get a confidence interval
params3=lreg3.params
conf3=lreg3.conf_int()
conf3["OR"]=params3
conf3.columns=["Lower CI", "Upper CI", "OR"] #The confidence intervals for the response variables overlap. Cannot tell which one is more strongly associated.
print(np.exp(conf3))
#If you add major depression to the model with panic disorder, there's no association
#with panic disorder and nicotine dependence
#Because of that, you don't interpret the odds ratio as anything for panic, but
#would do so for major depression.

#Always code your outcome variable as 0 means no outcome and 1 is outcome occurred.

