import numpy
import pandas
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 


data = pandas.read_csv('/Users/utpl/Documents/DataAnalysisTools/nesarc_pds.csv', low_memory=False)


#AMERICAN INDIAN OR ALASKA NATIVE Yes = 1 No = 2
#sub1_indianAlaska = data[(data['S1Q1D1']==1)]

#HISPANIC OR LATINO ORIGIN
#sub2_latino = data[(data['S1Q1C']==1)]

#ASIAN
#sub3_asian = data[(data['S1Q1D2']==1)]


#BLACK OR AFRICAN AMERICAN
#sub4_african = data[(data['S1Q1D3']==1)]



#NUMBER OF EPISODES OF ALCOHOL ABUSE

data['S2BQ3B'] = data['S2BQ3B'].replace('BL', numpy.nan)
data['S2BQ3B'] = data['S2BQ3B'].replace(' ', 0) 

data['S2BQ3B'] = pandas.to_numeric(data['S2BQ3B'])

sub1 = data[(data['S2BQ3B'] >= 20) & (data['S2BQ3B'] < 99)]

#NUMBER OF EPISODES OF ALCOHOL ABUSE and OCCUPATION: CURRENT OR MOST RECENT JOB
sub2 = sub1[['S2BQ3B', 'S1Q9B']].dropna()


model1 = smf.ols(formula='S2BQ3B ~ C(S1Q9B)', data=sub2).fit()
print (model1.summary())



mc1 = multi.MultiComparison(sub2['S2BQ3B'], sub2['S1Q9B'])
res1 = mc1.tukeyhsd()
print(res1.summary())


