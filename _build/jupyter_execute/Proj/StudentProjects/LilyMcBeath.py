#!/usr/bin/env python
# coding: utf-8

# # Russian Federation Economic Indicators
# 
# Author: Lily McBeath
# 
# Course Project, UC Irvine, Math 10, W22

# ## Introduction
# 
# I am planning to explore various economic indicators in the Russian Federation from 2006 to 2020, and to what extent they correlate with one another. I am using data on economic factors from [The World Bank's Data website](https://data.worldbank.org/country/russian-federation), cited below. I am also looking into whether any of these economic indicators correlate with the exchange rate between the Russian Ruble and US Dollar. This data may be interesting given the recent sharp fall in the value of the Ruble, following sanctions on Russia related to the war in Ukraine.

# ## Main portion of the project

# In[1]:


import numpy as np
import pandas as pd
df = pd.read_csv("russia.csv")


# In[2]:


df


# In[3]:


df.isna().sum()[36:]


# We can see that there is a lot of missing data here, which I will address soon.
# 
# First I want to drop the `Country Name`, `Country Code`, and `Unnamed: 65` columns, which are the same for all factors, and the `Indicator Code`, as the `Indicator Name` is more descriptive.

# In[4]:


df.drop(labels=["Country Name", "Country Code", "Unnamed: 65", "Indicator Code"], axis=1, inplace=True)


# Now I want to look at the factors that have no missing data, and starting in the year 2006.

# In[5]:


df.drop(df.iloc[:, 1:47], inplace=True, axis=1)
df.dropna(inplace=True)


# In[6]:


df


# I am going to transform the structure of this DataFrame somewhat to more closely mimic the data we are used to working with in this class.

# In[7]:


df = df.transpose().copy()
df.columns = df.iloc[0]
df = df[1:].copy()
df.index.name = 'Year'
df = df.reset_index()
df['Year'] = pd.to_datetime(df['Year']).dt.year
df.iloc[:,1:] = df.iloc[:,1:].apply(pd.to_numeric)


# In[8]:


df.head()


# In[9]:


df.columns


# I will perform an initial analysis with the population data, looking at how the rural population is related to dependency.

# In[10]:


df.iloc[:,df.columns.str.contains('Rural population')]


# In[11]:


df.iloc[:,df.columns.str.contains('dependency')]


# In[12]:


df = df.rename(columns={"Rural population (% of total population)":"Rural population percent of total", "Age dependency ratio (% of working-age population)":"Age dependency ratio"})


# According to the [World Bank's Glossary](https://databank.worldbank.org/metadataglossary/all/series), the age dependency ratio is the ratio of those younger than 15 or older than 64 to the working-age population, and the rural population as a percentage of total population is found according to the data from the United Nations Population Division.
# 
# I will use the rural population as a percentage of total population as input, and the age dependency ratio as a percent of working-age population as output. Since these are both percentile values, rescaling is likely unnecessary.

# In[13]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(df[['Rural population percent of total']], df['Age dependency ratio'])


# In[14]:


df['Age dependency ratio pred'] = reg.predict(df[['Rural population percent of total']])


# Now we will plot the results in Altair. The method for creating repeat layered charts is taken from the [Altair documentation](https://altair-viz.github.io/user_guide/compound_charts.html#repeated-charts) and requires Altair version 4.2.0 to run. If the charts display an error, uncomment the `!pip install altair==4.2.0` line, run the cell, comment the `!pip install altair==4.2.0` line again, and run the cell once more.

# In[15]:


import altair as alt
# !pip install altair==4.2.0 # (run this if graphs are not showing, then run graphs again)
c1 = alt.Chart(df).mark_line().encode(
    x="Rural population percent of total:Q",
    y=alt.Y('Age dependency ratio pred:Q',
        scale=alt.Scale(zero=False)
    )
).properties(
    title="Predicted and Actual Age Dependency Ratio"
)
c2 = alt.Chart(df).mark_point().encode(
    x="Rural population percent of total",
    y=alt.Y('Age dependency ratio:Q',
        scale=alt.Scale(zero=False)
    )
)
c3 = alt.Chart(df).mark_line().encode(
    x = 'Year:O',
    y=alt.Y(alt.repeat('layer'),
        type='quantitative',
        title='Rural Population and Age Dependency Ratio',
        scale=alt.Scale(zero=False)
        ),
    color=alt.ColorDatum(alt.repeat('layer'))
).properties(
    title="Actual Age Dependency vs. Rural Population"
).repeat(layer=["Age dependency ratio", "Rural population percent of total"])
alt.layer(c1, c2)|c3


# In[16]:


print(f"Age dependency ratio appears negatively correlated with rural population percentage, by a factor of {round(reg.coef_[0])}.")


# In[17]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(df['Age dependency ratio'],df['Age dependency ratio pred'])


# It appears from this preliminary work that rural population is a somewhat good indicator of the age dependency ratio, in that as the rural population as a percent of total population increases, the age dependency ratio as a percent of working population decreases.
# 
# The next step will be to consider some other non-population-related indicators and compare with the Russian Ruble/US Dollar exchange rate. The Ruble to US Dollar historical spot rates were obtained from the [Bank of England's Statistical Interactive Database](https://www.bankofengland.co.uk/boeapps/database/index.asp?first=yes&SectionRequired=I&HideNums=-1&ExtraInfo=true). We will reverse the order of the rows to sort by ascending years, and invert the average annual USD/RUB rates given to obtain the RUB/USD rates.

# In[18]:


rubusd = pd.read_csv("rubusd.csv")
rubusd = rubusd.reindex(index=rubusd.index[::-1])
rubusd.reset_index(inplace=True)
df["RATE"] = rubusd.iloc[:,2]
df["RATE"] = 1/df["RATE"]


# Let's examine how the indicators we looked at earlier compare to the RUB/USD rate from 2006 to 2020 using a chart, rescaling to more clearly compare the values.

# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = df.rename(columns={"Rural population percent of total":"Rural population percent", "Age dependency ratio":"Age dependency", "RATE":"Ruble to Dollar"})
dfscaled = df.copy()
dfscaled.iloc[:,1:] = scaler.fit_transform(df.iloc[:,1:])
c4 = alt.Chart(dfscaled).mark_line().encode(
    x = 'Year:O',
    y=alt.Y(alt.repeat('layer'), type='quantitative', title='Scaled Indicators and Exchange Rate'),
    color=alt.ColorDatum(alt.repeat('layer'))
).properties(
    title="Scaled Rural Population Percentage and Age Dependency Ratio vs. Ruble to Dollar Exchange Rates",
    width=700
).repeat(
    layer=["Rural population percent","Age dependency", "Ruble to Dollar"]
)
c4


# It is interesting that the rural population as a percent of total population in Russia (blue) seems to decline together with the RUB/USD rate (red), while the age dependency ratio as a percent of working population in Russia (orange) has been increasing since 2009.
# 
# What do the remaining indicators look like, in comparison to the Ruble / Dollar exchange rates?
# 
# Let us see if we can use some of these indicators to predict the Ruble to Dollar exchange rates. After looking into the [definitions of leading vs. lagging indicators](https://www.albany.edu/~bd445/Economics_301_Intermediate_Macroeconomics_Slides_Spring_2014/Leading_Economic_Indicators_(Print).pdf) and the [top economic indicators for the U.S. economy](https://www.conference-board.org/data/bci/index.cfm?id=2160) (acknowledging that these may not be optimal for an analysis of the Russian economy, but useful nonetheless for our purposes), I will try to predict the exchange rates using the following economic indicators, which include the rural population and age dependency metrics we looked at earlier, as well as others that are relevant to a nation's economy:
# 
# - `Rural population percent`
# - `Age dependency`
# - `GDP (current US$)`
# - `Population growth (annual %)`
# - `Real interest rate (%)`
# - `Inflation, consumer prices (annual %)`
# - `Unemployment, total (% of total labor force) (national estimate)`
# - `Stocks traded, total value (current US$)`
# - `Merchandise trade (% of GDP)`
# - `Air transport, passengers carried`
# - `International tourism, number of arrivals`
# - `Net primary income (Net income from abroad) (current US$)`
# - `Refugee population by country or territory of origin`
# - `Foreign direct investment, net inflows (% of GDP)`
# 
# Here are the indicators plotted together.

# In[20]:


indicators = ['Rural population percent', 'Age dependency', 'GDP (current US$)', 'Population growth (annual %)', 'Real interest rate (%)', 'Inflation, consumer prices (annual %)', 'Unemployment, total (% of total labor force) (national estimate)', 'Stocks traded, total value (current US$)', 'Merchandise trade (% of GDP)', 'Air transport, passengers carried', 'International tourism, number of arrivals', 'Net primary income (Net income from abroad) (current US$)', 'Refugee population by country or territory of origin', 'Foreign direct investment, net inflows (% of GDP)']
c5 = alt.Chart(dfscaled).mark_line().encode(
    x = 'Year:O',
    y = alt.Y(alt.repeat('layer'), type='quantitative', title='Scaled Indicators'),
    color = alt.ColorDatum(alt.repeat('layer')), #.Color(scale=alt.Scale(scheme = 'category20c'))
    strokeDash = alt.StrokeDashDatum(alt.repeat('layer')),
).properties(
    title="Scaled Economic Indicators in Russia from 2006 to 2020",
    width=600
).repeat(
    layer=indicators
)
c5


# We can see that there is a significant amount of variation in these indicators' values over the 15 years. Also, notice that international tourism decreases sharply in 2020 (as expected).
# 
# We will use the years 2006-2016 as our training set and the years 2017-2020 as our test set, so that we can test the accuracy of our model on the data and be warned of possible overfitting. Since I want to use these specific rows for my training and test sets, I will use `iloc` to define them rather than `train_test_split`.
# 
# First we will attempt to use Linear Regression, in hopes that the coefficients can give us some idea of how these indicators might be related to the exchange rate.

# In[21]:


X_train = df[indicators].iloc[:11]
X_test = df[indicators].iloc[11:]
y_train = df['Ruble to Dollar'].iloc[:11]
y_test = df['Ruble to Dollar'].iloc[11:]
df_train = df.iloc[:11].copy()
df_test = df.iloc[11:].copy()
lrg = LinearRegression()
lrg.fit(X_train, y_train)


# To visually evaluate the results of Linear Regression, I will use methods adapted from the [Pandas user guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions).

# In[22]:


coefficients = pd.DataFrame(indicators, columns=["Indicators"])
coefficients["Coefficients"] = lrg.coef_
coefficients.set_index("Indicators", inplace=True)
def style_positive(v, props=''):
    return props if v >= 0 else None
def style_negative(v, props=''):
    return props if v < 0 else None
def highlight_max(s, props=''):
    return np.where(s == np.max(s.values), props, '')
def highlight_min(s, props=''):
    return np.where(s == np.min(s.values), props, '')
coefficients[["Coefficients"]].style.format({"Coefficients": '{:.2E}'})                                    .applymap(style_positive, props='color:green;')                                    .applymap(style_negative, props='color:red;')                                    .applymap(lambda v: 'opacity: 20%;' if (v < 0.000001) and (v > -0.000001) else None)                                    .apply(highlight_max, props='color:white;background-color:green', axis=0)                                    .apply(highlight_min, props='color:white;background-color:red', axis=0)


# Above, we see indicators with relatively small coefficients in low opacity, negative coefficients (i.e. negative correlation according to the model) in red, and positive coefficients in green. We also see the highlighted maximum and minimum coefficients, representing the indicator with highest positive correlation and highest negative correlation, respectively, according to the model. It appears that net inflows of foreign direct investment as a percent of GDP may be an indicator of growth in the RUB/USD exchange rate, while inflation of consumer prices may be an indicator of decline in the RUB/USD exchange rate. This seems reasonable from an economic standpoint.
# 
# Now we will attempt to gage the accuracy of this model.

# In[23]:


from sklearn.metrics import mean_absolute_error
train_error_lrg = mean_absolute_error(lrg.predict(X_train), y_train)
test_error_lrg = mean_absolute_error(lrg.predict(X_test), y_test)
train_error_lrg


# In[24]:


test_error_lrg


# In[25]:


test_error_lrg/train_error_lrg


# In[26]:


test_error_lrg/(y_test.mean())


# Notice that the test error is larger than the training error by over 6 orders of magnitude, which would suggest overfitting. 
# 
# The test error from linear regression is about 25% of the average RUB/USD value in the test set.
# 
# Now we will try a different approach: K-Nearest Neighbors Regression.

# In[27]:


from sklearn.neighbors import KNeighborsRegressor
kng3 = KNeighborsRegressor(n_neighbors=3)
kng3.fit(X_train,y_train)


# In[28]:


train_error_kng3 = mean_absolute_error(kng3.predict(X_train), y_train)
test_error_kng3 = mean_absolute_error(kng3.predict(X_test), y_test)
train_error_kng3


# In[29]:


test_error_kng3


# In[30]:


test_error_kng3/train_error_kng3


# In[31]:


test_error_kng3/y_test.mean()


# Notice that this time our training and test errors are closer in value, which suggests that we do not have a problem with overfitting. However, the test error as a percentage of the mean of the RUB/USD values is larger, about 43%.
# 
# Let us visually compare our results from both methods.

# In[32]:


df_test["Linear regression prediction"] = lrg.predict(X_test)
df_test["3-Neighbors regression prediction"] = kng3.predict(X_test)
df_test["Actual Ruble to Dollar"] = y_test
c6 = alt.Chart(df_test).mark_line().encode(
    x = 'Year:O',
    y=alt.Y(alt.repeat('layer'),
        type='quantitative',
        title='Predicted and Actual RUB/USD'
        #scale=alt.Scale(zero=False)
        ),
    color=alt.ColorDatum(alt.repeat('layer'))
).properties(
    title="Predicted vs. Actual RUB/USD Using Linear Regression, K-Neighbors Regression",
    width=500,
    height=300
).repeat(layer=["Actual Ruble to Dollar", "Linear regression prediction", "3-Neighbors regression prediction"])
c6


# It is clear that the linear regression method underestimates the RUB/USD exchange rates, while the K-Neighbors prediction is an overestimate.
# 
# Would a different value of K give a better model? We investigate using the method for plotting the train and test curves from the [course notes](https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html).

# In[33]:


def get_scores(k):
    reg = KNeighborsRegressor(n_neighbors=k)
    reg.fit(X_train, y_train)
    train_error = mean_absolute_error(reg.predict(X_train), y_train)
    test_error = mean_absolute_error(reg.predict(X_test), y_test)
    return (train_error, test_error)
df_scores = pd.DataFrame({"k":range(1,12),"train_error":np.nan,"test_error":np.nan})
for i in df_scores.index:
    df_scores.loc[i,["train_error","test_error"]] = get_scores(df_scores.loc[i,"k"])
df_scores["kinv"] = 1/df_scores.k
c7 = alt.Chart(df_scores).mark_line().encode(
    x = "kinv",
    y=alt.Y(alt.repeat('layer'),
        type='quantitative',
        title='Mean Absolute Error'
        #scale=alt.Scale(zero=False)
        ),
    color=alt.ColorDatum(alt.repeat('layer'))
).properties(
    title="Train Error and Test Error for Different Values of K",
    width=600
).repeat(layer=["train_error", "test_error"])
c7


# It appears from the above graph that 1/K = 0.25 (so K = 4) may give a slightly better prediction than the original K = 3. We will avoid K = 1, despite the seemingly accurate results above, as this will result in overfitting.
# 
# As a final step, we will observe all of these predictions together along with the RUB/USD actual rates, from 2017 to 2020:

# In[34]:


kng4 = KNeighborsRegressor(n_neighbors=4)
kng4.fit(X_train,y_train)
df_test["4-Neighbors regression prediction"] = kng4.predict(X_test)
c8 = alt.Chart(df_test).mark_line().encode(
    x = 'Year:O',
    y=alt.Y(alt.repeat('layer'),
        type='quantitative',
        title='Predicted and Actual RUB/USD'
        #scale=alt.Scale(zero=False)
        ),
    color=alt.ColorDatum(alt.repeat('layer'))
).properties(
    title="Predicted vs. Actual RUB/USD Using Linear Regression, K-Neighbors Regression",
    width=500,
    height=300
).repeat(layer=["Actual Ruble to Dollar", "Linear regression prediction", "3-Neighbors regression prediction", "4-Neighbors regression prediction"])
c8


# ## Summary
# 
# Although we had only a few years of data to work with, it was possible to gain some insights into the relationship between key economic indicators in Russia and the Ruble to U.S. Dollar exchange rates over the past 15 years. The linear regression model pointed to inflation and foreign investment as potentially significant economic indicators relating to the value of the Ruble. Additionally, predictions using linear regression and K-nearest neighbors regression appear to have some degree of usefulness, although this is limited.
# 
# There are a few potential areas where this analysis could be improved through further work. First, if more frequent data were available (quarterly, monthly, weekly, etc.), this would likely improve the accuracy of our models. Unfortunately such data did not appear to be readily available.
# 
# Additionally, the year 2020 presents a possible outlier in our data, considering how international tourism and other economic indicators were at uncharacteristic levels during this year. That being said, the year 2020 remains in this project for two reasons: the first being that removing it would constitute a 7% reduction in the number of data points, and the second being that a good model of RUB/USD exchange rates with regard to economic factors would ideally be able to accurately model the Ruble's value even in times of crisis and uncertainty.

# ## References
# 
# Bank of England (2022) - "Interest & exchange rates data." Published online at BankOfEngland.co.uk. Retrieved from: https://www.bankofengland.co.uk/boeapps/database/index.asp?first=yes&SectionRequired=I&HideNums=-1&ExtraInfo=true 
# 
# Bruce C. Dieffenbach (2014) - "Leading Economic Indicators." Published online at Albany.edu/~bd445/. Retrieved from: https://www.albany.edu/~bd445/Economics_301_Intermediate_Macroeconomics_Slides_Spring_2014/Leading_Economic_Indicators_(Print).pdf 
# 
# The Conference Board (2012) - "Description of Components." Published online at Conference-Board.org. Retrieved from: https://www.conference-board.org/data/bci/index.cfm?id=2160 
# 
# Hannah Ritchie, Edouard Mathieu, Max Roser, Bastian Herre, Joe Hasell, Esteban Ortiz-Ospina, Bobbie Macdonald, Fiona Spooner and Pablo Rosado (2022) - "War in Ukraine." Published online at OurWorldInData.org. Retrieved from: https://ourworldindata.org/ukraine-war
# 
# Max Roser, Bastian Herre and Joe Hasell (2013) - "Nuclear Weapons." Published online at OurWorldInData.org. Retrieved from: https://ourworldindata.org/nuclear-weapons
# 
# World Bank (2020) - "Russian Federation Data." Published online at Data.WorldBank.org. Retrieved from:  https://data.worldbank.org/country/russian-federation

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=13ce6a9c-2012-4cc1-9560-d7cb6e77ad1b' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
