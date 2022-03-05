#!/usr/bin/env python
# coding: utf-8

# # Homework 4
# 
# **Remark**: This might not reflect updates, so check the Deepnote file for the official version.
# 
# List your name and the names of any collaborators at the top of this notebook.
# 
# (Reminder: It’s encouraged to work together; you can even submit the exact same homework as another student or two students, but you must list each other’s names at the top.)

# ## Overview
# 
# This project includes a csv file `unemployment.csv` which includes US unemployment rates for various industries between 2000 and 2010.  Load that csv file using `pd.read_csv` and save the resulting DataFrame as `df`.
# 
# The background question for this homework is,
# * How does the time of year affect unemployment in different industries?

# ## Extracting the month from the date
# 
# The DataFrame includes a "date" column, whose entries are strings.  Use each of the following methods to extract the month (represented as an integer from 1 to 12) from the date.
# 
# 1.  If we wanted to get the year from a string like `s = '2000-01-01 08:00:00+00:00'`, we could use `s[0:4]`.  Adapt that strategy to get a pandas Series containing the month for each row, using `map` and a `lambda` function.  Convert the entries from strings to integers using `int`.
# 
# 2.  Another option is to use `s.split("-")`.  Use that strategy, again together with `map`, a `lambda` function, and `int`.
# 
# 3.  A more sophisticated option is to, as a first step, convert the `dtype` of the "date" column using `to_datetime`.  (In other words, replace what's in the "date" column so they are no longer strings; this doesn't require `map`.)  If `t` is one of the entries in the resulting column, then you can use `t.month` to get the month.  Again use `map` and a `lambda` function.
# 
# 4.  The best method is probably to avoid using `map` completely.  Convert the column again using `to_datetime`, and then adapt one of the methods from the [pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.html) to convert each entry in the series into a month number.
# 
# 5.  Using that last method, create a new column in the in the DataFrame containing these months.  Name that column "month".
# 
# If everything went correctly, the `dtype` of the "date" column should now be `datetime64[ns, UTC]`, and the `dtype` of the "month" column should be `int64`.

# ## Plotting the data 1
# 
# Use Altair and `mark_circle` to make a scatter plot of the data, placing "date" along the x-axis, "rate" along the y-axis, and coloring the points using "industry".  Include a tooltip showing the "month" and "industry".

# ## Plotting the data 2
# 
# Adapt the [stacked bar chart](https://altair-viz.github.io/gallery/stacked_bar_chart.html) example from the Altair documentation to make a stacked bar chart of the data.  Use "month" for the x-axis, "mean(rate)" for the y-axis, and again "industry" for the color.  Specify an [encoding data type](https://altair-viz.github.io/user_guide/encoding.html#encoding-data-types) for the "month" column so that the plot looks better.  (What is the appropriate data type for the discrete set 1, 2, ..., 11, 12?)

# ## Plotting the data 3
# 
# Let `c1` denote the stacked bar chart you made in the previous part.
# 
# Create an Altair selection object using the following code:
# ```
# choice = alt.selection_multi(fields=['industry'], bind='legend')
# ```
# 
# Add that to `c1` using `.add_selection`.
# 
# You have already specified x, y, and color.  Also specify `opacity` for `c1` by using
# ```
# opacity=alt.condition(choice, alt.value(1), alt.value(0.2)),
# ```
# 
# Display `c1` and try clicking on one of the entries in the legend.  The selected entry should be emphasized.  You can also select multiple industries by holding down shift while you click.

# ## Plotting the data 4
# 
# Create a new Altair Chart `c2` using the following code.
# 
# ```
# c2 = alt.Chart(df).mark_bar().encode(
#     x='month:O',
#     y='mean(rate)',
#     color='industry',
# ).transform_filter(choice)
# ```
# 
# Then display both `c1` and `c2` side-by-side using `c1|c2`.  Now if you click on the legend, `c2` should only show that industry.

# ## Interpreting the results
# 
# What is an example of an industry that seems to have higher unemployment in the winter months?  What is an example of an industry that seems to have higher unemployment in the summer months?  (You should answer this question using the Altair charts, but if you want extra practice with pandas, you can also try to use pandas to answer it.)

# ## Submission
# Download the .ipynb file for this notebook (click on the folder icon to the left, then the … next to the file name) and upload the file on Canvas.
