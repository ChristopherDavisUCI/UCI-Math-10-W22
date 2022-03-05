# Week 2 Videos

## list vs tuple vs set vs range

<iframe width="560" height="315" src="https://www.youtube.com/embed/A_-oGyhfeg0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This video explains some of the advantages and disadvantages among the Python data types list, tuple, set, range.

## list vs NumPy array

<iframe width="560" height="315" src="https://www.youtube.com/embed/aqPr34hGhG8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Similar to the previous video, but for the data type Numpy array.  This data type is defined in the NumPy library, so we first need to get access to that library using `import numpy as np`.

NumPy arrays are extremely important.  Even at times when it seems like we are not using them, they may be getting used in the background.

Many operations can be performed much faster with NumPy arrays than with Python lists.

## Dictionaries and pandas Series

<iframe width="560" height="315" src="https://www.youtube.com/embed/iQqslErllcw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Python dictionaries and pandas Series have many similarities.  The `dict` data type in Python is the more fundamental data type, but we will use pandas Series more often in Math 10.  For example, when we import a dataset, each column in that imported dataset will be a pandas Series.

## Random simulation using NumPy

<iframe width="560" height="315" src="https://www.youtube.com/embed/Mf52Tcn44XY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

If you took my Math 9 class, this type of computation should feel familiar to you.  We use random simulations to estimate a probability.  The key formula we use is that the probability is estimated as "number of successes" divided by "number of experiments".

Pay attention to how the computations are done using NumPy.  Several important techniques are introduced in this video:
* accessing a column of a matrix;
* finding which rows in a matrix satisfy a given condition;
* counting how many rows in a matrix satisfy a given condition. 