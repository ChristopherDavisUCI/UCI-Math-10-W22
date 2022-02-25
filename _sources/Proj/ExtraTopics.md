# Possible extra topics

One of the rubric items for the course project is to include something "extra" that wasn't covered in Math 10.  Here are a few possibilities.  It's even better if you find your own extra topic; it can be anything in Python that interests you.

More possibilities will be added as I think of them.

## Different Python libraries

If you want to use a Python library that isn't by default installed in Deepnote, you can install it yourself within Deepnote, using a line of code like the following, which installs the `vega_datasets` library.  Notice the exclamation point at the beginning (which probably won't appear in the documentation you find for the library).
```
!pip install vega_datasets
```



## Kaggle

Browse [Kaggle](www.kaggle.com).  Go to a competition or dataset you find interesting, and then click on the *Code* tab near the top.  You will reach a page like this one about [Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist/code).  You can browse through the Kaggle notebooks for ideas.

## pandas groupby

A very useful tool in Math 10, which unfortunately we did not cover this quarter, is `groupby`, which gives a way to break a DataFrame up into different groups.  Here are examples from the pandas [user guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html).

## pandas styler

![pandas styler](../images/styler.png)

See these examples in the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions).  This provides a way to highlight certain cells in a pandas DataFrame, and is good practice using `apply` and `applymap`.


## Random forests in scikit-learn

[Random forests](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees). This is maybe the machine learning method I see most often in Kaggle competitions.

## Principal Component Analysis in scikit-learn

![faces with pca](../images/pca.png)

[Principal component analysis](https://scikit-learn.org/stable/modules/decomposition.html#pca).  We saw clustering as our only example of unsupervised learning.  Another type of unsupervised learning is *dimensionality reduction*.  PCA is a famous example.

## PyTorch extras

* Try some other optimizers (especially `Adam`) or loss functions, or go into more details about the ones we used.  (What is `momentum in stochastic gradient descent?  Why is it useful?  How does Softmax work?  How does Log Likelihood work?)
* Instead of a fully connected neural network, like what we did in class, try to make a *convolutional* neural network.

## More Machine Learning options

I don't know much about these, but some very popular tools in Machine learning include the following.  (Just getting some of them to run in Deepnote could already be impressive, I haven't tried.)

* [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)

## Other libraries
Here are a few other libraries that you might find interesting.  (Most of these are already installed in Deepnote.)
* [sympy](https://www.sympy.org/en/index.html) for symbolic computation, like what you did in Math 9 using Mathematica.
* [Pillow](https://pillow.readthedocs.io/en/stable/index.html) for image processing.
* [re](https://docs.python.org/3/library/re.html) for advanced string methods using regular expressions.
* [Seaborn](https://seaborn.pydata.org/) It's like a cross between Altair and Matplotlib.
* [Plotly](https://plotly.com/python/plotly-express/) It's like a cross between Altair and Streamlit.
* [Keras](https://keras.io/) The main rival to PyTorch for neural networks.