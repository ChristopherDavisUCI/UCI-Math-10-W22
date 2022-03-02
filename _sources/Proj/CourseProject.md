# Course Project

## Introduction
For the final project, you will share a Deepnote project analyzing a dataset of your choice.

## Logistics
The following are requirements to receive a passing score on the course project.
* Due date: Monday, March 14th 2022, 11:59pm California time.
* This is an individual project (not a group project).
* Use the *Project Template* available in the *Project* category on Deepnote.
* Share the project with both Chris and Thurman on Deepnote.  (This has the same deadline as the overall project.)
* For the submission on Canvas, upload the ipynb file.
* The primary focus of the project must be on something involving data, and primarily using one or more datasets that weren't covered in Math 10.  (You can either use datasets that are for example built in to PyTorch or Seaborn, or you can upload the dataset to Deepnote yourself.)
* The project should clearly build on the Math 10 material.  If you're an expert in Python material that was not covered in Math 10, you are of course welcome to use that, but the project should use the topics of Math 10 in an essential way (see the rubric below).
* Anything that is taken from another source (either the idea for the project or a piece of code that is longer than a few lines, even if you edit that code) should be explicitly mentioned in the **References** section.  (For example, you could write, "The configuration of the Altair chart was adapted from \[link\].").
## Rubric
The course project is worth 20% of your course grade, and we will grade the project out of 40 total points.
1. (Clarity, 12 points) Is the project clear and well-organized?  Does it use good coding style (for example, avoiding unnecessary for loops)?  Does it explore one or more clearly described datasets, and is it clear where those datasets came from?  Use text and markdown throughout the project to help the reader understand what is going on.  Is the reasoning clear? (It's fine if you have negative results like, "so there is no clear connection between these variables".  In fact, I prefer those sorts of results as opposed to unbelievable claims.)
1. (Machine Learning, 10 points) Does the project explore the data using a variety of tools from scikit-learn and/or PyTorch?  (You can use both scikit-learn and PyTorch, but that's not required.  Using one of them is enough.)  Does the project refer to essential aspects of data analysis, such as over-fitting or the importance of a test set?
1. (pandas, 6 points) Does the project make essential use of pandas to explore the data?  Using pandas to read and store the data is not sufficient.  It should also be used either to clean the data, or to analyze the data.
1. (Altair, 6 points) Does the project include a variety of interesting charts, made using Altair?  Do we learn something about the data from these charts?
1. (Extra, 6 points) Does the project include material that was not covered in Math 10?  (This could include different libraries, different machine learning algorithms, or deeper use of the libraries we covered in Math 10.  Here are [some possibilities](ExtraTopics), but you're very encouraged to choose your own.)
## Advice
Here is some general advice about the project.
* Don't repeat the same technique multiple times unless it's really essential.  For example, having one interesting chart and a different less interesting chart is better than having the same interesting chart made for two different datasets.
* Don't spend too much time searching for the perfect dataset or the perfect idea.  Having a great dataset is less important for the project than exploring the dataset in an interesting way.
* Keep your statements reasonable.  It's perfectly fine to conclude something like, "Therefore, we did not find a connection between A and B."
* What I most want from your project is to see what you've learned in Math 10.  If you were already an expert on Python before the class and you write a project based on different material, that probably won't get a good grade, even if it's very advanced.
* Include many markdown cells (possibly very short, just one sentence) explaining what you are doing.
* I already wrote it above but it's worth repeating.  Please err on the side of referencing everything.  If your project is based on an idea you saw somewhere else, that's totally fine, but include a clear link in your project to the original source.
## Frequently asked questions
Is there a length requirement?
* There is no specific length requirement, but as a rough estimate, spending approximately 20 productive hours on the project would be a good amount.  (The word "productive" is important.  Time spent browsing tutorials or datasets is not productive in this sense if you don't end up using that material.)

What should I focus on?
* The primary content should be from one or more of the Math 10 tools (NumPy, pandas, Altair, scikit-learn, PyTorch).  If you like pandas much more than Altair, for example, then it's okay to go more in-depth in the use of pandas, and less in-depth in the use of Altair.

Do I need to get original research results?
* No!  If you explore the data in an interesting way, but can't find any interesting conclusions, that's fine.  In fact, I prefer that to making claims that the data does not support.

What if a lot of my work was done cleaning the data, but it does not appear in the project?
* Include those cleaning steps in the project! (If the work was done outside of Python, like in Excel, then that portion will not count.)

Can I use a different plotting library?
* You need to use Altair for the Altair portion of the rubric.  You can definitely use a different library (like Seaborn, Plotly, or Matplotlib) for the extra part of the rubric.