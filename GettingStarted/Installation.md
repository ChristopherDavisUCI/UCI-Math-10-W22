# Installation (optional)

For the Winter 2022 version Math 10, because the class is starting out remote, we will primarily run Python code "in the cloud" using Deepnote.  This way you do not have to install anything on your own computers.

If you do wish to run Python code on your personal computer, here are instructions for installing Python via Miniconda.  See the [Alternatives section](subsec:alt-installation) below for more options.

Brief outline:

1.  Download and install Miniconda.
1.  Create a Conda virtual environment named `math10`.
1.  Activate that environment and then install the Python libraries we will use for this class.

## Warnings  

* If you already have Anaconda installed, don't install Miniconda; use Anaconda instead.  If you try to install both, probably neither will work correctly.
* Every computer is different, and there is no guarantee these instructions will work on your computer.  If you can't get Python installed on your own computer, you can always use the ALP lab computers.

## Instructions for Windows

<iframe width="560" height="315" src="https://www.youtube.com/embed/MTnTzlJA1To" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

* [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) and then install it.
* Search for Anaconda Prompt (not Anaconda Powershell Prompt) on your computer and then open it.
* From the Anaconda Prompt, create a new Python 3.8 environment (or any version you want from 3.7-3.10) for our class using the following command:
```
conda create --name math10 python=3.8
```
* Switch into that environment:
```
conda activate math10
```
* (This next step might take 20-30 minutes.)  Install the first set of Python libraries we will need, from the `conda-forge` channel:
```
conda install -c conda-forge ipykernel jupyter numpy pandas matplotlib seaborn altair vega_datasets scikit-learn plotly jupyter-book
```
* (This next step might take 20-30 minutes.) Install the next set of libraries we will need, from the `pytorch` channel:
```
conda install -c pytorch pytorch torchvision torchaudio
```

To check that things worked, try executing the command `jupyter notebook` from the Anaconda prompt.  It should open a page in your internet browser.  From that page, create a new Python 3 notebook from the `New` dropdown menu.  Try to execute the following commands in the notebook (execute by hitting `shift+enter`).
```
import numpy as np
np.array([3,1,4,1])
```
and
```
import torch
torch.rand((3,4))
```
If these commands work without any errors (the first should produce something like a vector and the second should produce something like a matrix), then your Python environment is working!


## Instructions for Mac

<iframe width="560" height="315" src="https://www.youtube.com/embed/FU9Ri7I9vkE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

* [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) and then install it.  I recommend downloading the `.pkg` file.  (I could not get the M1 file to work, even though I have an M1 Mac.)
* Open a new terminal window.  You should see `(base)` on the left side if Miniconda is installed correctly.  This indicates you are in the *base* Conda environment.
* From the terminal, create a new Python 3.8 environment (or any version you want from 3.7-3.10) for our class using the following command:
```
conda create --name math10 python=3.8
```
* Switch into that environment:
```
conda activate math10
```
* (This next step might take 20-30 minutes.)  Install the first set of Python libraries we will need, from the `conda-forge` channel:
```
conda install -c conda-forge ipykernel jupyter numpy pandas matplotlib seaborn altair vega_datasets scikit-learn plotly jupyter-book
```
* (This step might take 20-30 minutes.) Install the next set of libraries we will need, from the `pytorch` channel:
```
conda install -c pytorch pytorch torchvision torchaudio
```

To check that things worked, try executing the command `jupyter notebook` from the terminal prompt, still within the `math10` environment.  It should open a page in your internet browser.  From that page, create a new Python 3 notebook from the `New` dropdown menu.  Try to execute the following commands in the notebook (execute by hitting `shift+enter`).
```
import numpy as np
np.array([3,1,4,1])
```
and
```
import torch
torch.rand((3,4))
```
If these commands work without any errors (the first should produce something like a vector and the second should produce something like a matrix), then your Python environment is working!

(subsec:alt-installation)=
## Alternatives

There are some alternatives if you don't want to follow the above instructions.

* A very reasonable alternative, and what I recommended in Fall 2021, is to download Anaconda instead of Miniconda.  (You should not install both.)  Anaconda is more similar to what we would use on the lab computers.  The reason I suggest Miniconda is because that is what I personally use and because it installs less unnecessary software.