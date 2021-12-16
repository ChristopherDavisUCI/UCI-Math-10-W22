# Installation and Configuration

**Warning**.  Every computer is different, and there is no guarantee these instructions will work on your computer.  If you can't get Python installed on your own computer, you can use the ALP lab computers.

**Installing Miniconda**

I'll describe two different ways to use Python on your computer.  You should probably **only follow one** of these two options.

1. Use Miniconda.  This is what I currently use (in December 2021). I like that it is faster and more lightweight than Anaconda, but I found it confusing when I first started, and for years I used Anaconda Navigator instead.<br>Installing Miniconda on Windows.<br>Installing Miniconda on Mac.

2. Use Anaconda Navigator.  This option is easier and probably better for beginners, but the software takes up more space and is slower to load, etc.  The lab computers already have Anaconda Navigator installed.<br>Installing Anaconda Navigator (on Windows or Mac)



**Configuration**

We need to install a number of Python libraries (more may be added later).  From a terminal, run the following.

```
conda install -c conda-forge ipykernel jupyter numpy pandas matplotlib seaborn altair vega_datasets scikit-learn plotly jupyter-book
conda install -c pytorch pytorch torchvision torchaudio
```

The `-c conda-forge` indicates that the libraries should be downloaded from the conda-forge channel.  Similarly, the `-c pytorch` downloads from a different channel.  (Make sure you type `pytorch` twice, the first pytorch indicates where to download from, and the second indicates to download the PyTorch library.)

When you are done, you can close the terminal and then click the `Update index` button in Anaconda.  You should see the many libraries we just downloaded.

## Mac

Download Miniconda.  