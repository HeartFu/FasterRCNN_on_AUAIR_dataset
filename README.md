### Overview

This project is INM705's coursework which is implemented by Pytorch and Python. The network is Faster-RCNN.

### Installation

In this project, many libraries are used. The list is:
1. Pytorch and Torchvision: A framework of deep learning.
2. tqdm: A processing bar.
3. opencv: A tool library for loading images.
4. matplotlib: A tool library for drawing chart.

In this project, we use Python 3.7.4 and Pytorch 1.3.1.

In addition to Python, Pytorch and Torchvision, other libraries can install by this command:
```bash
pip install -r requirements.txt
```

### Dataset

We use the AU-AIR dataset which can be downloaded at this [websit](https://bozcani.github.io/auairdataset#download) or [google drive]() which contain the training and testing data after splitting.

### Train

Before training, there are three variables which are needed to change on *_main.py file.

original_dir: It is the dataset path which you download.

train_dir: It is the training path where you want to put the train data after splitting. If you download dataset from the google drive, this is the train path on dataset.

test_dir: It is the testing path where you want to put the test data after splitting. If you download dataset from the google drive, this is the test path on dataset.

There are three files to connect with our models. 

main.py: It is script for training the first model and second model. 

train_mymodel_main.py: It is the script for training the third model.

train_relation_module_main.py: It is the script for training the Faster-RCNN model with relation module.

The running command is:

```bash
python *_main.py
```

### Evaluation

We provided one ipynb file for evaluation and visualization. Run the jupyter file and follow the instruction for this part.