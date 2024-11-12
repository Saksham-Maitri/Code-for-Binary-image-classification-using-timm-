# Code-for-Binary-image-classification-using-timm-module 

Timm module is a module that contains many CNN networks. We can use the script.py to do binary image classification by finetuning the models based on our dataset. 

This is for binary image classification 

The parent folder should look like 

Parent folder 
  - script.py // does the classification
  - list_models.py // lists the usefull models
  - train  // Training images 
  - test  // Testing images
  - val  // Validation images
  - train.csv
  - test.csv
  - val.csv

The csv files have the structure 
image_id , label 
where image_id contains the name of the image from that specific folder and label contains value (0/1)

1 : Setup 
## Installation

To install this package, run:

```bash
pip install -r req.txt
```

This will install the required modules. 

Also, its better to resized all the images to a common resolution ( ex 512*512 ) 

2 : Get the models 

run:
```bash
python list_models.py
```

This will give you the list of pretrained models from which we can select model.

3 : Run the script

run:
```bash
code script.py
```

From here we can adjust the parameters like transform, number of epochs, device( GPU/CPU ), batch_size.
Please adjust this according to your machine 

run:
```bash
python script.py
```

We can see the progress bar due to tqdm module.

We can also modify this file a bit such that it automatically tests models. For that we need to figure which models we need to test and put them into a list and iterate over the list and record the AUC and accuracy.
