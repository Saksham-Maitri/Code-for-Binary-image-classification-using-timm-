# This file can be executed in google colab
# For google colab, do !pip install timm and then run this code

import timm

all = []
for i in timm.list_models():
    model = timm.create_model(i, pretrained=True)
    all.append(i)

print(i)
