import timm

all = []
for i in timm.list_models():
    model = timm.create_model(i, pretrained=True)
    all.append(i)

print(i)