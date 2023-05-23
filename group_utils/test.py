from model import Model


myModel = Model()

# ["Convolutional Layer", "MaxPooling Layer", "AvgPooling Layer", "Fully Connected Layer",  "Activation Layer", "Normalization Layer", "Dropout Layer"]


## Convolutional Layer
type = "Convolutional Layer"
name = "newConv"
params = {
    "in_channels": 3,
    "out_channels": 64,
    "kernel_size": (3, 3),
    "stride": (1, 1),
    "padding": (1, 1),
    "dilation": (1, 1),
    "groups": 1,
    "bias": True
}

myModel.addLayer(type, name, params)
print(name, myModel.hasLayer(name))

## MaxPooling Layer
type = "MaxPooling Layer"
name = "newMaxPool"
params = {
    "kernel_size": 2,
    "padding": 0,
    "dilation": 1,
    "return_indices": False,
    "ceil_mode": False
}

myModel.addLayer(type, name, params)
print(name, myModel.hasLayer(name))



## AvgPooling Layer
type = "AvgPooling Layer"
name = "newAvgPool"
params = {
    "kernel_size" : 3
}

myModel.addLayer(type, name, params)
print(name, myModel.hasLayer(name))



## Fully Connected Layer
type = "Fully Connected Layer"
name = "newFC"
params = {
    "in_features": 256,
    "out_features": 128,
    "bias": True
}

myModel.addLayer(type, name, params)
print(name, myModel.hasLayer(name))


## Activation Layer

type = "Activation Layer"
name = "Tanh"
params = None

myModel.addLayer(type, name, params)
print(name, myModel.hasLayer(name))



#Normalization Layer
type = "Normalization Layer"
name = "newNorm"
params = {
    "num_features": 3

}

myModel.addLayer(type, name, params)
print(name, myModel.hasLayer(name))


# Dropout Layer
type = "Dropout Layer"
name = "newDrop"
params = {


}

myModel.addLayer(type, name, params)
print(name, myModel.hasLayer(name))


print("Done adding a layer!")
print("_________________________________")

name = "newNorm"
myModel.removeLayer(name)
print(name, not myModel.hasLayer(name))
print("Done removing a layer!")
print("_________________________________")



name = "newConv"
newParams = {
    "in_channels": 3,
    "out_channels": 30,
    "kernel_size": (2, 2),
    "stride": (1, 1),
    "padding": (1, 1),
    "dilation": (1, 1),
    "groups": 1,
    "bias": True
}
print(getattr(myModel, name))
myModel.changeLayerParameters(name, newParams)
print(getattr(myModel, name))
print("Done changinga layer param!")
print("_________________________________")



name = "newConv"
print("Updated requires_grad:", getattr(myModel, name).weight.requires_grad)
myModel.freezeLayer(name)
print("Updated requires_grad:", getattr(myModel, name).weight.requires_grad)
print("Done freezing layer param!")
print("_________________________________")



name = "newConv"
print("Updated requires_grad:", getattr(myModel, name).weight.requires_grad)
myModel.unfreezeLayer(name)
print("Updated requires_grad:", getattr(myModel, name).weight.requires_grad)
print("Done freezing layer param!")
print("_________________________________")