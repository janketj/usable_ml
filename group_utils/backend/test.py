from modelLinkedBlocks import Model
from block import *


model = Model()

blockConv = ConvBlock("Block 1",  None)

## Convolutional Layer
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

print(blockConv.createConv( params ))
print("Convolutional Layer \n", blockConv.conv)
print()
## MaxPooling Layer

params = {
    "kernel_size": 2,
    "padding": 0,
    "dilation": 1,
    "return_indices": False,
    "ceil_mode": False
}
params = {"type": "max", "params" : params}


print(blockConv.createPool( params ))
print("MaxPooling Layer \n", blockConv.pool)
print()

## AvgPooling Layer
params = {
    "kernel_size" : 3
}
params = {"type": "avg", "params" : params}

print(blockConv.createPool( params ))
print("AvgPooling Layer \n", blockConv.pool)
print()


## Activation Layer

type = "Tanh"

print(blockConv.createActiv( type ))
print("Activation Layer Layer \n", blockConv.activ)
print()



#Normalization Layer

params = {
    "num_features": 3

}

print(blockConv.createNorm( params ))
print("Normalization Layer \n", blockConv.norm)
print()


# Dropout Layer
type = "Dropout Layer"
name = "newDrop"
params = {}

print(blockConv.createDrop( params ))
print("Dropout Layer \n", blockConv.drop)
print()

model.head = blockConv




full = FCBlock("new", None)
print("Fully con Layer \n", full.linear)

full_2 = FCBlock("new",blockConv)
print("Done_____________!")

