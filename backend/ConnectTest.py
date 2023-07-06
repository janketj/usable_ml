from modelLinkedBlocks import Model
from block import *


model =  Model(name = "Model")











 






# Tasks


# 1: CREAT BLOCK
# Input 
blockInfo_c ={
            "type": "ConvBlock",
            "name": "Con_2",
            "previous" : None,

            "layers": {
                "conv": {
                        "in_channels": 3,
                        "out_channels": 64,
                        "kernel_size":  3,
                        "stride": 1,
                        "padding":  1,
                        "dilation": 1,
                        "groups": 1,
                        "bias": True
                },
                "norm": { # can be None
                    "num_features": 3
                },
                "activ": { # can be None
                    "type": "ReLU",
                },
                "drop": {  # can be None
                        "p": 0.05, 
                        "inplace": True
                },
                "pool": { # can be None
                        "type": "max", 
                        "params" : {
                                    "kernel_size": 2,
                                    "padding": 0,
                                    "dilation": 1,
                                    "return_indices": False,
                                    "ceil_mode": False
                        },
                },
            }
        }

model.createBlock(blockInfo_c)
print(model.getBlockList()[0].getInfo())

blockInfo_fc ={
            "type": "FCBlock",
            "name": "Block_1fc",
            "previous" : 0,

            "layers": {
                "linear": {"in_features" : 10,
                           "out_features" : 10,
                          "bias" : True,
                  },
                "norm": {
                    "num_features": 32,
                    "momentum": 0.1,
                },
                "activ": {
                    "type": "ReLU",
                },
                "drop": {"p": 0.05, "inplace": True},
            },
        }


model.createBlock(blockInfo_fc)
print(model.getBlockList()[1].getInfo())




# CHANGE
# if value is None => No change needed
dict = {"id" : 0, # no chnage
            "info" : {
                        "type": "ConvBlock", # not changable
                        "name": "NewCon_2",
                        "previous" : None, # not changable

            "layers": {
                "conv": None ,
                "norm": None ,
                "activ": { # can be None
                    "type": "Tanh",
                },
                "drop": {  # can be None
                        "p": 0.01, 
                        "inplace": True
                },
                "pool": None ,
                },
            }
        }
    



model.changeBlockParameters(dict)
print(model.getBlockList()[0].getInfo())


# Removing layer
# if value is True => Removed
dict = {"id" : 0, 
            "info" : {
                        "type": "ConvBlock",   # do not need this info
                        "name": "NewCon_2",  # do not need this info
                        "previous" : None, # do not need this info

            "layers": {
                "conv": False, # can not be removed same for linear
                "norm": False ,
                "activ": True,
                "drop": True,
                "pool": False ,
                },
            }
        }






model.removeBlocklayer(dict)
print(model.getBlockList()[0].getInfo())



# Freeze layer
# if value is True => Freeze
dict = {"id" : 0, 
            "info" : {
                        "type": "ConvBlock",   # do not need this info
                        "name": "NewCon_2",  # do not need this info
                        "previous" : None, # do not need this info

            "layers": {
                "conv": True, 
                "norm": False ,
                "activ": False,
                "drop": False,
                "pool": False ,
                },
            }
        }





model.freezeBlocklayer(dict)

model.unfreezeBlocklayer(dict)


for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable layer: {name}")





dict = {"id" : 1 }

model.deleteBlock(dict)
print(model)

