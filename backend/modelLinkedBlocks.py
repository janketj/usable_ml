from block import *
import torch.nn as nn
import torch.nn.functional as F
import uuid

DEFAULT_CONV_BLOCK = {
    "type": "ConvBlock",
    "name": "Con_2",
    "previous": None,
    "layers": {
        "conv": {
            "in_channels": 1,
            "out_channels": 64,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "bias": True,
        },
        "activ": {  # can be None
            "type": "ReLU",
        },
        "drop": {"p": 0.05, "inplace": True},  # can be None
        "pool": {  # can be None
            "type": "max",
            "params": {
                "kernel_size": 4,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False,
            },
        },
    },
}

DEFAULT_FC_BLOCK = {
    "type": "FCBlock",
    "name": "Block_1fc",
    "previous": 0,
    "layers": {
        "linear": {
            "in_features": 28,
            "out_features": 10,
            "bias": True,
        },
        "activ": {
            "type": "ReLU",
        },
        "drop": {"p": 0.05, "inplace": True},
    },
}


class BlockList(list):
    def to_list(self):
        return [item.getInfo() for item in self]


class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()

        self.id = name #str(uuid.uuid4())

        self.name = name
        self.head = None
        self.blockList = BlockList()
        self.createBlock(DEFAULT_CONV_BLOCK)
        self.createBlock(DEFAULT_FC_BLOCK)

    def to_dict(self):
        return dict(name=self.name, id=self.id, blocks=self.blockList.to_list())

    def addBlock(self, block):
        self.blockList.append(block)

    def assignHead(self, block):
        self.head = block

    def getBlockList(self):
        return self.blockList

    def findBlockById(self, id):
        for block in self.blockList:
            if block.getId() == id:
                return block

        print("Wrong Id")
        return None

    def createBlock(self, blockInfo):
        if blockInfo["previous"] is not None:
            previous = self.findBlockById(id=blockInfo["previous"])
        else:
            previous = None

        if blockInfo["type"] == "ConvBlock":
            block = self.createConvBlock(
                blockInfo["name"], previous, blockInfo["layers"]
            )

        if blockInfo["type"] == "FCBlock":
            block = self.createFCBlock(blockInfo["name"], previous, blockInfo["layers"])

        if previous is None:
            self.assignHead(block)

        self.addBlock(block)

    def createConvBlock(self, name, previous, layers):
        convBlock = ConvBlock(name=name, previous=previous)

        convBlock.createConv(layers["conv"])

        if "norm" in layers:
            convBlock.createNorm(layers["norm"])

        if "activ" in layers:
            convBlock.createActiv(layers["activ"])

        if "drop" in layers:
            convBlock.createDrop(layers["drop"])

        if "pool" in layers:
            convBlock.createPool(layers["pool"])

        return convBlock

    def createFCBlock(self, name, previous, layers):
        fcBlock = FCBlock(name=name, previous=previous)

        fcBlock.createLinear(layers["linear"])

        if "norm" in layers:
            fcBlock.createNorm(layers["norm"])

        if "activ" in layers:
            fcBlock.createActiv(layers["activ"])

        if "drop" in layers:
            fcBlock.createDrop(layers["drop"])

        return fcBlock

    def changeBlockParameters(self, dict):
        info = dict["info"]
        block = self.findBlockById(dict["id"])

        if info["name"] is not None:
            block.changeName(info["name"])

        for type, params in info["layers"].items():
            if params is not None:
                block.changeLayerParameters(type, params)

    def removeBlocklayer(
        self,
        dict,
    ):
        info = dict["info"]
        block = self.findBlockById(dict["id"])

        for type, removed in info["layers"].items():
            if removed:
                block.removeLayer(type)

    def freezeBlocklayer(self, dict):
        info = dict["info"]
        block = self.findBlockById(dict["id"])

        for type, freeze in info["layers"].items():
            if freeze:
                block.freezeLayer(type)

    def unfreezeBlocklayer(self, dict):
        info = dict["info"]
        block = self.findBlockById(dict["id"])

        for type, unfreeze in info["layers"].items():
            if unfreeze:
                block.unfreezeLayer(type)

    def deleteBlock(self, dict):
        block = self.findBlockById(dict["id"])

        if block.previous is None:
            if block.next is None:
                self.assignHead(None)

            self.assignHead(block.next)
            block.next.assignPrevious(None)
        elif block.next is None:
            block.previous.assignNext(None)

        else:
            block.next.assignPrevious(block.previous)

        self.blockList.remove(block)

    def forward(self, x):
        x = self.head.forward(x)
        return x
