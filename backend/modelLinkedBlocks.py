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
            "out_channels": 16,
            "kernel_size": 6,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "bias": True,
        },
        "activ": {  # can be None
            "type": "ReLU",
        },
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

DEFAULT_FC_BLOCK1 = {
    "type": "FCBlock",
    "name": "Block_1fc",
    "previous": "Con_2",
    "next": "Block_2fc",
    "layers": {
        "linear": {
            "out_features": 64,
            "bias": True,
        },
        "activ": {
            "type": "ReLU",
        },
    },
}

DEFAULT_FC_BLOCK2 = {
    "type": "FCBlock",
    "name": "Block_2fc",
    "previous": "Block_1fc",
    "layers": {
        "linear": {
            "out_features": 10,
            "bias": True,
        }
    },
}


class BlockList(list):
    def to_list(self):
        return [item.getInfo() for item in self]

    def to_layers(self):
        conv_layers = []
        linear_layers = []
        for block in self:
            if isinstance(block, ConvBlock):
                conv_layers += block.to_layer_list()
            else:
                linear_layers += block.to_layer_list()
        return nn.Sequential(*conv_layers), nn.Sequential(*linear_layers)


class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()

        self.id = name  # str(uuid.uuid4())

        self.name = name
        self.blockList = BlockList()
        """ self.createBlock(DEFAULT_CONV_BLOCK)
        self.createBlock(DEFAULT_FC_BLOCK1)
        self.createBlock(DEFAULT_FC_BLOCK2)
        self.convolutional_layers, self.linear_layers = self.blockList.to_layers() """
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(512, 32),
            nn.Tanh(),
            nn.Linear(32, 10),
        )

    def to_dict(self):
        return dict(name=self.name, id=self.id, blocks=self.blockList.to_list())

    def addBlock(self, block):
        if block.previous:
            prevIndex = self.findBlockIndex(block.previous, block.previous)
            self.blockList.insert(prevIndex + 1, block)
        else:
            self.blockList.append(block)

    def assignHead(self, block):
        self.head = block

    def getBlockList(self):
        return self.blockList

    def findBlockById(self, id, name=None):
        if name is not None:
            for block in self.blockList:
                if block.name == name:
                    return block
        for block in self.blockList:
            if block.getId() == id or block.name == id:
                return block

        print("Wrong Id", id, name)
        return None

    def findBlockIndex(self, id, name=None):
        if name is not None:
            for index, block in enumerate(self.blockList):
                if block.name == name:
                    return index
        for index, block in enumerate(self.blockList):
            if block.id == id:
                return index
        return 0

    def createBlock(self, blockInfo):
        previousBlock = None
        if blockInfo["previous"] is not None:
            previousBlock = self.findBlockById(
                id=blockInfo["previous"], name=blockInfo["previous"]
            )

        if blockInfo["type"] == "ConvBlock":
            block = self.createConvBlock(
                blockInfo["name"], previousBlock, blockInfo["layers"]
            )

        if blockInfo["type"] == "FCBlock":
            block = self.createFCBlock(
                blockInfo["name"], previousBlock, blockInfo["layers"]
            )

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

    def changeBlockParameters(self, params):
        info = params["info"]
        block = self.findBlockById(params["id"], params["name"])

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
        block = self.findBlockById(dict["id"], dict["name"])
        block_index = self.findBlockIndex(dict["id"], dict["name"])
        if block.previous is not None:
            previous_block = self.findBlockById(block.previous, block.previous)
            previous_block.next = block.next
        if block.next is not None:
            next_block = self.findBlockById(block.next, block.next)
            next_block.previous = block.previous

        self.blockList.pop(block_index)

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
