from block import *
import torch.nn as nn
import torch.nn.functional as F
import uuid

DEFAULT_CONV_BLOCK1 = {
    "type": "ConvBlock",
    "name": "Convolution_1",
    "previous": None,
    "next": "Convolution_2",
    "layers": {
        "conv": {
            "in_channels": 1,
            "out_channels": 16,
            "kernel_size": 8,
            "stride": 2,
            "padding": 2,
            "dilation": 1,
            "groups": 1,
            "bias": True,
        },
        "activ": {  # can be None
            "type": "Tanh",
        },
        "pool": {  # can be None
            "type": "max",
            "params": {
                "kernel_size": 2,
                "padding": 0,
                "dilation": 1,
                "return_indices": False,
                "ceil_mode": False,
            },
        },
    },
}

DEFAULT_CONV_BLOCK2 = {
    "type": "ConvBlock",
    "name": "Convolution_2",
    "previous": "Convolution_1",
    "next": "Linear_3",
    "layers": {
        "conv": {
            "in_channels": 16,
            "out_channels": 32,
            "kernel_size": 4,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "groups": 1,
            "bias": True,
        },
        "activ": {  # can be None
            "type": "Tanh",
        },
        "pool": {  # can be None
            "type": "max",
            "params": {
                "kernel_size": 2,
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
    "name": "Linear_3",
    "previous": "Convolution_2",
    "next": "Linear_4",
    "layers": {
        "linear": {
            "in_features": 512,
            "out_features": 32,
            "bias": True,
        },
        "activ": {
            "type": "Tanh",
        },
    },
}

DEFAULT_FC_BLOCK2 = {
    "type": "FCBlock",
    "name": "Linear_4",
    "previous": "Linear_3",
    "layers": {
        "linear": {
            "in_features": 32,
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
        if name == "default":
            self.createBlock(DEFAULT_CONV_BLOCK1)
            self.createBlock(DEFAULT_CONV_BLOCK2)
            self.createBlock(DEFAULT_FC_BLOCK1)
            self.createBlock(DEFAULT_FC_BLOCK2)
        self.convolutional_layers, self.linear_layers = self.blockList.to_layers()

    def to_dict_state(self):
        """
        Convert the model to a dictionary representation.

        Returns:
            The dictionary representation of the model.
        """
        return {
            "name": self.name,
            "id": self.id,
            "blockList": self.blockList.to_list(),
            "convolutional_layers": self.convolutional_layers,
            "linear_layers": self.linear_layers,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_dict(cls, model_dict):
        """
        Create a model instance from a dictionary representation.

        Args:
            model_dict: The dictionary representation of the model.

        Returns:
            The created model instance.
        """
        model = cls(model_dict["name"])
        model.id = model_dict["id"]
        model.blockList = BlockList()
        model.convolutional_layers = model_dict["convolutional_layers"]
        model.linear_layers = model_dict["linear_layers"]
        model.load_state_dict(model_dict["state_dict"])

        for block_info in model_dict["blockList"]:
            model.createBlock(block_info)

        return model

    def to_dict(self):
        return dict(name=self.name, id=self.id, blocks=self.blockList.to_list())

    def addBlock(self, block):
        if block.previous:
            prevIndex = self.findBlockIndex(block.previous, block.previous)
            self.blockList.insert(prevIndex + 1, block)
        else:
            self.blockList.insert(0, block)
        self.convolutional_layers, self.linear_layers = self.blockList.to_layers()

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
        blockId = blockInfo["id"] if "id" in blockInfo else None
        if blockInfo["previous"] is not None:
            previousBlock = self.findBlockById(
                id=blockInfo["previous"], name=blockInfo["previous"]
            )

        if blockInfo["type"] == "ConvBlock":
            block = self.createConvBlock(
                blockInfo["name"], previousBlock, blockInfo["layers"], id=blockId
            )

        if blockInfo["type"] == "FCBlock":
            block = self.createFCBlock(
                blockInfo["name"], previousBlock, blockInfo["layers"], id=blockId
            )

        self.addBlock(block)

    def createConvBlock(self, name, previous, layers, id=None):
        convBlock = ConvBlock(name=name, previous=previous, id=id)

        convBlock.createConv(layers["conv"])

        if "norm" in layers and layers["norm"] is not None:
            convBlock.createNorm(layers["norm"])

        if "activ" in layers and layers["activ"] is not None:
            convBlock.createActiv(layers["activ"])

        if "drop" in layers and layers["drop"] is not None:
            convBlock.createDrop(layers["drop"])

        if "pool" in layers and layers["pool"] is not None:
            convBlock.createPool(layers["pool"])

        return convBlock

    def createFCBlock(self, name, previous, layers, id=None):
        fcBlock = FCBlock(name=name, previous=previous, id=id)

        fcBlock.createLinear(layers["linear"])

        if "norm" in layers and layers["norm"] is not None:
            fcBlock.createNorm(layers["norm"])

        if "activ" in layers and layers["activ"] is not None:
            fcBlock.createActiv(layers["activ"])

        if "drop" in layers and layers["drop"] is not None:
            fcBlock.createDrop(layers["drop"])

        return fcBlock

    def changeBlockParameters(self, params):
        block = self.findBlockById(params["id"], params["name"])

        for layer_type, layer_params in params["layers"].items():
            if layer_params is not None:
                block.changeLayerParameters(layer_type, layer_params)
            else:
                block.removeLayer(layer_type)
        self.convolutional_layers, self.linear_layers = self.blockList.to_layers()

    def freezeBlocklayer(self, params):
        block = self.findBlockById(params["id"])
        if "conv" in params["layers"]:
            block.freezeLayer("conv")
        if "linear" in params["layers"]:
            block.freezeLayer("linear")

    def unfreezeBlocklayer(self, params):
        block = self.findBlockById(params["id"])
        if "conv" in params["layers"]:
            block.unfreezeLayer("conv")
        if "linear" in params["layers"]:
            block.unfreezeLayer("linear")

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
        self.convolutional_layers, self.linear_layers = self.blockList.to_layers()

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
