from block import *
import torch.nn as nn
import torch.nn.functional as F
import uuid


class BlockList(list):
    def __str__(self):
        elements = ', '.join(str(item) for item in self)
        return f"[{elements}]"



class Model(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.id = str(uuid.uuid4())

        self.name = name
        self.head = None
        self.blockList = BlockList()


    def __str__(self):
        dict = {"name" : self.name,
                "id" : self.id,
                "blockList" : str(self.blockList)
                }
        return str(dict)

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

        if blockInfo["pervious"] is not None:
            pervious = self.findBlockById(id = blockInfo["pervious"])
        else:
            pervious = None

        
        if blockInfo["type"] == "ConvBlock":
            block = self.createConvBlock(blockInfo["name"], pervious, blockInfo["layers"])


        if blockInfo["type"] == "FCBlock":
            block = self.createFCBlock(blockInfo["name"], pervious, blockInfo["layers"])

        if pervious is  None:
            self.assignHead(block)

        self.addBlock(block)



    def createConvBlock(self, name, pervious, layers):

        convBlock = ConvBlock(name = name, pervious = pervious)

        convBlock.createConv(layers["conv"]) 

        if layers["norm"] is not None:
            convBlock.createNorm(layers["norm"]) 

        if layers["activ"] is not None:
            convBlock.createActiv(layers["activ"]) 

        if layers["drop"] is not None:
            convBlock.createDrop(layers["drop"]) 

        if layers["pool"] is not None:
            convBlock.createPool(layers["pool"]) 

        return convBlock


    def createFCBlock(self, name, pervious, layers):

        fcBlock = FCBlock(name = name, pervious = pervious)

        fcBlock.createLinear(layers["linear"])

        if layers["norm"] is not None:
            fcBlock.createNorm(layers["norm"]) 

        if layers["activ"] is not None:
            fcBlock.createActiv(layers["activ"]) 

        if layers["drop"] is not None:
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


    def removeBlocklayer(self, dict, ):

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

        if block.pervious is  None:
            if block.next is  None:
                self.assignHead(None)

            self.assignHead(block.next)
            block.next.assignPervious(None)
        elif block.next is  None:
            block.pervious.assignNext(None)

        else:
            block.next.assignPervious(block.pervious)
        
        self.blockList.remove(block)

    def forward(self, x):
        
        x = self.head.forward(x)
        return x