import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):


    def __init__(self, name):
        super().__init__()
        self.name = name 
        self.globals_dict = globals()
        self.globals_dict['nn'] = nn
        self.globals_dict['F'] = F
        self.globals_dict['self'] = self 



  


    def forward(self, x):

        return x
    


class ConvBlock(Block):

    
    def __init__(self, name):
        super().__init__(name) 

        self.conv = None 
        self.activ = None  
        self.pool = None 


        def createConv(self, params):

            """
            Creates a Convolutional layer.

            Args:
                params (dict): The possible arguments.

            Returns:
                bool:  True if the layer is created, False otherwise
            """
       
        
        # params = {"in_channels" : Any positive integer value,
        #          "out_channels" : Any positive integer value,
        #          "kernel_size" : A positive integer value k or  A tuple (kH, kW), 
        #          "stride" : A positive integer value s or  A tuple (sH, sW),
        #          "padding" : 0 (No padding) or A positive integer value p or A tuple (pH, pW),
        #          "dilation" : A positive integer value d or  A tuple (dH, dW),, 
        #          "groups" : Any positive integer value. Default is 1, # I think we can skip this 
        #          "bias" : A Boolean value. Default is True
        #          }

        # in_channels: This is the number of input channels (or features) in the input tensor. It corresponds to the depth or the number of channels in the input image.
        # out_channels: This is the number of output channels (or features) produced by the convolutional layer. Each channel represents a specific filter or kernel applied to the input.
        # kernel_size: This specifies the size of the convolutional kernel. It can be a single integer or a tuple of two integers (kH, kW) to specify different height and width dimensions.
        # stride: This parameter determines the stride of the convolution. It can be a single integer or a tuple of two integers (sH, sW) to specify different stride values for height and width.
        # padding: This controls the amount of padding added to the input tensor. It can take different forms:
        #   0: No padding
        #   n: Pad n pixels on all sides
        #   (pH, pW): Pad pH pixels to the height and pW pixels to the width
        # dilation: This paramseter controls the spacing between the kernel elements. It can be a single integer or a tuple of two integers (dH, dW) to specify different dilation values for height and width.
        # groups: This controls the connections between input and output channels. By default, it is set to 1, which means each input channel is connected to each output channel. For grouped convolution, you can set the groups parameter to a specific value.
        # bias: This is a Boolean value that determines whether to include a bias term in the convolution operation. By default, it is set to True.

            if self.conv is None:
                self.conv = nn.Conv2d(**params)
                return True
            else:
                print("The block already has a convolutional layer.")
                return False
            



    def createActiv(self, type):

        """
        Creates a activation layer.

        Args:
            type (str): The type of activation layer ["ReLU", "LeakyReLU", "Tanh"].


        Returns:
            bool:  True if the layer is added, False otherwise
        """


        if self.activ is None:
            exec(f"self.activ  = nn.{type}()", self.globals_dict)
            return True
        else:
            print("The block already has a activation layer.")
            return False
        


    def createPool(self,  params):

        """
        Creates a Pooling layer.

        Args:
            params (dict): {"type" : type,"params : params"} The type of the pooling ["max", "avg"] and The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """
        type = params["type"]
        params = params["params"]

        if type ==  "max":
            self.createMaxPool(params)
        if type ==  "avg":
            self.createAvgPool(params)
        else:
            print("Wrong type")
            return False



    def createMaxPool(self, params):

        """
        Creates a MaxPooling layer.

        Args:
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """


        # params = {"kernel_size" : A positive integer value k or  A tuple (kH, kW),
        #           "stride" : A positive integer value s or  A tuple (sH, sW),
        #          "padding" : 0 (No padding) or A positive integer value p or A tuple (pH, pW),
        #          "dilation" : A positive integer value d or  A tuple (dH, dW),
        #          "return_indices" : bool, Default is False.
        #          "ceil_mode" : bool, Default is False.
        #          }


        # kernel_size: Input data type: int or Tuple[int, int]. Specifies the size of the max pooling window. If an integer is provided, it represents a square window size. If a tuple of two integers (kH, kW) is provided, it represents different height and width dimensions for the window.
        # stride: Input data type: int or Tuple[int, int]. Specifies the stride of the max pooling operation. If an integer is provided, it represents a square stride value. If a tuple of two integers (sH, sW) is provided, it represents different stride values for height and width.
        # padding: Input data type: int or Tuple[int, int]. Specifies the amount of padding added to the input tensor. If an integer is provided, it adds the same amount of padding to all sides. If a tuple of two integers (pH, pW) is provided, it adds different padding values to the height and width dimensions.
        # dilation: Input data type: int or Tuple[int, int]. Specifies the spacing between the kernel elements during the max pooling operation. If an integer is provided, it represents a square dilation value. If a tuple of two integers (dH, dW) is provided, it represents different dilation values for height and width.
        # return_indices: Input data type: bool. Specifies whether to return the indices of the max values along with the outputs. Default is False.
        # ceil_mode: Input data type: bool. Specifies whether to use ceil mode for the output size calculation. Default is False.

        if self.pool in None:
            self.pool = nn.MaxPool2d(**params)
            return True
        
        else:
            print("The block already has a pooling layer.")
            return False

        
    

    def createAvgPool(self, params):

        """
        Creates a AvgPooling layer.

        Args:
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """


        # params = {"kernel_size" : A positive integer value k or  A tuple (kH, kW),
        #           "stride" : A positive integer value s or  A tuple (sH, sW),
        #           "padding" : 0 (No padding) or A positive integer value p or A tuple (pH, pW),
        #           "ceil_mode" : bool, Default is False,
        #           "count_include_pad" : bool Default is True
        #          }


        # kernel_size: Input data type: int or Tuple[int, int]. Specifies the size of the average pooling window. If an integer is provided, it represents a square window size. If a tuple of two integers (kH, kW) is provided, it represents different height and width dimensions for the window.
        # stride: Input data type: int or Tuple[int, int]. Specifies the stride of the average pooling operation. If an integer is provided, it represents a square stride value. If a tuple of two integers (sH, sW) is provided, it represents different stride values for height and width.
        # padding: Input data type: int or Tuple[int, int]. Specifies the amount of padding added to the input tensor. If an integer is provided, it adds the same amount of padding to all sides. If a tuple of two integers (pH, pW) is provided, it adds different padding values to the height and width dimensions.
        # ceil_mode: Input data type: bool. Specifies whether to use ceil mode for the output size calculation. Default is False.
        # count_include_pad: Input data type: bool. Specifies whether to include padding in the averaging calculation. Default is True.



        if self.pool in None:
            self.pool = nn.AvgPool2d(**params)
            return True
        
        else:
            print("The block already has a pooling layer.")
            return False
        



    def removeLayer(self, layerType):

        """
        Removes the layer.

        Args:
            layerType (str): The type of the new layer [Conv, Activ, Pool].

        Returns:
            bool:  True if the layer is removed, False otherwise
        """

        if layerType is "conv" and self.conv is not None:
            self.conv = None
            return True

        elif layerType is "activ" and self.activ is not None:
            self.activ = None
            return True

        elif layerType is "pool" and self.pool is not None:
            self.pool = None
            return True
        else:
            print("Wrong type or the layer does not exist")
            return False
        






    def changeLayerParameters(self, layerType, newParams):

        """
        Changes the parametres of the layer.

        Args:
            layerType (str): The type of the layer.
            params (dict): The new possible arguments.

        Returns:
            bool:  True if the parametres of the layer is changed, False otherwise
        """
        # We will chnage parametres of "Convolutional Layer", "MaxPooling Layer", "AvgPooling Layer"


        if self.removeLayer(layerType):
            exec( f"self.creat{layerType}({newParams}), self.globals_dict" )
            return True
        


    def freezeLayer(self, name):

        """
        Freezes the Layer.

        Args:
            name (str): The name of the layer.

        Returns:
            bool:  True if the layer is frozen, False otherwise
        """

        if self.conv is not None:
            for param in self.conv.parameters():
                param.requires_grad = False
            return True
        else:
            print("No Conv layer defined")
            return False

        
        



    def unfreezeLayer(self):

        """
        Unfreezes the Layer.

        Args:
            name (str): The name of the layer.

        Returns:
            bool:  True if the layer is unfrozen, False otherwise
        """

        if self.conv is not None:
            for param in self.conv.parameters():
                param.requires_grad = True
            return True
        
        else:
            print("No Conv layer defined")
            return False
    


    def forward(self, x):

        if self.conv is not None:
            x = self.conv(x)

        if self.activ is not None:
            x = self.activ(x)

        if self.pool is not None:
            x = self.pool(x)




        return x