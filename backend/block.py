import torch.nn as nn
import torch.nn.functional as F
import math


class Block(nn.Module):


    def __init__(self, name, next):
        super().__init__()

        self.name = name 

        self.next = None
        self.pervious = None

        if next is not None:
            self.assigeNext(next)


        self.norm = None 
        self.activ = None  
        self.drop = None

        self.normParamas = None
        self.activType = None
        self.dropParamas = None
        

        self.globals_dict = globals()
        self.globals_dict['nn'] = nn
        self.globals_dict['F'] = F
        self.globals_dict['self'] = self 

    def assigeNext(self, next):
        self.next = next
        print(self.next)
        self.next.pervious = self
        

    def deleteNext(self):
        self.next.pervious = None
        self.next = None
        



    def createActiv(self, type):

        """
        Creates a activation layer.

        Args:
            type (str): The type of activation layer ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"].


        Returns:
            bool:  True if the layer is added, False otherwise
        """


        if self.activ is None:
            exec(f"self.activ  = nn.{type}()", self.globals_dict)
            self.activType = type
            return True
        else:
            print("The block already has a activation layer.")
            return False
        
    def createNorm(self, params):

        """
        Creates a normalization Layer.

        Args:
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """

        # params = {"num_features" : A positive integer value,
        #           "out_features" : A positive floating-point value, usually a small value like 1e-5,
        #           "momentum" : A floating-point value between 0 and 1, typically around 0.1,
        #           "affine" : A boolean value,
        #           "track_running_stats" : A boolean value
        #          }


        # num_features (int): The number of channels in the input.
        # eps (float): A value added to the denominator for numerical stability.
        # momentum (float): The value used for the running mean and variance computation.
        # affine (bool): If set to True, learnable affine parameters are applied.
        # track_running_stats (bool): If set to True, running estimates are used during inference.

        if self.norm is None:
            self.norm  = nn.BatchNorm2d(**params)
            self.normParamas = params
            return True
        else:
            print("The block already has a normalization layer.")
            return False

    def createDrop(self, params):

        """
        Creates a dropout layer.

        Args:
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """



        # params = {"p" : A floating-point value between 0 and 1 (around 0.5 ),
        #           "inplace" : A boolean value ,
        #          }


        # p (float): The probability of an element to be zeroed. It must be a value between 0 and 1.
        # inplace (bool): If set to True, the operation is performed in-place, i.e., it modifies the input tensor.

        if self.drop is None and self.next is None:
            self.drop  = nn.Dropout(**params)
            self.dropParamas = params
            return True
        else:
            print("The block already has a dropout layer or it is final block.")
            return False


    def removeLayer(self, layerType):

        """
        Removes the layer.

        Args:
            layerType (str): The type of the new layer [Conv, Activ, Pool].

        Returns:
            bool:  True if the layer is removed, False otherwise
        """
        # conv,  activ,  pool, linear, drop, norm
        layerType = getattr(self, layerType)

        if  layerType is None:
            setattr(self, layerType, None)
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
        # We will chnage parametres of "Convolutional Layer", "MaxPooling Layer", "AvgPooling Layer", "Linear Layer"


        if self.removeLayer(layerType):
            # createConv, createPool, createLinear, createNorm, createDrop
            # conv,  pool, linear, norm, drop
            layerClass = "create" + layerType.capitalize()
            exec(f"self.{layerType}  = self.{layerClass}({newParams}*)", self.globals_dict)
            return True
        
    def freezeLayer(self, layerType):

        """
        Freezes the Layer.

        Args:
            layerType (str): The type of the layer.

        Returns:
            bool:  True if the layer is frozen, False otherwise
        """
        # conv, linear

        if hasattr(self, layerType):
            layer = getattr(self, layerType)
            for param in layer.parameters():
                param.requires_grad = False
            return True
        else:
            print("No Conv layer defined")
            return False
        



    def unfreezeLayer(self, layerType):

        """
        Unfreezes the Layer.

        Args:
            layerType (str): The type of the layer.

        Returns:
            bool:  True if the layer is unfrozen, False otherwise
        """

        if hasattr(self, layerType):
            layer = getattr(self, layerType)
            for param in layer.parameters():
                param.requires_grad = True
            return True
        else:
            print("No Conv layer defined")
            return False
        

  


    def forward(self, x):

        return x
    


class ConvBlock(Block):

    
    def __init__(self, name, next):
        super().__init__(name,  next)
        
        self.conv = None
        self.pool = None 


        self.convParamas = None
        self.poolParamas = None 

        self.outputDim = None

        #dummy params
        params = {"in_channels" : None, "out_channels" : 3, "kernel_size" : (3, 3)} 
        self.createConv(params)

         


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

            if self.pervious is not None:
                params["in_channels"] = self.pervious.outputDim[0]
            else:
                params["in_channels"] = 1

            self.conv = nn.Conv2d(**params)
            self.convParamas = params
            self.mutateOutputDim()
            return True
        else:
            print("The block already has a convolutional layer.")
            return False
        


    def createPool(self,  params):

        """
        Creates a Pooling layer.

        Args:
            params (dict): {"type" : type,"params : params"} The type of the pooling ["max", "avg"] and The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """

        self.poolParamas = params
        type = params["type"]
        params = params["params"]

        if type ==  "max":
            return self.createMaxPool(params)
        elif type ==  "avg":
            return self.createAvgPool(params)
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

        if self.pool is None:
            self.pool = nn.MaxPool2d(**params)
            self.mutateOutputDim()
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



        if self.pool is None:
            self.pool = nn.AvgPool2d(**params)
            self.mutateOutputDim()
            return True
        
        else:
            print("The block already has a pooling layer.")
            return False
        
    def mutateOutputDim(self):

        # Input dimensions
        if self.pervious is not None:
            inputWidth, inputHeight, _  = self.pervious.outputDim
        else:
            inputWidth, inputHeight, _  = (28, 28, 1)


        # Convolutional layer parameters
        kernel_size = self.conv.kernel_size
        stride = self.conv.stride
        padding = self.conv.padding
        num_filters = self.conv.out_channels

        
        # Calculate output size for the convolutional layer
        outWidth = math.floor((inputWidth - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
        outHeight = math.floor((inputHeight - kernel_size[1]+ 2 * padding[1]) / stride[1]) + 1
        outChannels = num_filters


        # Pooling layer parameters
        if self.pool is not None:
            Pool_size = self.pool.kernel_size
            Pool_stride = self.pool.stride

            # Calculate output size for the pooling layer
            outWidth = math.floor((outWidth - Pool_size) / Pool_stride) + 1
            outHeight = math.floor((outHeight - Pool_size) / Pool_stride) + 1

        self.outputDim = (outWidth, outHeight, outChannels)




    def forward(self, x):

        if self.conv is not None:
            x = self.conv(x)

            if self.norm  is not None:
                x = self.norm(x)

            if self.activ is not None:
                x = self.activ(x)

            if self.drop is not None:
                x = self.drop(x)

            if self.pool is not None:
                x = self.pool(x)

        if self.next is not None:
            x = self.next(x)
    
        return x
    

class FCBlock(Block):

    def __init__(self, name, next):
        super().__init__(name, next) 

        self.linear = None 

        self.outputDim = None

        self.linearParams = None 

        #dummy params
        params = {"in_features" : 10, "out_features" : 10}
        self.createLinear(params)




    def createLinear(self, params):

        """
        Creates a fully connected layer.

        Args:
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """

        # params = {"in_features" : A positive integer value,
        #           "out_features" : A positive integer value,
        #           "bias" : A Boolean value. Default is True,
        #          }


        # in_features: Data type: int. Specifies the size of each input sample. It represents the number of input features.
        # out_features: Data type: int. Specifies the size of each output sample. It represents the number of output features.
        # bias: Data type: bool, optional. Specifies whether to include a bias term in the linear transformation. Default is True. If set to False, the layer will not learn an additive bias.
        

        if self.linear is None:
                if self.pervious is not None:
                    if isinstance(self.pervious , ConvBlock):
                        params["in_features"]  = self.pervious.outputDim[0] * self.pervious.outputDim[1] * self.pervious.outputDim[2]
                    elif isinstance(self.pervious , FCBlock):
                       params["in_features"]  = self.pervious.outputDim

                if self.next is  None:
                    params["out_features"]  = 10


                self.linear = nn.Linear(**params)
                self.linearParams = params
                self.mutateOutputDim()
                return True
        else:
            print("The block already has a fully conected linear layer.")
            return False
        
    def assigeNext(self, next):
        if isinstance(next , FCBlock):
            return super().assigeNext(next)
        else:
            print("After a fully connected block, you are not allowed to have a Conv layer")

    
    def deleteNext(self):
        super().deleteNext()
        self.linearParams["out_features"]  = 10
        self.changeLayerParameters(self, "linear", self.linearParams)
        return super().deleteNext()

    def mutateOutputDim(self):

        output = self.linear.out_features

        self.outputDim =  (output)
        


    def forward(self, x):

        if self.linear is not None:
            x = self.liner(x)

            if self.norm  is not None:
                x = self.norm(x)

            if self.activ is not None:
                x = self.activ(x)

            if self.drop is not None:
                x = self.drop(x)

        if self.next is not None:
            x = self.next(x)
    
        return x
            
    