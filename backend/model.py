import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, name, model_id):
        super().__init__()

        self.name = name
        self.model_id = model_id

        self.globals_dict = globals()
        self.globals_dict['nn'] = nn
        self.globals_dict['F'] = F
        self.globals_dict['self'] = self


        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)


    #add layer: convolutional, linear, pooling ...

    def addLayer(self, type, name, params):

        """
        Adds a new layer.

        Args:
            type (str): The type of the new layer from this list ["Convolutional Layer", "MaxPooling Layer", "AvgPooling Layer", "Fully Connected Layer",  "Activation Layer", "Normalization Layer", "Dropout Layer"].
            name (str): The name of the new layer.
            params (dict): The possible arguments.

        Returns:
            bool:  True if the new layer is added, False otherwise
        """

        layerType = {
            "Convolutional Layer": "Conv2d",
            "MaxPooling Layer": "MaxPool2d",
            "AvgPooling Layer": "AvgPool2d",
            "Fully Connected Layer": "Linear",
            "Activation Layer": "Act",
            "Normalization Layer": "BatchNorm2d",
            "Dropout Layer": "Dropout"
        }

        if " " in type:
            type =  layerType[type]


        self.globals_dict['params'] = params

        if type == "Conv2d":
            return self.createConv(name)


        if type == "MaxPool2d":
            return self.createMaxPool(name)


        if type == "AvgPool2d":
            return self.createAvgPool(name)

        if type == "Linear":
            return self.createLinear(name)

        if type == "Act":
            return self.createActivation(name)

        if type == "BatchNorm2d":
            return self.createNorm(name)


        if type == "Dropout":
            return self.createDrop(name)




    def createConv(self, name):

        """
        Creates a Convolutional layer.

        Args:
            name (str): The name of the layer.
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """
        # All options
        # nn.Conv1d: 1D convolutional layer.
        # nn.Conv2d: 2D convolutional layer. Our!
        # nn.Conv3d: 3D convolutional layer.
        # nn.ConvTranspose2d: 2D transposed convolutional layer (also known as deconvolution).



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



        exec(f"self.{name} = nn.Conv2d(**params)", self.globals_dict)


        # for now I assume that all params are correct and the layer is created
        return True



    def createMaxPool(self, name):

        """
        Creates a MaxPooling layer.

        Args:
            name (str): The name of the layer.
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """

        # Pooling Layers:
        # nn.MaxPool1d: 1D max pooling layer.
        # nn.MaxPool2d: 2D max pooling layer.
        # nn.MaxPool3d: 3D max pooling layer.


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


        exec(f"self.{name} = nn.MaxPool2d(**params)", self.globals_dict)
        # for now I assume that all params are correct and the layer is created
        return True


    def createAvgPool(self, name):

        """
        Creates a AvgPooling layer.

        Args:
            name (str): The name of the layer.
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """

        # Pooling Layers:
        # nn.AvgPool1d: 1D average pooling layer.
        # nn.AvgPool2d: 2D average pooling layer.
        # nn.AvgPool3d: 3D average pooling layer.


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

        exec(f"self.{name} = nn.AvgPool2d(**params)", self.globals_dict)

        # for now I assume that all params are correct and the layer is created
        return True


    def createLinear(self, name):

        """
        Creates a fully connected layer.

        Args:
            name (str): The name of the layer.
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


        exec(f"self.{name} = nn.Linear(**params)", self.globals_dict)

        # for now I assume that all params are correct and the layer is created
        return True



    def createActivation(self, name):

        """
        Creates a activation layer.

        Args:
            name (str): The name of activation layer ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"].
            params (dict):  None.

        Returns:
            bool:  True if the layer is added, False otherwise
        """


        if not self.hasLayer(name):

            exec(f"self.{name} = nn.{name}()", self.globals_dict)
        # for now I assume that all params are correct and the layer is created
        return True



    def createNorm(self, name):

        """
        Creates a normalization Layer.

        Args:
            name (str): The name of the layer.
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """

        # Normalization Layers:
        # nn.BatchNorm1d: Batch normalization layer for 1D data.
        # nn.BatchNorm2d: Batch normalization layer for 2D data.
        # nn.BatchNorm3d: Batch normalization layer for 3D data.

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


        exec(f"self.{name} = nn.BatchNorm2d(**params)", self.globals_dict)

        # for now I assume that all params are correct and the layer is created
        return True

    def createDrop(self, name):

        """
        Creates a dropout layer.

        Args:
            name (str): The name of the layer.
            params (dict): The possible arguments.

        Returns:
            bool:  True if the layer is added, False otherwise
        """



        # params = {"p" : A floating-point value between 0 and 1 (around 0.5 ),
        #           "inplace" : A boolean value ,
        #          }


        # p (float): The probability of an element to be zeroed. It must be a value between 0 and 1.
        # inplace (bool): If set to True, the operation is performed in-place, i.e., it modifies the input tensor.


        exec(f"self.{name} = nn.Dropout(**params)", self.globals_dict)

        # for now I assume that all params are correct and the layer is created
        return True


    def removeLayer(self, name):
        """
        Removes the layer.

        Args:
            name (str): The name of the new layer.

        Returns:
            bool:  True if the layer is removed, False otherwise
        """

        if hasattr(self, name):
            delattr(self, name)
            return True

        else:
            print(f"No layer with name {name} exists.")
            return False



    def freezeLayer(self, name):

        """
        Freezes the Layer.

        Args:
            name (str): The name of the layer.

        Returns:
            bool:  True if the layer is frozen, False otherwise
        """

        layer = getattr(self,  name)
        for param in layer.parameters():
            param.requires_grad = False

        return True

    def unfreezeLayer(self, name):

        """
        Unfreezes the Layer.

        Args:
            name (str): The name of the layer.

        Returns:
            bool:  True if the layer is unfrozen, False otherwise
        """

        layer = getattr(self, name)
        for param in layer.parameters():
            param.requires_grad = True

        return True

    def changeLayerParameters(self, name, newParams):

        """
        Changes the parametres of the layer.

        Args:
            name (str): The name of the layer.
            params (dict): The new possible arguments.

        Returns:
            bool:  True if the parametres of the layer is changed, False otherwise
        """
        # We will chnage parametres of "Convolutional Layer", "MaxPooling Layer", "AvgPooling Layer", "Fully Connected Layer", "Normalization Layer", "Dropout Layer"
        typeOfLayer = type(getattr(self, name)).__name__

        if self.removeLayer(name):
            self.addLayer(typeOfLayer, name, newParams)



        # print(layer)
        # exec(f"self.{name}.load_state_dict({newParams})", self.globals_dict)
        # print(layer)
        # layer.load_state_dict(newParams)

        return True


    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = con_new
        return F.relu(self.conv2(x))





    def hasLayer(self, name):
        return name in [name for name, module in self.named_modules()]
