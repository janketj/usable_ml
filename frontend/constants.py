from backend.MessageType import MessageType

COLORS = {
    "bg-paper": "rgb(38,39,48)",
    "bg-primary": "rgb(14, 17, 23)",
    "bg-secondary": "rgb(0, 0, 23)",
    "bg-light": "rgb(30, 30, 40)",
    "bg-box": "rgb(1, 1, 90)",
    "bg-red": "rgb(91,26,26)",
    "primary": "#fff",
    "secondary": "#a8a8a8",
    "red": "#ff4b4b",
}

LAYER_NAMES = {
    "conv": "Convolutional",
    "drop": "Dropout",
    "activ": "Activation",
    "pool": "Pooling",
    "linear": "Linear",
    "norm": "Normalization",
}

USED_PARAMS = {
    "linear_bias": "bias",
    "conv_padding": "padding",
    "conv_in_channels": "input channels",
    "conv_out_channels": "output channels",
    "conv_kernel_size": "kernel size",
    "conv_stride": "stride",
    "conv_bias": "bias",
    "activ_type": "function",
    "drop_p": "probability",
    "pool_type": "type",
    "pool_kernel_size": "kernel size",
}
ACTIVATION_TYPES = ["None", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"]

POOLING_TYPES = ["max", "avg"]

BLOCK_TYPES = [
    ["Please Select", "select"],
    ["Fully Connected (linear) Block", "FCBlock"],
    ["Convolutional Block", "ConvBlock"],
]


def get_block_type(type):
    return BLOCK_TYPES[type][0]


def get_pool_type(type):
    return POOLING_TYPES[type][0]


PLACEHOLDER_ACCURACY = [
    {
        "id": "accuracy",
        "color": "#0f0",
        "data": [
            {"x": "0", "y": 39},
            {"x": "1", "y": 44},
            {"x": "2", "y": 50},
            {"x": "3", "y": 55},
            {"x": "4", "y": 56},
            {"x": "5", "y": 57},
            {"x": "6", "y": 60},
            {"x": "7", "y": 61},
            {"x": "8", "y": 62},
            {"x": "9", "y": 63},
            {"x": "10", "y": 64},
            {"x": "11", "y": 65},
            {"x": "12", "y": 66},
        ],
    }
]

PLACEHOLDER_LOSS = [
    {
        "id": "loss",
        "color": "#f00",
        "data": [
            {"x": "0", "y": 67},
            {"x": "1", "y": 60},
            {"x": "2", "y": 55},
            {"x": "3", "y": 50},
            {"x": "4", "y": 45},
            {"x": "5", "y": 44},
            {"x": "6", "y": 43},
            {"x": "7", "y": 42},
            {"x": "8", "y": 40},
            {"x": "9", "y": 39},
            {"x": "10", "y": 38},
            {"x": "11", "y": 36},
            {"x": "12", "y": 34},
        ],
    }
]

BLOCK_DEFAULT_PARAMS = {
    "linear_bias": True,
    "linear_in_features": 16,
    "linear_out_features": 32,
    "conv_padding": 2,
    "conv_in_channels": 1,
    "conv_out_channels": 8,
    "conv_kernel_size": 4,
    "conv_stride": 1,
    "conv_dilation": 1,
    "conv_bias": True,
    "use_norm_layer": False,
    "norm_num_features": 1,
    "norm_out_features": 1,
    "norm_momentum": 0.5,
    "activ_type": 1,
    "use_drop_layer": False,
    "drop_p": 0.5,
    "drop_inplace": False,
    "use_pool_layer": False,
    "pool_type": 0,
    "pool_stride": 1,
    "pool_padding": 0,
    "pool_kernel_size": 4,
}

PLACEHOLDER_MODEL = {
    "name": "Default Model",
    "id": "default",
    "blocks": [],
}


MODEL_MESSAGES = [
    MessageType.LOAD_MODEL,
    MessageType.SAVE_MODEL,
    MessageType.ADD_BLOCK,
    MessageType.EDIT_BLOCK,
    MessageType.REMOVE_BLOCK,
    MessageType.REMOVE_BLOCK_LAYER,
    MessageType.FREEZE_BLOCK_LAYER,
    MessageType.UNFREEZE_BLOCK_LAYER,
]
