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

ACTIVATION_TYPES = ["None", "ReLU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"]
POOLING_TYPES = ["max", "avg"]
BLOCK_TYPES = ["FCBlock", "ConvBlock"]

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
    "in_channels": 16,
    "out_channels": 32,
    "Linear_bias": True,
    "Conv_padding": 2,
    "Conv_kernel_size": 4,
    "Conv_stride": 1,
    "Conv_dilation": 1,
    "Conv_bias": True,
    "use_Norm_layer": False,
    "Norm_num_features": 1,
    "Norm_out_features": 1,
    "Norm_momentum": 0.5,
    "Activ_type": 1,
    "use_Drop_layer": False,
    "Drop_p": 0.5,
    "Drop_inplace": False,
    "use_Pool_layer": False,
    "Pool_type": "max",
    "Pool_stride": 1,
    "Pool_padding": 2,
    "Pool_kernel_size": 4,
}

PLACEHOLDER_MODEL = {
    "name": "PLACEHOLDER",
    "model_id": "tets",
    "blocks": [],
}
