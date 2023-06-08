COLORS = {
    "bg-primary": "rgb(14, 17, 23)",
    "bg-secondary": "rgb(0, 0, 23)",
    "bg-light": "rgb(30, 30, 40)",
    "bg-box": "rgb(1, 1, 90)",
    "bg-red": "rgb(91,26,26)",
    "primary": "#fff",
    "secondary": "#a8a8a8",
    "red": "#ff4b4b",
}


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


PLACEHOLDER_BLOCKS = [
    {
        "name": "Block 1",
        "type": "ConvBlock",
        "index": 1,
        "layers": [
            {
                "layerType": "Conv",
                "params": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "stride": 1,
                    "padding": 2,
                    "kernel_size": 4,
                    "dilation": 0,
                    "bias": True,
                },
            },
            {
                "layerType": "Norm",
                "params": {
                    "num_features": 32,
                    "out_features": 32,
                    "momentum": 0.1,
                    "affine": True,
                    "track_running_stats": False,
                },
            },
            {
                "layerType": "Activ",
                "params": {
                    "type": "ReLU",
                },
            },
            {
                "layerType": "Drop",
                "params": {"p": 0.05, "inplace": True},
            },
            {
                "layerType": "Pool",
                "params": {"type": "max"},
            },
        ],
    },
    {
        "name": "Block 2",
        "type": "FCBlock",
        "index": 2,
        "layers": [
            {
                "layerType": "Linear",
                "params": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "bias": True,
                },
            },
            {
                "layerType": "Norm",
                "params": {
                    "num_features": 32,
                    "out_features": 32,
                    "momentum": 0.1,
                    "affine": True,
                    "track_running_stats": False,
                },
            },
            {
                "layerType": "Activ",
                "params": {
                    "type": "ReLU",
                },
            },
            {
                "layerType": "Drop",
                "params": {"p": 0.05, "inplace": True},
            },
        ],
    },
    {
        "name": "Block 3",
        "type": "FCBlock",
        "index": 3,
        "layers": [
            {
                "layerType": "Linear",
                "params": {
                    "in_channels": 16,
                    "out_channels": 32,
                    "bias": True,
                },
            },
            {
                "layerType": "Norm",
                "params": {
                    "num_features": 32,
                    "out_features": 32,
                    "momentum": 0.1,
                    "affine": True,
                    "track_running_stats": False,
                },
            },
        ],
    },
]