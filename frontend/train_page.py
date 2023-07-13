import streamlit as st
from streamlit_elements import mui, nivo
from functions import update_params
from parameter import global_parameter
from constants import PLACEHOLDER_ACCURACY, PLACEHOLDER_LOSS, COLORS
from nivo_timechart import nivo_timechart


def train_page():
    with mui.Box(
        sx={
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "minHeight": 500,
            "display": "flex",
            "justifyContent": "space-evenly",
        }
    ):
        with mui.Box(
            sx={
                "width": "40%",
                "padding": "8px",
                "borderRadius": "8px",
                "background": COLORS["bg-light"],
            }
        ):
            mui.Typography(
                "Global Training Parameters",
                sx={
                    "width": "100%",
                },
            )
            with mui.Stack(direction="column", sx={"paddingTop": "8px"}, spacing="8px"):
                global_parameter(
                    "Epochs",
                    "epochs",
                    type="slider",
                    sliderRange=[1, 30],
                    subtitle="How many times to iterate over the whole dataset",
                    tooltip="In each epoch, the neural network trains over all samples in the training dataset once. \
                        For the task of predicting handwritten digits, you should not need much more than 10 epochs.\
                        You can see in the graphs to the right if the network is still learning something new. When the accuracy curve starts flattening\
                        or even decreasing again, you should stop the training.",
                )
                global_parameter(
                    "Batch Size",
                    "batch_size",
                    type="slider",
                    sliderRange=[4, 512],
                    step=4,
                    subtitle="How many samples to use for each step",
                    tooltip="For each epoch the neural network trains on all training samples once. To introduce a little bit \
                        of randomness to this process, the samples are fed to it in batches. So if the batch size is 64, \
                        the network will average the update over those 64 samples instead of one single one. This way, \
                        the descent of the loss might be a bit wiggly in the high-dimensional space which increases the chance \
                        of finding not just a local but a global optimum",
                )
                global_parameter(
                    "Optimizer",
                    "optimizer",
                    options=[
                        {"label": "Stochastic Gradient Descent", "key": "SGD"},
                        {"label": "Adam Optimizer", "key": "Adam"},
                    ],
                    type="select",
                    subtitle="Which function to use to update the weights and biases",
                    tooltip="There are many ways one can update the weights and biases of the neural network after each time it has seen some data.\
                        All versions are effectively trying to go down the gradient of the loss function, meaning they are trying to find a direction in the\
                        high-dimensional parameter space where the loss decreases the fastest.\
                        The standard version of Stochastic Gradient Descent, or short SGD, updates model parameters based on the gradient of the loss \
                        function computed on a small randomly selected mini subset of the current batch. It takes small steps in random directions to \
                        find the optimal solution. Adam optimizer, on the other hand, adapts the learning rate (step size) for each parameter based on \
                        both the gradient direction and the historical magnitudes of the gradients. It adjusts the steps by considering the gradients' \
                        history, allowing for faster and more effective convergence in complex optimization problems.",
                )
                global_parameter(
                    "Learning Rate",
                    "learning_rate",
                    type="slider",
                    sliderRange=[0.01, 1],
                    step=0.02,
                    subtitle="How much of the current batch should be remembered",
                    tooltip="In each iteration of a neural network the weights or all neurons in all hidden layers are updated in the direction, \
                        which decreases the prediction error by the optimizer. If the learning rate is low, the weights of the neural network \
                        are only changed very little in each step. This is good if there is a lot of data and you don't want the network to overfit to a few samples. \
                        Overfitting means that it can predict those samples well but will fail in a more general case. For example if the current \
                        batch contains mostly fives, it might only learn to predict dives well. If you set the learning rate higher, the network learns faster \
                        but with the risk of overfitting or not finding an optimum.",
                )
                mui.Box(
                    mui.Typography(
                        "The parameters are only updated once you click here:",
                        sx={"fontSize": 14, "color": COLORS["secondary"]},
                    ),
                    mui.Button(
                        mui.Typography(
                            "Update Parameters", sx={"width": "80%"}
                        ),
                        endIcon=mui.icon.Send(),
                        onClick=update_params,
                        variant="outlined",
                        sx={"width": 300},
                    ),
                    sx={"display": "flex", "justifyContent": "space-between", "padding": "16px"},
                )

        with mui.Stack(
            sx={"width": "60%", "maxHeight": 500, "height": 500, "margin": "16px"}
        ):
            acc = st.session_state.vis_data["accuracy"]
            loss = st.session_state.vis_data["loss"]
            mui.Box(
                mui.Typography("Real Time Accuracy"),
                mui.IconButton(
                    mui.icon.Info(),
                    title="The accuracy is computed on the test dataset. It is the percentage of test samples for which the prediction was correct.",
                ),
                sx={"display": "flex", "justifyContent": "space-between"},
            )
            nivo_timechart(acc, "Accuracy", 300, margin=50)
            mui.Box(
                mui.Typography("Real Time Loss"),
                mui.IconButton(
                    mui.icon.Info(),
                    title="The loss refers to a measure of how well the network's predictions align with the true values or labels of the training data. \
                        It is the difference between the actual labels and the predicted labels. For the handwritten digits dataset MNIST the loss should \
                        be initially between 2 and 4 and can come very close to 0 as training goes on.".replace("  ",""),
                ),
                sx={"display": "flex", "justifyContent": "space-between"},
            )
            nivo_timechart(loss, "Loss", 200, margin=60, maxY=5)
