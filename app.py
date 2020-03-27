import time

import numpy as np
import streamlit as st
from keras.datasets import mnist
import plotly.express as px

from models import get_model_mlp, get_model_cnn


colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
]


@st.cache
def get_data(flatten=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if flatten:
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
    else:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = np.eye(10)[y_train, :]
    y_test = np.eye(10)[y_test, :]

    return x_train, y_train, x_test, y_test


progress_bar = st.progress(0)
status_text = st.empty()
plotly_chart_1 = st.empty()
plotly_chart_2 = st.empty()


option = st.sidebar.selectbox(
    "Which architecture do you want to use ?", ("None", "MLP", "CNN")
)

x_train, y_train, x_test, y_test = get_data(flatten=(option == "MLP"))

if option == "MLP":
    model, model_aux = get_model_mlp()
else:
    model, model_aux = get_model_cnn()

if option in ("MLP", "CNN"):

    accuracies = []

    for i in range(20):
        progress_bar.progress(5 * (i + 1))

        history = model.fit(
            x_train, y_train, validation_data=(x_test, y_test), nb_epoch=1
        )

        acc = float(history.history["val_acc"][0])
        accuracies.append(acc)

        fig_1 = px.line(
            x=list(range(len(accuracies))),
            y=accuracies,
            title="Validation Accuracy vs Epoch",
            labels={"x": "Epoch", "y": "Accuracy"},
        )
        fig_1.update_xaxes(range=[-1, 21])
        fig_1.update_yaxes(range=[accuracies[0] - 0.1, 1])

        plotly_chart_1.plotly_chart(fig_1)

        test_repr = model_aux.predict(x_test)
        test_label = np.argmax(y_test, axis=-1)

        fig_2 = px.scatter(
            x=test_repr[:, 0],
            y=test_repr[:, 1],
            title="Test Sample representation",
            labels={"x": "Dim 1", "y": "Dim 2"},
            color=[
                str(a) for a in test_label.tolist()
            ],  # [colors[a] for a in test_label.tolist()]
        )

        plotly_chart_2.plotly_chart(fig_2)

        time.sleep(0.1)

    status_text.text("Done!")

    st.balloons()
