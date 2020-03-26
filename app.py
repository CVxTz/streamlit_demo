import time

import numpy as np
import streamlit as st
from keras.datasets import mnist
import plotly.express as px

from models import get_model_mlp, get_model_cnn


@st.cache
def get_data(flatten=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if flatten:
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np.eye(10)[y_train, :]
    y_test = np.eye(10)[y_test, :]

    return x_train, y_train, x_test, y_test


progress_bar = st.progress(0)
status_text = st.empty()
plotly_chart = st.empty()


option = st.sidebar.selectbox(
    'Which architecture do you want to use ?',
    ("None", "MLP", "CNN"))

x_train, y_train, x_test, y_test = get_data(flatten=(option == "MLP"))

if option == "MLP":
    model = get_model_mlp()
else:
    model = get_model_cnn()

if option in ("MLP", "CNN"):

    accuracies = []

    for i in range(20):
        progress_bar.progress(5 * (i+1))

        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=1)

        acc = float(history.history['val_acc'][0])
        accuracies.append(acc)

        fig = px.line(x=list(range(len(accuracies))), y=accuracies, title="Val accuracy")
        fig.update_xaxes(range=[-1, 21])
        fig.update_yaxes(range=[accuracies[0]-0.1, 1])

        plotly_chart.plotly_chart(fig)


        time.sleep(0.1)

    status_text.text('Done!')

    st.balloons()
