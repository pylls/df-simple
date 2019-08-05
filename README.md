# Simple Deep Fingerprinting
This is a simple version of the [Deep Fingerprinting (DF) attack by Sirinam et
al.](https://github.com/deep-fingerprinting/df). It is more-or-less easy to use,
and limited to the open world setting with distinct labels for monitored websites (and one
for all unmonitored websites).

Running `df_train.py` takes significant time and is ideally used with access to a decent
GPU for deep learning. The result of training is a trained model (for the model itself,
see `df_model.py`). The model is used to predict (classify) traffic traces using
`df_predict.py`. The result of prediction is a file with the resulting predicitions.
