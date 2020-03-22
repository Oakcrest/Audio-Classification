# Audio-Classification
classify .wav files into multi-class - There are a few options to perform this task: 
1. convert .wav files into spectrogram files and use 2D CNN to train the multi-class classifier model
2. extract speech/melody features from .wav files and use XGB, deep learning, or others to build traditional models
3. possible combination of the 1 and 2. For instance, the "mfcc" values are nothing but a windowed array like "sprectrogram", so it can be classfied by 2D CNN as in 1, instead of being aggregated into feature vectors and then trained
