import numpy as np
import librosa
import tensorflow as tf

def mel_spectrogram(file_path, shape) -> np.ndarray:
    y, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # Note: expanding the dimension to the Mel Spectrogram before resize,
    # since image resize needs 3 dimensions.
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = tf.image.resize(mel_spectrogram, shape)
    return mel_spectrogram
