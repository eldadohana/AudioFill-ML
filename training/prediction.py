from data_set.data_set import DataSet
import tensorflow as tf


def sound_for_prediction(filepath):
    mel_spectrogram_representation = DataSet.audio_representation(filepath)
    mel_spectrogram_representation = tf.reshape(mel_spectrogram_representation, (1,) + (128,128) + (1,))
    return mel_spectrogram_representation
