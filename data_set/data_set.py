import numpy as np

from glob import glob
from audio_features.mel_spectrogram import mel_spectrogram

class DataSet:

    def __init__(self,
                 training_set_path,
                 cv_set_path,
                 test_set_path,
                 is_regression) -> None:
        
        self.x_training, self.y_training = self.__prepare_data_set_with_path(training_set_path, is_regression)
        self.x_cv, self.y_cv = self.__prepare_data_set_with_path(cv_set_path, is_regression)
        self.x_test_set, self.y_test_set = self.__prepare_data_set_with_path(test_set_path, is_regression)

    def __y_data_set_for_regression(self, filepath):
        # the value after the delimiter indicates its percentage
        delimiter_for_percent = "_"
        return int(filepath.split(delimiter_for_percent)[-1].split(".")[0]) / 100.0
    
    def __y_data_set_for_classification(self, filepath):
        # the value after the delimiter indicates its percentage
        delimiter_for_percent = "_"
        percent = int(filepath.split(delimiter_for_percent)[-1].split(".")[0])
        return 1 if (percent == 100) else 0

    @classmethod
    def audio_representation(self, filepath):
        return mel_spectrogram(file_path=filepath, shape=(128, 128))
    
    def __prepare_data_set_with_path(self, path: str, is_regression):
        y_data_set_method = self.__y_data_set_for_regression \
            if (is_regression==True) \
            else self.__y_data_set_for_classification
        list_of_sounds = glob(path)
        dataset_x = []
        dataset_y = []
        
        for filepath in list_of_sounds:
            mel_spectrogram_representation = self.audio_representation(filepath)
            dataset_x.append(mel_spectrogram_representation)
            dataset_y.append(y_data_set_method(filepath))

        return np.array(dataset_x), np.array(dataset_y)
