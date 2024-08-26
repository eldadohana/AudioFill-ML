
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from data_set.data_set import DataSet
from pathlib import Path

class TrainingPlan:
    def __init__(self, 
                 name: str,
                 create_model: any,
                 loss: str, 
                 epochs: int,
                 batch_size: int, 
                 create_optimizer,
                 optimizer_value: float,
                 metrics: list,
                 verbose: int,
                 active: int = 1) -> None:
        self.name = name
        self.create_model = create_model
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.create_optimizer = create_optimizer
        self.optimizer_value = optimizer_value
        self.metrics = metrics
        self.verbose = verbose
        self.active = active

    def description(self):
        return self.name \
            + " with epochs: " \
            + str(self.epochs) \
            + ", optimizer value: " \
            + str(self.optimizer_value) \
            + " and batch size: " \
            + str(self.batch_size)

class Model:
    model: any
    history: any

    def __init__(self,
                 training_plan: TrainingPlan,
                 data_set: DataSet) -> None:
        self.training_plan = training_plan
        self.data_set = data_set

    def train(self):
        tf.random.set_seed(42)
        input_shape = self.data_set.x_training.shape[1:]
        optimizer = self.training_plan.create_optimizer(self.training_plan.optimizer_value)
        loss = self.training_plan.loss
        self.model = self.training_plan.create_model(input_shape)
        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=self.training_plan.metrics)
        self.history = self.model.fit(self.data_set.x_training,
                                      self.data_set.y_training,
                                      epochs=self.training_plan.epochs,
                                      batch_size=self.training_plan.batch_size,
                                      validation_data=(self.data_set.x_cv, self.data_set.y_cv),
                                      verbose=self.training_plan.verbose)
    
    def retrain(self, epochs):
        self.model.fit(self.data_set.x_training,
                       self.data_set.y_training,
                       epochs=epochs,
                       batch_size=self.training_plan.batch_size,
                       validation_data=(self.data_set.x_cv, self.data_set.y_cv),
                       verbose=self.training_plan.verbose)
        
    def predict(self, input: np.array):
        return self.model.predict(input)

    def evaluate(self, x_test=None, y_test=None, batch_Size=None):
        if x_test==None or y_test==None:
            x_test = self.data_set.x_test_set
            y_test = self.data_set.y_test_set

        if batch_Size==None:
            batch_Size = self.training_plan.batch_size

        self.model.evaluate(x_test, y_test, batch_Size)

    def save_lite_model(self, model_name):
        converter = tf.lite.TFLiteConverter.from_keras_model(model=self.model)
        tf_model = converter.convert()
        fileWithPath = "saved_models/"+model_name+".tflite"
        output_file = Path(fileWithPath)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        open(fileWithPath,"wb").write(tf_model)

    def description(self) -> str:
        return self.training_plan.name \
            + "_epochs_" \
            + str(self.training_plan.epochs) \
            + "_batch_size_" \
            + str(self.training_plan.batch_size)

    def training_results(self) -> Figure:
        metrics = ["loss"]
        if len(self.training_plan.metrics) > 0:
            metrics += self.training_plan.metrics
        
        return self.plotForMetrics(metrics)
    
    def plotForMetrics(self, metrics):
        number_of_metrics = len(metrics)
        fig, axs = plt.subplots(number_of_metrics)
        fig.suptitle(self.training_plan.description() + " - " + ",".join(metrics))
        for i in range(number_of_metrics):
            plt.ylim(0, 1)
            metric_name = metrics[i]
            ax = axs if number_of_metrics==1 else axs[i]
            ax.plot(self.history.history[metric_name])
            ax.plot(self.history.history["val_"+metric_name])
            ax.set_ylabel(metric_name)
            ax.set_xlabel("epoch")
            ax.legend(["train", "test"], loc="upper left")
        plt.close(fig)
        return fig
