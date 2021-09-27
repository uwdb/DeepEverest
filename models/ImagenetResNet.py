from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from models.BaseModel import BaseModel


class ImagenetResNet50(BaseModel):

    def __init__(self):
        optimizer = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-04)
        BaseModel.__init__(self, model=ResNet50(weights='imagenet'), optimizer=optimizer)

    def preprocess_input_for_inference(self, x):
        return preprocess_input(x)
