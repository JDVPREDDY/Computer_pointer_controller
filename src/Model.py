import cv2
import os
from openvino.inference_engine import IENetwork, IECore
import logging


class Model:
    def __init__(self, model_name, device='CPU', threshold = .7, extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.mode = 'async'
        self.exec_network = None
        self.device = device
        self.request_id = 0
        self.threshold = threshold
        self.core = IECore()
        self.network = self.core.read_network(model=str(model_name),
                                              weights=str(os.path.splitext(model_name)[0] + ".bin"))
        self.input = next(iter(self.network.inputs))
        self.output = next(iter(self.network.outputs))

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.exec_network = self.core.load_network(self.network, self.device)
        return self.exec_network


    def predict(self):
        pass

    def preprocess_output(self):
        pass

    def preprocess_input(self, image):

    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        net_input_shape = self.network.inputs[self.input].shape
        ip_frame = cv2.resize(image, (net_input_shape[3], net_input_shape[2]))
        ip_frame = ip_frame.transpose(2, 0, 1)
        ip_frame = ip_frame.reshape(1, *ip_frame.shape)
        return ip_frame

    def check_model(self):
        logger = logging.getLogger()
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [layer for layer in self.network.layers.keys() if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            print("Error! check extention for the following unsupported layers =>" + str(unsupported_layers))
            logger.error('Error! check extention for the unsupported layers')
            exit(1)
        print("All layers are supported !!")
        logger.info("All layers are supported !!")

