'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import os
import numpy as np
import logging as log
from Model import Model


class FaceDetectionModel(Model):
    '''
    Class for the Face Detection Model.
    '''
    
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_input = self.preprocess_input(image)
        self.exec_network.start_async(request_id=self.request_id,
                                      inputs={self.input: processed_input})

        if self.mode == 'async':
            self.exec_network.requests[self.request_id].wait()
            result = self.exec_network.requests[self.request_id].outputs[self.output]
            face, box = self.preprocess_output(result[0][0], image)
            return face, box
        else:
            if self.exec_network.requests[self.request_id].wait(-1) == 0:
                result = self.exec_network.requests[self.request_id].outputs[self.output]
                face, box = self.preprocess_output(result[0][0], image)
                return face, box


    
    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        area = []
        cords = []
        for id, label, confidence, x_min, y_min, x_max, y_max in outputs:
            if confidence > self.threshold:
                width = x_max - x_min
                height = y_max - y_min
                area.append(width * height)
                cords.append([x_min, y_min, x_max, y_max])
        
        if len(area) > 0:
            box = cords[int(np.argmax(area))]
            if box is None:
                return None, None

            h, w = image.shape[0:2]
            box = box * np.array([w, h, w, h])
            box = box.astype(np.int32)
            x_min, y_min, x_max, y_max = box
            cropped_face = image[y_min:y_max, x_min:x_max]
            return cropped_face, box
        else:
            return None, None
