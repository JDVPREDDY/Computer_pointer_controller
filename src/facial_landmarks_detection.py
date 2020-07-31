'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import numpy as np
from Model import Model

class FacialLandmarks(Model):
    '''
    Class for the Face Detection Model.
    '''
    
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_input = self.preprocess_input(image)
        self.exec_network.start_async(request_id=0,
                                      inputs={self.input: processed_input})
        if self.mode == 'async':
            self.exec_network.requests[0].wait()
            result = self.exec_network.requests[0].outputs[self.output]
            return self.preprocess_output(result, image, eye_surrounding_area)

        else:
            if self.exec_network.requests[0].wait(-1) == 0:
                result = self.exec_network.requests[0].outputs[self.output]
                return self.preprocess_output(result, image, eye_surrounding_area)

    
    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        leye_x = outputs[0][0].tolist()[0][0]
        leye_y = outputs[0][1].tolist()[0][0]
        reye_x = outputs[0][2].tolist()[0][0]
        reye_y = outputs[0][3].tolist()[0][0]

        box = (leye_x, leye_y, reye_x, reye_y)

        h, w = image.shape[0:2]
        box = box * np.array([w, h, w, h])
        box = box.astype(np.int32)

        (lefteye_x, lefteye_y, righteye_x, righteye_y) = box
        
        le_xmin = lefteye_x - eye_surrounding_area
        le_ymin = lefteye_y - eye_surrounding_area
        le_xmax = lefteye_x + eye_surrounding_area
        le_ymax = lefteye_y + eye_surrounding_area

        re_xmin = righteye_x - eye_surrounding_area
        re_ymin = righteye_y - eye_surrounding_area
        re_xmax = righteye_x + eye_surrounding_area
        re_ymax = righteye_y + eye_surrounding_area

        left_eye = image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin, le_ymin, le_xmax, le_ymax], [re_xmin, re_ymin, re_xmax, re_ymax]]
        
        return left_eye, right_eye, eye_coords