'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import numpy as np
from Model import Model

class HPE(Model):
    '''
    Class for the Face Detection Model.
    '''
    
    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_frame = self.preprocess_input(image)
        
        self.exec_network.start_async(request_id=0,
                                      inputs={self.input: processed_frame})

        if self.mode == 'async':
            self.exec_network.requests[0].wait()
            return self.preprocess_output(self.exec_network.requests[0].outputs)
        else:
            if self.exec_network.requests[0].wait(-1) == 0:
                return self.preprocess_output(self.exec_network.requests[0].outputs)


    
    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        output = []
        output.append(outputs['angle_y_fc'].tolist()[0][0])
        output.append(outputs['angle_p_fc'].tolist()[0][0])
        output.append(outputs['angle_r_fc'].tolist()[0][0])

        return output