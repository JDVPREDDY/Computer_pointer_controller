'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import math
from Model import Model


class GazeEstimation(Model):
    '''
    Class for the Face Detection Model.
    '''
    
    def predict(self, left_eye, right_eye, head_position):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_left_eye = self.preprocess_input(left_eye)
        processed_right_eye = self.preprocess_input(right_eye)
        self.exec_network.start_async(request_id=0,
                                      inputs={'left_eye_image': processed_left_eye,
                                              'right_eye_image': processed_right_eye,
                                              'head_pose_angles': head_position})

        if self.exec_network.requests[0].wait(-1) == 0:
            result = self.exec_network.requests[0].outputs[self.output]
            cords = self.preprocess_output(result[0], head_position)
            return cords, result[0]

    def preprocess_output(self, outputs, head_pose_estimation_output):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        roll = head_position[2]
        gaze_vector = output / cv2.norm(output)

        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)


        x = gaze_vector[0] * cosValue * gaze_vector[1] * sinValue
        y = gaze_vector[0] * sinValue * gaze_vector[1] * cosValue
        return (x, y)