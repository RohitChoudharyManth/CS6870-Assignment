#Non Parametric Method/ Kernel Method
import numpy as np
import cv2
import time

#N is the Frame Block Size

#Small Offset to avoid (1/0) scenario , warning
Offset = 1e-10

class BSNonParametric:
    Video_Frame_Stack = None
    Thres = None
    N = None

    def __init__(self, Thres, N):
        self.height = self.width = self.Number_of_Frames = None
        self.Thres = Thres
        self.N = N

    def ScanVideo(self, Video_File):
        Video = cv2.VideoCapture(Video_File)
        Frame_List = []
        Frame_Frame_Count = 0
        print('Scanning video')

        if (Video.isOpened()== False):
            print("Error opening video stream or file")

        while Video.isOpened(): # retvalurns true if video capturing has been initialized already
            retvalval, frame = Video.read() # retvalval is whether a frame is read or not
            if (retvalval != False):
                #RGBframe to Grayscale Frame
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                #Append Gray to list
                Frame_List.append(gray)
            if frame is None:
                break
            Frame_Frame_Count = Frame_Frame_Count + 1
        #
        #for easy accessibilty of frames 2D list converted to 3D Arrayay with 3rd dimension as frame number
        Frame_3D_Arrayay = np.dstack(tuple(Frame_List))
        self.Video_Frame_Stack = Frame_3D_Arrayay
        self.height, self.width, self.Number_of_Frames = Frame_3D_Arrayay.shape
        print('Video converted to Stack of Frames')
        self.BackgroundDetection()



    def BackgroundDetection(self):
        print('Total number of Frames :'+ str(self.Number_of_Frames))
        for Frame_number in range(self.N, self.Number_of_Frames, self.N):   #range(start,stop,step)
            print('Processing Frame Number:' + str(Frame_number-self.N) + 'to' + str(Frame_number))
            Output_Frame_Image = np.zeros((self.height, self.width))
            for c in range(self.width):
                for r in range(self.height):

                    #xt is considered the pixel value of pixel position (r,c) in the last frame of frame block size N thus reducing the considered size to (N-1)
                    xt=np.array([self.Video_Frame_Stack[r, c, Frame_number].astype(np.float)] * (self.N - 1))
                    xi=(np.array(self.Video_Frame_Stack[r, c,Frame_number - self.N: Frame_number].astype(np.float)))[:-1]

                    #Finding Sigma using median
                    #For N enteries (N-1) median are found
                    Block_Pixel_values = self.Video_Frame_Stack[r, c,Frame_number - self.N: Frame_number].astype(np.float)
                    Shifted_Block_Pixel_values = np.zeros(Block_Pixel_values.shape[0]).astype(np.float)                    
                    Shifted_Block_Pixel_values[1:] = Block_Pixel_values[:-1]
                    Sigma = (np.abs(Block_Pixel_values - Shifted_Block_Pixel_values)[1:]) / (0.68 * np.sqrt(2)) + Offset

                    #Finding the Estimate Gaussian
                    Gaussian_Estimate = (1 / (self.N - 1)) * np.sum((1 / np.sqrt(2 * np.pi * Sigma ** 2)) * np.exp(-(np.square((xt - xi) / Sigma)) / 2))

                    #if Estimate less than a thresold then make it foreground
                    if (Gaussian_Estimate < self.Thres):
                        Output_Frame_Image[r, c] = 255
            cv2.imwrite('Non_Parametric_Output/Non_Parametric_Highway' + str(Frame_number) + '.png', Output_Frame_Image)


#First Argument is threshold
#Second Argument is Frame Block Size (Frame number size in a frame block)
Obj = BSNonParametric(0.05,20)
Obj.ScanVideo('highway')
