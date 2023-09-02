import cv2
import numpy as np

class Order_Based_Background_Subtraction:
    Video_Frame_Stack = height = width = Number_of_Frames = None
    ImageBlockList = []
    Image_Block_Dictionary = {}
    Threshold = 0.03

    def __init__(self, Block_Size, frame_per_second=10):
        self.Block_Size = Block_Size
        self.Frame_Rate = frame_per_second

    def ScanVideo(self, Video_File):
        #Extract frames from the video
        Video = cv2.VideoCapture(Video_File)
        Frame_List = []
        Frame_Count = 0
        Next_Image_Frame = None
        print('Scanning video')
        
        if (Video.isOpened()== False):
            print("Error opening video stream or file")
            
        while Video.isOpened(): # Returns true if video capturing has been initialized already
            retval, frame = Video.read()# retval is whether a frame is read or not
            if Frame_Count % self.Frame_Rate == 0 and (retval != False):
                #RGBframe to Grayscale Frame
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                #Append Gray to list
                Frame_List.append(gray)
            if frame is None:
                break
            Frame_Count = Frame_Count + 1

        #for easy accessibilty of frames 2D list converted to 3D array with 3rd dimension as frame number
        Frame_3D_Array = np.dstack(tuple(Frame_List))
        self.Video_Frame_Stack = Frame_3D_Array
        self.height, self.width, self.Number_of_Frames = Frame_3D_Array.shape
        self.Main_Function()

    def Get_Pixel_Position_Function(self, Block_Number):

        Image_Block_Present = self.Image_Block_Dictionary[Block_Number].Block_Array()

        #Getting total number of Pixels in the Block size considered
        Total_Number_Of_Pixels_In_Block = Image_Block_Present.shape[0]

        #Randomly picking up Numerator Pixel count and choosing  random pixel position as many as pixel count 
        Numerator_Pixel_Count_Selection = 3
        Numerator_Pixel_Position_List = Image_Block_Present[np.random.choice(Total_Number_Of_Pixels_In_Block, Numerator_Pixel_Count_Selection, replace=False), :]

        #Randomly picking up Denominator Pixel count and choosing  random pixel position as many as pixel count
        Denominator_Pixel_Count_Selection = 9
        Denominator_Pixel_Position_List = Image_Block_Present[np.random.choice(Total_Number_Of_Pixels_In_Block, Denominator_Pixel_Count_Selection, replace=False), :]
        
        return Numerator_Pixel_Position_List, Denominator_Pixel_Position_List

    def Main_Function(self):

        #Converting Image to Blocks 
        Block_Number = 1
        for Initial_Row_Number in range(0, self.height, self.Block_Size):
            for Initial_Col_Number in range(0, self.width, self.Block_Size):
                Last_Row_Number = Initial_Row_Number + self.Block_Size
                Last_Col_Number = Initial_Col_Number + self.Block_Size

                #Getting the present mage block
                Image_Block_Present = np.array([(i, j) for i in range(Initial_Col_Number, Last_Col_Number) for j in range(Initial_Row_Number, Last_Row_Number) if i < self.height and j < self.width])
                
                if Image_Block_Present.size == 0:
                    break
                
                self.Image_Block_Dictionary[Block_Number] =Image_Block_Class(Block_Number, Image_Block_Present)    
                Block_Number = Block_Number + 1
        
        for Frame_Index in range(0, self.Number_of_Frames):
            for Block_Number in self.Image_Block_Dictionary.keys():

                #Breaking when Frame index + 1 gets to last frame or more
                if (Frame_Index + 1) >= self.Number_of_Frames:
                    break

                #getting two consecutive Frame
                Present_Image_Frame = self.Video_Frame_Stack[:, :, Frame_Index]
                Next_Image_Frame = self.Video_Frame_Stack[:, :, Frame_Index + 1]

                # Getting random pairs for Numerators and Denominators
                Numerator_Pixel_Position_List, Denominator_Pixel_Position_List = self.Get_Pixel_Position_Function(Block_Number)

                #Finding Intesnity ratios for both the Consecutive frmaes in the same block of each for random pixel poistions 
                Present_Image_Frame_ratio = np.round(((Present_Image_Frame[tuple(np.transpose(Numerator_Pixel_Position_List))].sum()) / (Present_Image_Frame[tuple(np.transpose(Denominator_Pixel_Position_List))].sum()) ),4)
                Next_Image_Frame_ratio = np.round(((Next_Image_Frame[tuple(np.transpose(Numerator_Pixel_Position_List))].sum()) / (Next_Image_Frame[tuple(np.transpose(Denominator_Pixel_Position_List))].sum())),4)

                #Thresholding the difference
                if np.abs(Present_Image_Frame_ratio - Next_Image_Frame_ratio) > self.Threshold:
                    self.Image_Block_Dictionary[Block_Number].is_Background(False)
                else:
                    self.Image_Block_Dictionary[Block_Number].is_Background(True)
            
            Output_Image = np.zeros((self.height, self.width))

            #Output save
            for Block_Number in self.Image_Block_Dictionary.keys():
                if not self.Image_Block_Dictionary[Block_Number].is_Background_Region():
                    Output_Image[tuple(np.transpose(self.Image_Block_Dictionary[Block_Number].Block_Array()))] = 255
            cv2.imwrite('./Order_Output/Order_Based_Highway' + str(Frame_Index) + '.png', Output_Image)


class Image_Block_Class:

    def __init__(self, Block_Number, Image_Block_Array, num_random_pairs=10):
        self.Block_Number = Block_Number
        self.Image_Block_Array = Image_Block_Array
        self.is_BackGround_Flag = None

    def Block_Array(self):
        return self.Image_Block_Array

    def is_Background_Region(self):
        return self.is_BackGround_Flag

    def is_Background(self, isBackground):
        self.is_BackGround_Flag = isBackground

#First argument is block size
#Second Argument is Frame Rate
Obj = Order_Based_Background_Subtraction(3, 20)
Obj.ScanVideo('highway')
