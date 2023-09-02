import numpy as np
import cv2

offset= 1e-10

def Gaussian(X, Mean_Value, Sigma):
    # where X is a random variable denoting gray level intensity of a particular point
    PDF = (1 / ((Sigma + offset) * np.sqrt(2 * np.pi))) * np.exp(-(1 / 2) * ((X - Mean_Value) / (Sigma + offset)) ** 2)
    return PDF


class GMM:
    #learning constant
    alpha = 0.6

    #Number of Gaussian Distribution
    K = None

    BG_thresh = 0.6

    Omega = Mean_Array = Sigma_Array = height = width = Number_of_Frames = Video_Frame_Stack = Background_Frame = Frame_Rate = MAX_ITERATIONS = None

    def __init__(self, Frame_Rate, K):
        self.K = K
        self.Frame_Rate = Frame_Rate

    #Extract frames from the video
    def ScanVideo(self, Video_File):
        Video = cv2.VideoCapture(Video_File)
        Frame_List = []
        Frame_Count = 0
        print('Scanning video')

        if (Video.isOpened()== False):
            print("Error opening video stream or file")

        while Video.isOpened(): # Returns true if video capturing has been initialized already
            retval, frame = Video.read() # retval is whether a frame is read or not
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
        self.Initialising_Various_Array()

    def Initialising_Various_Array(self):

        #The weight Array initially has equal weight given to each pixel and each frame number
        self.Omega = np.ones([self.height, self.width, self.K])
        self.Mean_Array = np.zeros([self.height, self.width, self.K])
        for i in range(self.height):
            for j in range(self.width):
                pixel_intensity_array = self.Video_Frame_Stack[i,j, :]

                #we re randomly choosing K frames out of Number_of_Frames 
                self.Mean_Array[i, j, :] = np.random.choice(pixel_intensity_array,self.K)
                
                self.Omega[i,j,:]= np.round((1 / self.K),6)

        self.Sigma_Array = 0.02 * self.Mean_Array

        #Initialize a dummy background image of the same size as that of the individual frames
        self.Background_Frame = np.ones(self.Video_Frame_Stack.shape) * 255
        self.Main_Function()


    # The following is the exact implementation of the Stauffer and Grimson Paper on adaptive background subtraction
    

    def Main_Function(self):
        print('Total number of Frames:' + str(self.Number_of_Frames))
        for T in range(self.Number_of_Frames):
            print('Processing Frame ' + str(T + 1))
            for col in range(self.width):
                for row in range(self.height):
                
                    pixel_intensity_array = self.Video_Frame_Stack[row, col, :].astype(np.float)

                    #the present value of various Matrices
                    Mean_Array_at_t = self.Mean_Array[row, col, :]
                    Sigma_Array_at_t = self.Sigma_Array[row, col, :]
                    Omega_at_t = self.Omega[row, col, :]
    
                    #getting pixel intensity at a particular pixel position([row ,col]) of frame at T instant
                    pixel_intensity = self.Video_Frame_Stack[row, col, T]
    
                    #Probability For particular pixel position and various matrix(Mean, S.D) present value for each Gaussian distribution                    
                    Prob = np.array([Gaussian(pixel_intensity, Mean_Array_at_t[gaussian_index],Sigma_Array_at_t[gaussian_index])for gaussian_index in range(self.K)])
    
                    #Boolean value array corresponding to the match and unmatch       
                    matched_gaussians = self.GaussianMatching(row, col, pixel_intensity)
    
                        
                    self.NonMatchedGaussian(row, col, pixel_intensity, Prob, matched_gaussians)
    
    
                    #Update weight

                    #Decrease weight for non matched and Increase wesiht for matched Gaussian
                    Omega_at_tplus1 = (1 - self.alpha) * Omega_at_t + self.alpha * matched_gaussians
                    Omega_at_tplus1 = np.round((Omega_at_tplus1 / np.sum(Omega_at_tplus1)),6)
                    self.Omega[row, col, :] = Omega_at_tplus1
    
                    rho = self.alpha * Prob
    
                    #Update Mean Array
                    Mean_Array_at_tplus1 = np.copy(Mean_Array_at_t)
                    Mean_Array_at_tplus1[matched_gaussians] = ((1 - rho) * Mean_Array_at_t + rho * pixel_intensity)[matched_gaussians]
                    self.Mean_Array[row, col, :] = Mean_Array_at_tplus1
    
                    #Update Sigma Array
                    Sigma_Array_at_tplus1 = np.copy(Sigma_Array_at_t)
                    Sigma_Array_at_tplus1[matched_gaussians] = np.sqrt(np.absolute(((1 - rho) * np.square(Sigma_Array_at_t) + rho * np.square((pixel_intensity - Mean_Array_at_t))))[matched_gaussians])
                    self.Sigma_Array[row, col, :] = Sigma_Array_at_tplus1
    
                    if self.isBackgroundPixel(np.round((Omega_at_t / (Sigma_Array_at_t + offset)),6)):
                        self.Background_Frame[row, col, T] = 0
    

        #save detected background/foreground frames
        for i in range(GMMObject.Number_of_Frames):
            cv2.imwrite('./GMM_Output/GMM_Canoe' + str(i) + '.png', GMMObject.Background_Frame[:, :, i])                    

    def NonMatchedGaussian(self, row, col, pixel_intensity, Prob, matched_gaussian):
        if np.sum(matched_gaussian) == 0: #if all are non matched then
            argmin_gaussian = np.argmin(Prob)
            self.Mean_Array[row, col, argmin_gaussian] = pixel_intensity
            self.Sigma_Array[row, col, argmin_gaussian] = 0.02* pixel_intensity


    # A Match is defined as a pixel value within 2.5 standard deviation
    def GaussianMatching(self, row, col, pixel_intensity):
        Mean = self.Mean_Array[row, col, :]
        Sigma = self.Sigma_Array[row, col, :]
        return np.logical_and(pixel_intensity > (Mean-(2.5 * Sigma)), pixel_intensity < (Mean+(2.5 * (Sigma + offset))))

    def isBackgroundPixel(self, OmegaBySigma):
        OmegaBySigma[::-1].sort()
        return np.sum(OmegaBySigma[:self.K - 2]) <= self.BG_thresh

#First Argument is Frame Rate
#Second Argument is Number of Gaussians
GMMObject = GMM(15, 5)
GMMObject.ScanVideo('canoe')
