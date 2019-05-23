import cv2
import numpy as np
import scipy
from scipy import ndimage, misc
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def noisy(noise_typ, image, var, mean=0):
    '''
    Applies either: Gaussian Noise (white noise, 0 mean unless otherwise specified)
    or: Salt and pepper noise
    '''

    if noise_typ == "gauss": #adds a gausian filter to the image, using standard deviation
        row,col,ch = image.shape
        sigma=var**1.1 #Sigma variable multiplier at 1.1 adds a great amount of noise
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        noisy = image + gauss

        return noisy.astype(np.uint8)

    # elif noise_typ=="gauss15":
    #     row,col,ch= image.shape
    #     mean=0
    #     var=15
    #     sigma=var**1.1 #Sigma variable multiplier at 1.1 adds a great amount of noise
    #     gauss = np.random.normal(mean,sigma,(row,col,ch))
    #     noisy = image + gauss

    #     # print(np.unique(noisy))
    #     return noisy
    # if noise_typ=="gauss25":
    #     row,col,ch= image.shape
    #     mean=0
    #     var=25
    #     sigma=var**1.1 #Sigma variable multiplier at 1.1 adds a great amount of noise
    #     gauss = np.random.normal(mean,sigma,(row,col,ch))
    #     noisy = image + gauss

    #     # print(np.unique(noisy))
    #     return noisy
    # elif noise_typ=="gauss35":
    #     row,col,ch= image.shape
    #     mean=0
    #     var=35
    #     sigma=var**1.1 #Sigma variable multiplier at 1.1 adds a great amount of noise
    #     gauss = np.random.normal(mean,sigma,(row,col,ch))
    #     noisy = image + gauss

    #     # print(np.unique(noisy))
    #     return noisy
    # elif noise_typ=="gauss45":
    #     row,col,ch= image.shape
    #     mean=0
    #     var=45
    #     sigma=var**1.1 #Sigma variable multiplier at 1.1 adds a great amount of noise
    #     gauss = np.random.normal(mean,sigma,(row,col,ch))
    #     noisy = image + gauss

    #     # print(np.unique(noisy))
    #     return noisy
    
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.08
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0

        return out

def darken(img, value=0.50 ):
    '''
    Changes pixel brightness based on "value"
    ''' 

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim1 = 0
    lim2 = 150

    height, width = v.shape

    for i in range(height):
        for j in range(width):
            if v[i, j] * value > 255:
                v[i, j] = 255
            else:
                v[i, j] = v[i, j] * value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img.astype(np.uint8)

def generate_labels(path_raw, path_new, n):
    raw_img_paths = []
    new_img_paths = []

    for i in range(1,n+1):
        raw_img_paths.append(path_raw + "\\{0:05d}.jpg".format(i))
        new_img_paths.append(path_new + "\\{0:05d}.jpg".format(i))

    return raw_img_paths, new_img_paths

# # CHANGE THIS!!! 
# path = "C:\\Users\\Rasmus\\Desktop\\Deep Learning\\concrete_data\\Positive"

# for i in range(10000, 19379):
#     img = cv2.imread(path + "\\{0:05d}_1.jpg".format(i)) 
#     cv2.imwrite(path + "\\{0:05d}.jpg".format(i), img)

'''
How to use this script to generate folders with augmented images:

!!! IF YOU CRACK IMAGE FOLDERS CONTAINS IMAGES FROM 10000 -> 19378 WHICH IS NAMED "10000_1.jpg, 10001_1.jpg etc" run the 4 lines of code above to unfuck your folder. !!! REMEMBER TO CHANGE THE PATH

1: create the folder that should contain the augmented images (example: gauss10, bright08, bright05_gauss25 etc.)
2: Change the code below so it contains the right paths, noise function and number of images you would like to augment.
3: To review noise functions look further below
'''
# Generate labels for loading raw images and saving augmented images
raw, new = generate_labels("C:\\Users\\Rasmus\\Desktop\\Deep Learning\\concrete_data\\Positive", "C:\\Users\\Rasmus\\Desktop\\Deep Learning\\concrete_data\\augmentation", 50)

# Loads -> applies noise -> Saves 
im = [cv2.imread(i) for i in raw]
gauss = [noisy("gauss", i, 5) for i in im]
out = [cv2.imwrite(new[i], img) for i, img in enumerate(gauss)]


''' Uncomment section below to view examples of noise, remember to change path!!!! ''' 

# Import random image from folder
# crack_pos=cv2.imread("C:\\Users\\Rasmus\\Desktop\\Deep Learning\\concrete_data\\Positive\\{0:05d}.jpg".format(np.random.randint(1,10000)))
# crack_neg=cv2.imread("C:\\Users\\Rasmus\\Desktop\\Deep Learning\\concrete_data\\Negative\\{0:05d}.jpg".format(np.random.randint(1,10000)))


# # Gaussian Noise
# # White noise (gaussian), variance = 5
# g5 = noisy("gauss", crack_pos, 5)
# cv2.imshow("Gauss, Var 5", g5)
# # White noise (gaussian), variance = 15
# g15 = noisy("gauss", crack_pos, 15)
# cv2.imshow("Gauss, Var 15", g15)
# # White noise (gaussian), variance = 25
# g25 = noisy("gauss", crack_pos, 25)
# cv2.imshow("Gauss, Var 25", g25)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Brighten the imgaes
# # by 20 %
# bright20 = darken(crack_pos, 1.2)
# cv2.imshow("Bright 20 %", bright20)
# # by 40 %
# bright40= darken(crack_pos, 1.4)
# cv2.imshow("Bright 40 %", bright40)
# # by 60 %
# bright60 = darken(crack_pos, 1.6)
# cv2.imshow("Bright 60 %", bright60)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Darken the imgaes
# # by 20 %
# dark20 = darken(crack_pos, 0.8)
# cv2.imshow("Dark 20 %", dark20)
# # by 40 %
# dark40= darken(crack_pos, 0.6)
# cv2.imshow("Dark 40 %", dark40)
# # by 60 %
# dark60 = darken(crack_pos, 0.4)
# cv2.imshow("Dark 60 %", dark60)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''  Un-used Code from here '''

# Change color space on pictures
# crack_pos_gs=cv2.cvtColor(crack_pos,cv2.COLOR_BGR2GRAY)
# crack_neg_gs=cv2.cvtColor(crack_neg, cv2.COLOR_BGR2GRAY)
# crack_pos_HSV = cv2.cvtColor(crack_pos, cv2.COLOR_BGR2HSV)
# crack_neg_HSV = cv2.cvtColor(crack_neg, cv2.COLOR_BGR2HSV)

# Darkened Image
# data = darken(crack_pos, 0.10)
# crack_pos_d10= data.astype(np.uint8)
# data = darken(crack_pos, 0.2)
# crack_pos_d20= data.astype(np.uint8)
# data = darken(crack_pos, 0.40)
# crack_pos_d40= data.astype(np.uint8)
# data = darken(crack_pos, 0.60)
# crack_pos_d60= data.astype(np.uint8)
# data = darken(crack_pos, 0.80)
# crack_pos_d80 =data.astype(np.uint8)




# # Brightened Image
# crack_pos_b20= data.astype(np.uint8)
# data = darken(crack_pos, 1.20)
# crack_pos_b40= data.astype(np.uint8)
# data = darken(crack_pos, 1.4)
# crack_pos_b60 = data.astype(np.uint8)
# data = darken(crack_pos, 1.6)
# crack_pos_b80 = data.astype(np.uint8)
# data = darken(crack_pos, 1.8)
# crack_pos_b90 = data.astype(np.uint8)
# data = darken(crack_pos, 1.9)

# cv2.imshow("Darkened 90%", crack_pos_d10)
# cv2.imshow("Darkened 80%", crack_pos_d20)
# cv2.imshow("Darkened 60%", crack_pos_d40)
# cv2.imshow("Darkened 40%", crack_pos_d60)
# cv2.imshow("Darkened 20%", crack_pos_d80)
# cv2.imshow("Brighened 20%", crack_pos_b20)
# cv2.imshow("Brightened 40%", crack_pos_b40)
# cv2.imshow("Brightened 60%", crack_pos_b60)
# cv2.imshow("Brightened 80%", crack_pos_b80)
# cv2.imshow("Brightened 90%", crack_pos_b90)
# cv2.imshow("Crack Pos", crack_pos)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Negative images
# # Darkened Image
# data = darken(crack_neg, 0.10)
# crack_neg_d10= data.astype(np.uint8)
# data = darken(crack_neg, 0.2)
# crack_neg_d20= data.astype(np.uint8)
# data = darken(crack_neg, 0.40)
# crack_neg_d40= data.astype(np.uint8)
# data = darken(crack_neg, 0.60)
# crack_neg_d60= data.astype(np.uint8)
# data = darken(crack_neg, 0.80)
# crack_neg_d80 =data.astype(np.uint8)

# # Brightened Image
# crack_neg_b20 = data.astype(np.uint8)
# data = darken(crack_neg, 1.20)
# crack_neg_b40 = data.astype(np.uint8)
# data = darken(crack_neg, 1.4)
# crack_neg_b60 = data.astype(np.uint8)
# data = darken(crack_neg, 1.6)
# crack_neg_b80 = data.astype(np.uint8)
# data = darken(crack_neg, 1.8)
# crack_neg_b90 = data.astype(np.uint8)
# data = darken(crack_neg, 1.9)

# cv2.imshow("Darkened 90%", crack_neg_d10)
# cv2.imshow("Darkened 80%", crack_neg_d20)
# cv2.imshow("Darkened 60%", crack_neg_d40)
# cv2.imshow("Darkened 40%", crack_neg_d60)
# cv2.imshow("Darkened 20%", crack_neg_d80)
# cv2.imshow("Brighened 20%", crack_neg_b20)
# cv2.imshow("Brightened 40%", crack_neg_b40)
# cv2.imshow("Brightened 60%", crack_neg_b60)
# cv2.imshow("Brightened 80%", crack_neg_b80)
# cv2.imshow("Brightened 90%", crack_neg_b90)
# cv2.imshow("Crack Pos", crack_neg)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Positive Crack
# # Gaussian
# data = noisy("gauss5", crack_pos)
# gaus_img5 = data.astype(np.uint8)
# Gaus5_gs_pos = cv2.cvtColor(gaus_img5,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss15", crack_pos)
# gaus_img15 = data.astype(np.uint8)
# Gaus15_gs_pos = cv2.cvtColor(gaus_img15,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss25", crack_pos)
# gaus_img25 = data.astype(np.uint8)
# Gaus25_gs_pos = cv2.cvtColor(gaus_img25,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss35", crack_pos)
# gaus_img35 = data.astype(np.uint8)
# Gaus35_gs_pos = cv2.cvtColor(gaus_img35,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss45", crack_pos)
# gaus_img45 = data.astype(np.uint8)
# Gaus45_gs_pos = cv2.cvtColor(gaus_img45,cv2.COLOR_BGR2GRAY)

# # Salt and pepper
# data = noisy("s&p", crack_pos)
# sp_img =data.astype(np.uint8)
# sp_gs_pos = cv2.cvtColor(sp_img, cv2.COLOR_BGR2GRAY)

# # # Add filter that changes a random amount of pixels with 20% intensity
# # # Make different variations of 3 filters with different levels of noise
# # # Reduce value, Transform image to HSV and change V with intervals of 5% to perhaps 40%

# #Negative Crack
# # Gaussian
# data = noisy("gauss5", crack_neg)
# gaus5_neg_img = data.astype(np.uint8)
# Gaus5_gs_neg = cv2.cvtColor(gaus5_neg_img,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss15", crack_neg)
# gaus15_neg_img = data.astype(np.uint8)
# Gaus15_gs_neg = cv2.cvtColor(gaus15_neg_img,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss25", crack_neg)
# gaus25_neg_img = data.astype(np.uint8)
# Gaus25_gs_neg = cv2.cvtColor(gaus25_neg_img,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss35", crack_neg)
# gaus35_neg_img = data.astype(np.uint8)
# Gaus35_gs_neg = cv2.cvtColor(gaus35_neg_img,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss45", crack_neg)
# gaus45_neg_img = data.astype(np.uint8)
# Gaus45_gs_neg = cv2.cvtColor(gaus45_neg_img,cv2.COLOR_BGR2GRAY)

# #Salt And Pepper
# data = noisy("s&p", crack_neg)
# sp_img =data.astype(np.uint8)
# sp_gs_neg = cv2.cvtColor(sp_img, cv2.COLOR_BGR2GRAY)

# #
# #
# # Show crack Possitive augmented images
# cv2.imshow("Grayscale Pos", crack_pos_gs)
# cv2.imshow("Gaus 5 Pos", Gaus5_gs_pos)
# cv2.imshow("Gaus 15 Pos", Gaus15_gs_pos)
# cv2.imshow("Gaus 25 Pos", Gaus25_gs_pos)
# cv2.imshow("Gaus 35 Pos", Gaus35_gs_pos)
# cv2.imshow("Gaus 45 Pos", Gaus45_gs_pos)
# cv2.imshow("S&P Pos", sp_gs_pos)




# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Show crack negative augmented images
# cv2.imshow("Gaus 5 Neg", Gaus5_gs_neg)
# cv2.imshow("Gaus 15 Neg", Gaus15_gs_neg)
# cv2.imshow("Gaus 25 Neg", Gaus25_gs_neg)
# cv2.imshow("Gaus 35 Neg", Gaus35_gs_neg)
# cv2.imshow("Gaus 45 Neg", Gaus45_gs_neg)
# cv2.imshow("S&P Neg", sp_gs_neg)
# cv2.imshow("Grayscale neg", crack_neg_gs)




# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Positive Crack Decreased brightness decreased by 20%
# # Gaussian
# data = noisy("gauss5", crack_pos_d80)
# gaus_img5_d80 = data.astype(np.uint8)
# Gaus5_gs_pos_d80 = cv2.cvtColor(gaus_img5,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss15", crack_pos_d80)
# gaus_img15_d80 = data.astype(np.uint8)
# Gaus15_gs_pos_d80 = cv2.cvtColor(gaus_img15_d80,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss25", crack_pos_d80)
# gaus_img25_d80 = data.astype(np.uint8)
# Gaus25_gs_pos_d80 = cv2.cvtColor(gaus_img25_d80,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss35", crack_pos_d80)
# gaus_img35_d80 = data.astype(np.uint8)
# Gaus35_gs_pos_d80 = cv2.cvtColor(gaus_img35_d80,cv2.COLOR_BGR2GRAY)
# data = noisy("gauss45", crack_pos_d80)
# gaus_img45 = data.astype(np.uint8)
# Gaus45_gs_pos_d80 = cv2.cvtColor(gaus_img45,cv2.COLOR_BGR2GRAY)

# # Show crack Possitive augmented images
# cv2.imshow("Grayscale Pos", crack_pos_d80)
# cv2.imshow("Gaus 5 Pos d80", Gaus5_gs_pos_d80)
# cv2.imshow("Gaus 15 Pos d80", Gaus15_gs_pos_d80)
# cv2.imshow("Gaus 25 Pos d80", Gaus25_gs_pos_d80)
# cv2.imshow("Gaus 35 Pos d80", Gaus35_gs_pos_d80)
# cv2.imshow("Gaus 45 Pos d80", Gaus45_gs_pos_d80)

# cv2.waitKey(0)
# cv2.destroyAllWindows()