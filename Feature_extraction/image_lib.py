# Setup
import csv
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

def feature_extraction(img, binary):
    gray1 = binary[0]
    binary = binary[1]

    b_img = cv2.medianBlur(img, 5)
    # Mean intensity and Std. Dev.
    mean_i = gray1[binary == 255].mean()
    std_i = b_img[binary == 255].std()

    # Ratio 
    ratio = mean_i / b_img.mean()

    # Laplace of guassian
    gauss = cv2.GaussianBlur(img, (5, 5), 0)
    lap = cv2.Laplacian(gauss, cv2.CV_64F)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.dilate(binary, kernel)
    pixel_gs = np.sort(lap[binary == 255]).ravel()




    # Mean of top 20 % gradients
    mean_g = pixel_gs[-1*int(len(pixel_gs)*0.2):]
    mean_g = mean_g.mean()
    mag, angle, max_mag, mean_mag, std_mag, mean_int, std_int = gradients(gauss)

    return mean_i, std_i, ratio, mean_g, mag, angle, max_mag, mean_mag, std_mag, mean_int, std_int

def pip1(img, blurKernel=5, e_it=1, e_kern=5, d_it=1, d_kern=5):

    ''' First Pipeline : Greyscale -> Median Blur -> Otsu's Method + Inverse Threshold Binarization -> Erosion/Dilation  '''
    '''
    blurKernel: Size of median kernel (3x3, 5x5 ...)
    e_it: Number of erosion iterations 
    e_kern: Kernel size of erosion
    d_it: Number of dilation iterations
    d_kern: Kernel size of dilation
    '''
    
    # Linear Greyscale conversion (Y = 0.299*R + 0.587*G +0.114*B)
    # g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Applies a median blur (blurKernek x blurKernel)
    blur = cv2.medianBlur(img, blurKernel)
    
    # Otsu's method finds dynamic threshold value, pixels are assigned either 0 or 255
    th_val, img_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Erosion and Dilation
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (e_kern,e_kern))
    ero = cv2.erode(img_bin, kernel, iterations=e_it)
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (d_kern,d_kern))
    dil = cv2.dilate(ero, kernel, iterations=d_it)
    
    return (img, dil)

def pip2(img, blurKernel=5, open_k=7, open_its=(3, 1), e_it=1, e_kern=7, d_it=0, d_kern=3):

    ''' Second Pipeline : Greyscale -> Median Blur -> Morphological Transformation (Opening)  
    -> Otsu's Method + Inverse Threshold Binarization -> Erosion/Dilation '''
    '''
    blurKernel: Size of median kernel (3x3, 5x5 ...)
    e_it: Number of erosion iterations 
    e_kern: Kernel size of erosion
    d_it: Number of dilation iterations
    d_kern: Kernel size of dilation
    '''
    
    # Linear Greyscale conversion (Y = 0.299*R + 0.587*G +0.114*B)
    # g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applies a median blur (blurKernek x blurKernel)
    blur = cv2.medianBlur(img, blurKernel)
    
    # Opening (Erosion -> Dilation) on greyscale. 
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (open_k,open_k))
    erosion = cv2.erode(blur, kernel, iterations = open_its[0])
    dilation = cv2.dilate(erosion, kernel, iterations = open_its[1])
     
    # Otsu's method finds dynamic threshold value, pixels are assigned either 0 or 255
    th_val, img_bin = cv2.threshold(dilation, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Erosion and Dilation
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (e_kern,e_kern))
    ero = cv2.erode(img_bin, kernel, iterations=e_it)
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (d_kern,d_kern))
    dil = cv2.dilate(ero, kernel, iterations=d_it)
    
    return dilation, dil 

    ''' Fourth Pipeline : Greyscale -> Median Blur -> Non-Linear Greyscale Transformation -> 
    Otsu's Method on grey pixels only (no 0 or 255) + Inverse Threshold Binarization -> Erosion/Dilation '''

def pip3(img, blurKernel=5, open_k=7, e_it=1, e_kern=5, d_it=1, d_kern=5):

    '''
    Third Pipeline : Greyscale -> Median Blur -> Non-linear Greyscale  
    -> Otsu's Method + Inverse Threshold Binarization -> Erosion/Dilation 


    blurKernel: Size of median kernel (3x3, 5x5 ...)
    e_it: Number of erosion iterations 
    e_kern: Kernel size of erosion
    d_it: Number of dilation iterations
    d_kern: Kernel size of dilation
    '''
    
    # Linear Greyscale conversion (Y = 0.299*R + 0.587*G +0.114*B)
    #g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applies a median blur (blurKernek x blurKernel)
    blur = cv2.medianBlur(img, blurKernel)
    
    # Otsu's method finds dynamic threshold value, pixels are assigned either 0 or 255
    th_val, img_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Opening (Erosion -> Dilation) on greyscale. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,open_k))
    erosion = cv2.erode(img_bin, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
     
    ub_lim = round(np.count_nonzero(dilation)/(227*227), 4)
    
    # Non Linear Greyscale
    nl = nl_grayscale(blur, 0.02, ub_lim)
    nl_grey = nl[nl != 0]
    nl_grey = nl_grey[nl_grey != 255]

    #
    th_val, img_bin = cv2.threshold(nl_grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)    
    th_val, img_bin = cv2.threshold(nl, th_val, 255, cv2.THRESH_BINARY_INV)
  
    # Erosion and Dilation
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (e_kern,e_kern))
    ero = cv2.erode(img_bin, kernel, iterations=e_it)
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (d_kern,d_kern))
    out = cv2.dilate(ero, kernel, iterations=d_it)
    
    return img, out

def pip4(img, blurKernel=5, open_k=7, gray_e=1, e_it=1, e_kern=5, d_it=1, d_kern=5):

    '''
    Fifth Pipeline : Greyscale -> Median Blur -> Non-linear Greyscale  
    Grayscale Morphology -> Otsu's Method + Inverse Threshold Binarization 
    -> Erosion/Dilation 

    blurKernel: Size of median kernel (3x3, 5x5 ...)
    e_it: Number of erosion iterations 
    e_kern: Kernel size of erosion
    d_it: Number of dilation iterations
    d_kern: Kernel size of dilation
    '''
    
    # Linear Greyscale conversion (Y = 0.299*R + 0.587*G +0.114*B)
    #g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applies a median blur (blurKernek x blurKernel)
    blur = cv2.medianBlur(img, blurKernel)
    
    # Otsu's method finds dynamic threshold value, pixels are assigned either 0 or 255
    th_val, img_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Opening (Erosion -> Dilation) on greyscale. 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    erosion = cv2.erode(img_bin, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
     
    ub_lim = round(np.count_nonzero(dilation)/(227*227), 4)
    
    # Non Linear Greyscale
    nl = nl_grayscale(blur, 0.02, ub_lim)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    o_nl = cv2.erode(nl, kernel, iterations=gray_e)
    o_nl = cv2.dilate(o_nl, kernel)

    nl_grey = o_nl[o_nl != 0]
    nl_grey = nl_grey[nl_grey != 255]

    #
    th_val, img_bin = cv2.threshold(nl_grey, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)    
    th_val, img_bin = cv2.threshold(o_nl, th_val, 255, cv2.THRESH_BINARY_INV)
  
    # Erosion and Dilation
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (e_kern,e_kern))
    ero = cv2.erode(img_bin, kernel, iterations=e_it)
    kernel = cv2.getStructuringElement(cv2.cv2.MORPH_ELLIPSE, (d_kern,d_kern))
    out = cv2.dilate(ero, kernel, iterations=d_it)
    
    return o_nl, out

def find_limits(img, lb, ub):
    w, h = img.shape
    im_arr = np.reshape(img, (w*h))
    
    if ub < 0.03:
        ub = 0.03

    im_arr = np.sort(im_arr)
    
    i_a = int(round(w*h * lb, 0))
    i_b = int(round(w*h * ub, 0))
    
    a = im_arr[i_a-1]
    b = im_arr[i_b-1]
    
    t = find_t(im_arr, i_a, i_b)
    
    return a, b, t

def find_t(arr, ia, ib):
    a, b = float(arr[ia]), float(arr[ib])
    
    interval = arr[ia:ib+1]
    mean_i = interval.mean()
    
    t = (2*mean_i*(b - a)) / (255*(a+b))
    return t

def nl_grayscale(img, lb, ub):
    '''
    Non-Linear Greyscale Transformation
    
    Need's fixing: Upper Bound (ub) values .. if above 1.0 or below 0.02 it will mess up
    '''
    new_a = 0.0
    new_b = 255.0
    a, b, t = find_limits(img, lb, ub)
    w, h = img.shape
    
    im_out = img.copy()
    
    i = img < a
    im_out[i] = new_a
    
    i = img > b
    im_out[i] = new_b
    
    for i in range(w):
        for j in range(h):
            if img[i, j] < a and not img[i, j] > b:
                new_a + ((new_a - new_b) / (a**t - b**t)) * (img[i, j]**t - a**t)
            
    return im_out.astype(np.uint8)

def gradients(img):
    

    mean_int = img.mean()
    std_int = img.std()
    
    ############ POSITIVE IMAGES ############################
    # Calculating the gradients
    sobelX_pos = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    sobelY_pos = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Using vector tool to find the magnitudes and the angles
    mag, angle = cv2.cartToPolar(sobelX_pos, sobelY_pos, angleInDegrees=True)
        
    # This is the version where we do not compress values into bins
    mean_mag = np.mean(mag)
    std_mag = np.std(mag)
    un_mag = len(np.unique(mag))
    un_angle = len(np.unique(angle))
    max_mag = np.amax(mag)

    return un_mag, un_angle, max_mag, mean_mag, std_mag, mean_int, std_int