import cv2
import numpy as np
#
# A module to hold several image-data related helper functions
#

def read_img(img_name, color=False, scales=(1,1), brightness_is_normed=False):
    ''' 
    The image at the given path will be read, normalized and rescaled using opencv.
    
    Args:
        img_name (str):
            The path to the image that will be loaded.
        color (bool):
            If True the image will be loaded as RGB otherwise as grayscale
        scales (tuple of int):
            Scaling factor by which x- and y-Dimension of the input image is divided
        brightness_is_normed (bool):
            If True the brightness across the input image will be normed

    Returns:
        img (np.ndarray):
            The read image.

    '''
    img = cv2.imread(img_name, color)
    h, w = img.shape
    dh, dw = scales
    img = cv2.resize(img, ( int(np.ceil(w/dw)), int(np.ceil(h/dh)) ))
    if brightness_is_normed:
        img = normalize_brightness(img)
    img = cv2.normalize(img.astype(np.float), None, 0.0, 1.0, cv2.NORM_MINMAX)
    return img

def normalize_brightness(img, sig=10):
    '''
    Use ratio of gaussians to equalize illumination across a grayscale image

    Args:
        img (numpy.ndarray): a grayscale image.

        sig (int) : the variance of the gaussian filter used.

    '''
    filt = cv2.GaussianBlur(img,(0,0),sig)
    rog = np.divide(img, filt)
    return rog

def get_hessian_det(L, t):
    ''' 
    Compute the hessian determinant for a grayscale image blurred with t.

    Args:
        L (np.ndarray):
            A grayscale image blurred with a sigma of t.

        t (int):
            Sigma parameter for gaussian blur.

    Returns:
        H (np.ndarray):
            The hessian determinant of the given image.

    '''
    L_x, L_y = np.gradient(L)
    L_xx, L_xy = np.gradient(L_x)
    _, L_yy = np.gradient(L_y)
    H = t**2 * (L_xx * L_yy - L_xy**2)
    return H

def get_scale_space(img, scales, compute_hessian=False):
    '''
    For a list of scales, return a list of scale space transformed images.
    
    Args:
        img (np.ndarray):
            Grayscale input image.
        scales (list of int):
            List of the scales to be used for blurring.
        compute_hessian (bool):
            If True, the hessian determinant is also computed for every scale.

    Returns:
        (sc_sp, hes_det) (tuple of list of np.ndarray):
            A tuple holding a list of the scale transformed images and a list of hessian determinants for the corresponding scales. If compute_hessian is False, the second list will be empty.

    '''
    sc_sp = []
    hes_det = []
    for scale in scales:
        blur = cv2.GaussianBlur(img, (0,0), scale)
        sc_sp.append(blur)
        if compute_hessian:
            hes_det.append(get_hessian_det(blur, scale))
    return sc_sp, hes_det