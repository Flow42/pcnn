import numpy as np
import os, time, shutil
import imageio
import cv2

class pcnn:
    '''
    A class implementing various methods using pulse-coupled neural networks (PCNN)

    Attributes:
        kernel (int or np.ndarray):
            A kernel of ones to implement local summation using convolution. When an int is given a ndarray of shape (int,int) is used. One can also give a kernel of choice to implement weighted sums in the receptive field. You can also give 1, so no receptive field is used. Default is a square array of ones with shape (3,3).

        epochs (int):
            The number of iterations to run the PCNN. Default is 10.

        do_rog (bool): 
            If True the read images brightness is normalized using a Ratio of Gaussian (rog) approach. Default is False.

        V (int or np.ndarray): 
            parameter to control the adaption of the threshold. Give a ndarray to specify a value for each pixel. Default is 0.1

        alpha (int or np.ndarray): 
            parameter to control the adaption of the threshold. Give a ndarray to specify a value for each pixel. Default is 0.005

        beta (int or np.ndarray): 
            linking strength. Determines how much of the activations from the linking channel is used for the internal activation. Give a ndarray to specify a value for each pixel. Default is 0.2

        dropout (bool): 
            If True, a random number of kernel entries is set to 0 in each epoch. Default is False.

        scales (tuple of int): 
            A tuple of length 2, that defines a downscaling factor for each dimension of the input image. Default is (1,1), meaning no rescaling.

        t (list of int): 
            Sigma values by which the image is blurred for the scale space representation. Default is [2,4,6,8]

        write_gif (bool): 
            set to True if the pulses of each epoch should be written to a gif in the current directory. Uses imageio package. Default is False.

        show_epoch (bool): 
            If True, each frame of the written gif shows the number of the current epoch. Default is false.

        out_path (str): 
            The path where outputs, like the gif, are written to. The folder will be deleted if it already exists, otherwise it is created. Default is "out/" in the current dir.

    '''
    def __init__(self, kernel=None, epochs=None, do_rog=False, V=None, alpha=None, beta=None,
                 F_init=None, Y_init=None, dropout=False, scales=(1,1), t=[2,4,6,8],
                 write_gif=False, show_epoch=False, out_path='out'):
        ### input check
        if kernel is None:
            self.kernel = np.ones((3,3))
        elif isinstance(kernel, int):
            assert kernel >= 1, 'Kernel size can not be 0 or negative!'
            self.kernel = np.ones((kernel,kernel))
        elif isinstance(kernel, np.ndarray):
            self.kernel = kernel
        elif isinstance(kernel, list):
            assert all(isinstance(x,np.ndarray) for x in kernel), 'all elements of the list must be kernels of type ndarray!'
            self.kernel = kernel
        else:
            raise ValueError('Invalid datatype for kernel! Pass an integer, a numpy ndarray or a list of arrays')
        
        if epochs is None:
            self.epochs = 10
        else:
            assert isinstance(epochs,int), 'epochs must be given as integer!'
            self.epochs = epochs
        
        assert isinstance(do_rog, bool), 'do_rog must be a boolean!'
        self.do_rog = do_rog

        if alpha is None:
            self.alpha = 0.005
        else:
            assert isinstance(alpha,(float,np.ndarray)), 'alpha must be given as float or ndarray!!'
            self.alpha = alpha

        if beta is None:
            self.beta = 0.2
        else:
            assert isinstance(beta,(float,np.ndarray)), 'beta must be given as float or ndarray!!'
            self.beta = beta

        if V is None:
            self.V = 1.0
        else:
            assert isinstance(V,(float, int, np.ndarray)), 'V must be given as float, int or ndarray!'
            self.V = V

        assert isinstance(dropout, bool), 'dropout must be bool!'
        self.dropout = dropout

        assert isinstance(scales, tuple) and all(isinstance(x, int) for x in scales) and len(scales) == 2,\
            'scales must be a tuple of 2 ints: (dh, dw)'
        assert all(scales) >= 0, 'a scaling factor can not be negative!'
        self.scales = scales

        assert isinstance(t, list) and all(isinstance(x, int) for x in t), 't must be a list of ints!'
        self.t = t

        assert isinstance(write_gif, bool), 'write_gif must be bool!'
        self.write_gif = write_gif

        assert isinstance(show_epoch, bool), 'show_epoch must be bool!'
        self.show_epoch = show_epoch

        assert isinstance(out_path, str), 'out_path must be given as string'
        self.out_path = out_path
        # create folder later, when it is needed
        ###
    
    def _create_out_folder(self):
        out_path = self.out_path
        if os.path.isdir(out_path):
            shutil.rmtree(out_path)
            os.makedirs(out_path)
        else:
            os.makedirs(out_path)

    def _normalize_brightness(self, img, sig=10):
        '''
        Use ratio of gaussians to equalize illumination across a grayscale image

        Args:
            img (numpy.ndarray): a grayscale image.

            sig (int) : the variance of the gaussian filter used.

        '''
        filt = cv2.GaussianBlur(img,(0,0),sig)
        rog = np.divide(img, filt)
        return rog

    def _get_hessian_det(self, L, t):
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

    def _get_scale_space(self, img, scales, compute_hessian=False):
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
                hes_det.append(self._get_hessian_det(blur, scale))
        return sc_sp, hes_det

    def _read_img(self, img_name, color=False):
        ''' 
        The image at the given path will be read, normalized and rescaled using opencv.
        
        Args:
            img_name (str):
                The path to the image that will be loaded.
            color (bool):
                If True the image will be loaded as RGB otherwise as grayscale

        Returns:
            img (np.ndarray):
                The read image.

        '''
        img = cv2.imread(img_name, color)
        h, w = img.shape
        dh, dw = self.scales
        img = cv2.resize(img, ( int(np.ceil(w/dw)), int(np.ceil(h/dh)) ))
        if self.do_rog:
            img = self._normalize_brightness(img)
        img = cv2.normalize(img.astype(np.float), None, 0.0, 1.0, cv2.NORM_MINMAX)
        return img

    def gray_pcnn(self, img, Y_init=None, F_init=None, thresh=None, gif_name=None):
        '''
        Compute the response of a Unit-linking Pulse-Coupled Neural Network (PCNN) on a given gray image.

        The neurons will be structured in 2D. Thus this implementation can be used with any 2-dimensional np.ndarray.

        Args:
            img (np.ndarray): 
                The image to run the PCNN on. Is asumed to be gray-scale.

            F_init (np.ndarray): 
                The feeding channel. Has the same shape as img. Defaults to the values of the given image.

            Y_init (np.ndarray): 
                The initial pulse values for every neuron. Has the same shape as img. Default is zeros.

            thresh (int or np.ndarray): 
                Initial threshold. Give a ndarray to specify a value for each pixel. Defaults to the values of the given image.

        Returns:
            Y_out (np.ndarray):
                2D Array of pulses, having the highest entropy.

        '''
        pcnn_gen = self.gray_pcnn_gen(img, Y_init, F_init, thresh)
        h,w = img.shape
        write_gif = self.write_gif
        out_path = self.out_path
        if gif_name is None:
            gif_name = 'movie.gif'
        assert isinstance(gif_name, str), 'The gif_name must be a valid string!'
        assert gif_name.endswith('.gif'), 'The .gif extension must be given explicitly!'
        max_ent = -np.Inf
        if write_gif:
            out_list=[]
        for epoch in range(self.epochs):
            Y = next(pcnn_gen)
            # compute entropy of current segmentation
            P_0 = np.sum(Y == 0) / h*w
            P_1 = np.sum(Y == 1) / h*w
            ent = -P_0*np.log2(P_0) - P_1*np.log2(P_1)
            # and store the result with the highest entropy
            if ent > max_ent:
                max_ent = ent
                Y_out = Y
            # spatial_hist = spatial_hist + Y
            if write_gif:
                # convert to float for correct behaviour with opencv
                out_list.append((255*Y).astype(np.uint8))
        if write_gif:
            with imageio.get_writer(os.path.join(out_path, gif_name), mode='I',fps=8) as writer:
                for idx, Y in enumerate(out_list):
                    out = np.dstack((Y,)*3)
                    if self.show_epoch:
                        out = cv2.putText(out, text=str(idx), org=(int(w-w*0.1),int(h-h*0.05)), fontFace=3, fontScale=1.1, color=(0,0,1), thickness=4)
                    writer.append_data(out)
                print('gif {} has been written'.format(os.path.join(out_path, gif_name)))
        return Y_out

    def gray_pcnn_gen(self, img, Y_init=None, F_init=None, thresh=None):
        '''
        A generator with the same functionality as gray_pcnn.

        The neurons will be structured in 2D. Thus this implementation can be used with any 2-dimensional np.ndarray.
        This generator runs in an endless loop, so its termination must be implemented by its caller.

        Args:
            img (np.ndarray):
                the image to run the PCNN on. Is asumed to be gray-scale.

            Y_init (np.ndarray):
                The initial pulse values for every neuron. Has the same shape as img. Default is zeros.

            F_init (np.ndarray):
                The feeding channel. Has the same shape as img. Defaults to the values of the given image.

            thresh (int or np.ndarray):
                Initial threshold. Give a ndarray to specify a value for each pixel. Defaults to the values of the given image.
            
        Yields:
            Y (np.ndarray):
                The current state of all pulses.
           
        '''
        h,w = img.shape
        if Y_init is None:
            Y = np.zeros((h,w))
        else:
            assert isinstance(Y_init, np.ndarray), 'Y_init must be given as ndarray, if it is given!!'
            Y = Y_init

        if F_init is None:
            F = img
        else:
            assert isinstance(F_init, np.ndarray), 'F_init must be given as ndarray, if it is given!!'
            F = F_init
        
        if thresh is None:
            # thresh = F
            thresh = np.ones_like(img)
        else:
            assert isinstance(thresh,(float,np.ndarray)), 'thresh must be given as float or ndarray!!'

        V = self.V
        alpha = self.alpha
        beta = self.beta

        while True:
            k = self.kernel
            if self.dropout:
                k = k * np.random.randint(2, size=k.shape)            
            Y_sum = cv2.filter2D(Y.astype(np.float), -1, k, borderType=cv2.BORDER_REFLECT_101)
            L = Y_sum > 0
            # compute inner activation
            U = F*(1 + beta*L)
            # compute each neurons pulse state
            Y = ((U-thresh) > 0)
            thresh = thresh + (-1*alpha + V * Y)
            yield Y

    def get_gen(self, img_name, color=False):
        '''
        This function returns a pcnn generator, expecting an images name instead of an actual image

        Args:
            img_name (str):
                Path to the image on which the pcnn will be run.

        Returns:
            Python generator object, yielding a pulse state for the given image on each next() call.

        '''
        return self.gray_pcnn_gen(self._read_img(img_name, color))

    def run(self, input, color=False, write_results=True, use_scale_space=False, use_hessian=False, Y_init=None, F_init=None, thresh=None):
        ''' 
        Wrapper for all the different models of pcnn

        Use this function as access point for all functionalities

        Args:
            input(str): 
                Give an images filename to be computed or give a folder to compute the result for all images in that folder.

            color (bool): 
                if True the image will be loaded as RGB otherwise as grayscale. Default is False.

            write_results (bool): 
                if True all results will be written to disk. Default is True.

            use_scale_space (bool): 
                If True the input images scale space is computed and the pcnn is applied to each of the resulting images. Default is False.

            use_hessian (bool): 
                If True the input images scale space and the hessian determinant for each scale is computed and the pcnn is applied to each of the resulting images. Default is False.

            Y_init (np.ndarray): 
                The initial pulse values for every neuron. Has the same shape as img. Default is zeros.

            F_init (np.ndarray): 
                The feeding channel. Has the same shape as img. Defaults to the values of the given image.

            thresh (int or np.ndarray): 
                Initial threshold. Give a ndarray to specify a value for each pixel. Defaults to the values of the given image.

        Returns:
            result (list of np.ndarray):
                The pulse state of maximum entropy for each given image.

        '''
        result = []
        self._create_out_folder()
        if os.path.isdir(input):
            img_path = input
            img_name_list = []
            for dirpath, dirnames, files in os.walk(img_path):
                for f in files:
                    if f.endswith(('.png', '.bmp', '.jpg')):
                        img_name_list.append(os.path.join(dirpath, f))

            for img_name in img_name_list:
                img = self._read_img(img_name, color)
                out_name = img_name.split('/')[-1]
                out_sub_path = os.path.join(self.out_path, os.path.splitext(out_name)[0])
                # make a subfolder for the current image
                os.makedirs(out_sub_path)
                if use_scale_space or use_hessian:
                    sc_sp, hes_det = self._get_scale_space(img, self.t, use_hessian)
                    inter_list = hes_det if use_hessian else sc_sp
                    for idx, scale in enumerate(inter_list):
                        r = self.gray_pcnn(scale, Y_init, F_init, thresh, 
                                           gif_name=os.path.splitext(out_name)[0] + 'scale_{}.gif'.format(self.t[idx]))
                        if write_results:
                            cv2.imwrite(os.path.join(out_sub_path, 'res_sc{}.png'.format(self.t[idx])), (r*255).astype(np.uint8))
                        result.append(r)
                r = self.gray_pcnn(img, Y_init, F_init, thresh, gif_name=os.path.splitext(out_name)[0]+'.gif')
                if write_results:
                    cv2.imwrite(os.path.join(out_sub_path, 'res.png'), (r*255).astype(np.uint8))
                result.append(r)
        else:
            img_name = input
            img = self._read_img(img_name, color)
            out_name = img_name.split('/')[-1]
            if use_scale_space or use_hessian:
                sc_sp, hes_det = self._get_scale_space(img, self.t, use_hessian)
                inter_list = hes_det if use_hessian else sc_sp
                for idx, scale in enumerate(inter_list):
                    r = self.gray_pcnn(scale, Y_init, F_init, thresh, gif_name=os.path.splitext(out_name)[0] + 'scale_{}.gif'.format(self.t[idx]))
                    if write_results:
                        cv2.imwrite(os.path.join(self.out_path, os.path.splitext(out_name)[0] + 'res_sc{}.png'.format(self.t[idx])), (r*255).astype(np.uint8))
                    result.append(r)
            r = self.gray_pcnn(img, Y_init, F_init, thresh, gif_name=os.path.splitext(out_name)[0]+'.gif')
            if write_results:
                cv2.imwrite(os.path.join(self.out_path, os.path.splitext(out_name)[0] + 'res.png'), (r*255).astype(np.uint8))
            result.append(r)
        return result
        
