import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import scipy

def Gunner_Farneback(im_stack,pyr_scale=0.5,levels=3,winsize=15):
    """
    im_stack (np.array) depth is temporal sequence of greyscale images ranging from 0 to 1
    @param pyr_scale parameter, specifying the image scale (&lt;1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.

    @param levels number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.

    @param winsize averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    """
    hsv_mask = np.zeros((im_stack.shape[0],im_stack.shape[1],3)) #hue saturation value
    hsv_mask[:,:,1] = 255 
    hsv_mask.shape

    FB_list = []
    for i in range(1,im_stack.shape[2]):
        im_prev = im_stack[:,:,i-1]*255
        im_prev = im_prev.astype(np.uint8)
        im_next = im_stack[:,:,i]*255
        im_next = im_next.astype(np.uint8)
        #calcOpticalFlowFarneback: (prev: Any, next: Any, flow: Any, pyr_scale: Any, levels: Any, winsize: Any, iterations: Any, poly_n: Any, poly_sigma: Any, flags: int)
        flow = cv.calcOpticalFlowFarneback(im_prev,im_next,None,pyr_scale,levels,winsize,4,5,1.2,0)
        #convert flow from vector to polar coordinates and map the angle to diff colour maps
        mag, ang = cv.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees=True) #input the x and y component of velocity
        hsv_mask[:,:,0] = ang/2 #hue is determined by angle
        hsv_mask[:,:,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX) #value
        rgb = cv.cvtColor(np.float32(hsv_mask),cv.COLOR_HSV2RGB).astype(np.uint8)
        FB_list.append(rgb)

    return FB_list

def get_fft_shift(img1,img2):
    # Calculate the discrete 2D Fourier transform of both images.
    img1_fs = np.fft.fft2(img1)
    img2_fs = np.fft.fft2(img2)
    # Calculate the cross-power spectrum by taking the complex conjugate of the second result, 
    # multiplying the Fourier transforms together elementwise, and normalizing this product elementwise.
    cross_power_spectrum = (img1_fs * img2_fs.conj()) / np.abs(img1_fs * img2_fs.conj())
    # inverse
    r = np.abs(np.fft.ifft2(cross_power_spectrum))
    r = np.fft.fftshift(r)
    [py,px] = np.argwhere(r==r.max())[0]
    cx,cy = img2.shape[0]//2,img2.shape[1]//2
    shift_x = cx - px
    shift_y = cy - py
    print(f'Shift measured X:{shift_x}, Y:{shift_y}')
    return shift_x,shift_y
    
def derivatives(I1g, I2g, plot_figure=False):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    # w = window_size//2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255. # normalize pixels
    I2g = I2g / 255. # normalize pixels
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = scipy.signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode) #compute gradient across the x axis through convolving, works the same as taking one column - adjacent column
    fy = scipy.signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode) #compute gradient across the y axis
    ft = scipy.signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + scipy.signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
    laplacian = scipy.ndimage.laplace(I1g)
    d_dict = {'Ix':fx,'Iy':fy,'Laplacian':laplacian,'It':ft,'Image1':I1g,'Image2':I2g}

    def normalise(im):
        max = im.max()
        min = im.min()
        return (im - min)/(max-min)

    if plot_figure is True:
        fig,axes = plt.subplots(2,3,figsize=(15,10))
        for ax, (name,im) in zip(axes.flatten(),d_dict.items()):
            xim = ax.imshow(normalise(im))
            ax.set_title(name)
            plt.colorbar(xim,ax=ax)
        plt.show()
        plt.tight_layout()

    return fx,fy,ft