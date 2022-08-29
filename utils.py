import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib import colors, colorbar,cm
import matplotlib as mpl
import skimage.io
from skimage import filters
from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
import scipy
from math import ceil
import cv2 as cv
from math import sin,cos, tan,pi


def shift_images(shift_x,shift_y,frame,dim,x0,y0):
    """
    shift_x,shift_y = how much shift to apply
    frame (full image)
    dim = dimension of the cropped window
    x0,y0 = specify the starting coord
    returns an image list that is shifted by shift_x and shift_y
    returns a list of coord tuples (lower left coord, dim) of the shifted image frame corresponding to image list
    """
    frame_grey = rgb2gray(frame)
    nrow,ncol = frame_grey.shape
    y0_new = y0
    x0_new = x0
    image_frames = []
    coord_list = []
    while (y0_new + dim < nrow) and (y0_new >= 0) and (x0_new + dim < ncol) and (x0_new >= 0):
        img = frame_grey[y0_new:y0_new+dim,x0_new:x0_new+dim]
        image_frames.append(img)
        coord_list.append(((x0_new,y0_new+dim),dim))
        y0_new += shift_y
        x0_new += shift_x
    #stack all images tgth
    im_stack = np.stack(image_frames,axis=2)
    return im_stack, coord_list

def show_image_sequence(im_stack,step_size):
    """
    im_stack (np.array) depth is temporal sequence of greyscale images ranging from 0 to 1
    step_size (int): show image every step_size
    """
    
    fig,axes = plt.subplots(4,4,figsize=(20,20)) # plot every 10 image
    if type(im_stack) is list:
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(im_stack[i*step_size])

        plt.show()

    else:
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(im_stack[:,:,i*step_size])

        plt.show()
    return


def animate_u_v(im_stack,u_stack,v_stack,fname,fps=30):
    fig, ax = plt.subplots(1,4,figsize=(20,5))
    im_u = ax[0].imshow(u_stack[:,:,0])
    im_v = ax[1].imshow(v_stack[:,:,0])
    uv_sq = u_stack[:,:,0]**2 + v_stack[:,:,0]**2
    im_sq = ax[2].imshow(uv_sq)
    im = ax[3].imshow(im_stack[:,:,0])
    axes = [im_u,im_v,im_sq,im]

    for a,labels in zip(ax.flatten(),['u','v',r'$u^2 + v^2$']):
        a.axis('off')
        a.set_title(labels)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        axes[0].set_array(u_stack[:,:,i])
        axes[1].set_array(v_stack[:,:,i])
        uv_sq = u_stack[:,:,i]**2 + v_stack[:,:,i]**2
        axes[2].set_array(uv_sq)
        axes[3].set_array(im_stack[:,:,i])

        return axes

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = u_stack.shape[2]-1,#nSeconds * fps,
                                interval = 1000 / fps, # in ms
                                )

    anim.save('{}.mp4'.format(fname), fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Done!')
    return

def animate_OF(im_stack,u_stack,v_stack,fname,fps=30):
    """
    flow_int (int): Flow interval
    """
    fig = plt.figure(figsize=(8,8))
    im = plt.imshow(im_stack[:,:,0])
    X,Y = np.meshgrid(list(range(im_stack.shape[1])),list(range(im_stack.shape[0])))
    quiver = plt.quiver(X,Y,u_stack[:,:,0],v_stack[:,:,0],minshaft = 1, minlength=0)
    axes = [im,quiver]

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        axes[0].set_array(im_stack[:,:,i])
        axes[1].set_UVC(u_stack[:,:,i],v_stack[:,:,i])
        return axes

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = u_stack.shape[2]-1,
                                interval = 1000 / fps, # in ms
                                blit=False
                                )
                            
    anim.save('{}.mp4'.format(fname), fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Done!')
    return

def animate_frames(image_frames,fname,fps=30):
    """
    fps = frame per second
    nSeconds = length of video
    image_frames (list of np arrays or stacked np.array): containing the temporal sequence
    """

    snapshots = image_frames#[ np.random.rand(5,5) for _ in range( nSeconds * fps ) ]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )

    if type(image_frames) is list:
        a = image_frames[0]
    else:
        a = image_frames[:,:,0]

    print(f'Max: {np.max(a)}, Min: {np.min(a)}')

    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

    def animate_func_list(i):
        if i % fps == 0:
            print( '.', end ='' )

        im.set_array(image_frames[i])
        return [im]

    def animate_func_stack(i):
        if i % fps == 0:
            print( '.', end ='' )

        im.set_array(image_frames[:,:,i])
        return [im]
    
    if type(image_frames) is list:
        anim = animation.FuncAnimation(
                                    fig, 
                                    animate_func_list, 
                                    frames = len(image_frames)-1,
                                    interval = 1000 / fps, # in ms
                                    )
    else:
        anim = animation.FuncAnimation(
                                    fig, 
                                    animate_func_stack, 
                                    frames = image_frames.shape[-1],
                                    interval = 1000 / fps, # in ms
                                    )

    anim.save('{}.mp4'.format(fname), fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Done!')
    return

def hsv_colour_wheel():
    fg = plt.figure(figsize=(5,5))
    ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')

    # Define colormap normalization for 0 to 2*pi
    norm = mpl.colors.Normalize(0, 2*np.pi) 

    # Plot a color mesh on the polar plot
    # with the color set by the angle

    n = 200  #the number of secants for the mesh
    t = np.linspace(0,2*np.pi,n)   #theta values
    r = np.linspace(.6,1,2)        #radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
    c = tg                         #define color values as theta value
    im = ax.pcolormesh(t, r, c.T,norm=norm,cmap='hsv')  #plot the colormesh on axis with colormap
    ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
    ax.tick_params(pad=15,labelsize=24)      #cosmetic changes to tick labels
    ax.spines['polar'].set_visible(False)    #turn off the axis spine.
    plt.show()
    return

def binarize_obj(im_stack):
    """
    im_stack (np.array) depth is temporal sequence of images
    create binary image using otsu thresholding
    """
    closing_stack = np.zeros(im_stack.shape)
    for i in range(im_stack.shape[2]):
        im = im_stack[:,:,i]*255
        im = im.astype(np.uint8)
        # Otsu's thresholding after Gaussian filtering
        blur = cv.GaussianBlur(im,(5,5),0)
        ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        #morphological operation
        kernel = np.ones((50,50),np.uint8)
        closing = cv.morphologyEx(th3/255, cv.MORPH_CLOSE, kernel).astype(np.uint8)
        closing_stack[:,:,i] = closing
    
    return closing_stack

def image_pyramid(im,levels):
    """
    im (np.array): greyscale image
    levels (int): number of levels of downsampling image (reducing resolution)
    e.g. levels == 1, means only downsampled once
    pyr_list (list): to store the image pyramid, where pyr_list[0] is the original image
    """
    
    # if levels == 0:
    #     return im

    # else:
    #     nrow,ncol = im.shape[0],im.shape[1]
    #     im_down = cv.pyrDown(image_pyramid(im,levels-1))

    pyr_dict = {i: None for i in range(levels+1)}
    pyr_dict[0] = im
    im_copy = im.copy()

    for i in range(levels):
        nrows,ncols = im_copy.shape[0], im_copy.shape[1]
        im_copy = cv.pyrDown(im_copy, dstsize=(ncols // 2, nrows // 2))
        pyr_dict[i+1] = im_copy

    return pyr_dict