import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from matplotlib import colors, colorbar,cm
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
    image_frames (list of np arrays): containing the temporal sequence
    """

    snapshots = image_frames#[ np.random.rand(5,5) for _ in range( nSeconds * fps ) ]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(8,8) )

    a = image_frames[0]
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )

        im.set_array(image_frames[i])
        return [im]

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = len(image_frames)-1,
                                interval = 1000 / fps, # in ms
                                )

    anim.save('{}.mp4'.format(fname), fps=fps, extra_args=['-vcodec', 'libx264'])

    print('Done!')
    return