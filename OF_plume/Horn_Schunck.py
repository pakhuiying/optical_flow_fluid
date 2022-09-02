from unittest.mock import NonCallableMagicMock
from utils import *

def computeAvg(x,kernel,boundaryCondition='periodical'):
    """
    Return the local average matrix
    """
    if boundaryCondition == 'periodical':
        x1 = np.row_stack([x[-1,:],x,x[0,:]])
        x = np.column_stack([x1[:,-1],x1,x1[:,0]])
    
    else:
        x1 = np.row_stack([x[0,:],x,x[-1,:]])
        x = np.column_stack([x1[:,0],x1,x1[:,-1]])
    
    xAvg = scipy.signal.convolve2d(x,kernel,mode='same')
    xAvg = xAvg[1:-1,1:-1]
    return xAvg

def computeCurl(uv):
    u = uv[:,:,0]
    v = uv[:,:,1]

    # simple kernel
    # add boundary
    u1 = np.row_stack((u[0,:],u,u[-1,:]))
    u = np.column_stack((u1[0,:],u1,u1[-1,:]))
    # add boundary
    v1 = np.row_stack((v[0,:],v,v[-1,:]))
    v = np.column_stack((v1[:,0],v1,v1[:,-1]))

    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])*1/8 #used to detect vertical lines
    uy = scipy.signal.convolve2d(u,kernel.T,mode='same') #detect horizontal lines for x velocity
    vx = scipy.signal.convolve2d(v,kernel,mode='same') #detect vertical lines for y velocity
    uy = uy[1:-1,1:-1]
    vx = vx[1:-1,1:-1]
    vort = vx - uy

    # complex kernel
    h = np.array([[-1, 9, -45, 0, 45, -9, 1]])/60;        # derivative used by Bruhn et al "combing "IJCV05' page218
    vx = scipy.ndimage.correlate(v, h, mode='reflect') #MATLAB returns transpose of the expected answer
    uy = scipy.ndimage.correlate(u, h.T, mode='reflect')
    vort = vx - uy
    return vort

def computeDerivatives_f(im1,im2,boundaryCondition='periodical'):
    """
    compute Ix,Iy,It
    """
    # compute the local average
    kernel = np.ones((3,3))
    kernel[1,1] = 8
    kernel = kernel*1/16
    im1 = scipy.ndimage.correlate(im1,kernel, mode='reflect')
    im2 = scipy.ndimage.correlate(im2,kernel, mode='reflect')

    # # Horn-Schunck original method
    # kernel_x = np.array([[-1., 1.], [-1., 1.]])
    # kernel_y = np.array([[-1., -1.], [1., 1.]])
    # kernel_t = np.array([[1., 1.], [1., 1.]])
    # fx = scipy.signal.convolve2d(im1,kernel_x*0.25,mode='same') + scipy.signal.convolve2d(im2,kernel_x*0.25,mode='same')
    # fy = scipy.signal.convolve2d(im1,kernel_y*0.25,mode='same') + scipy.signal.convolve2d(im2,kernel_y*0.25,mode='same')
    # ft = scipy.signal.convolve2d(im1,kernel_t*0.25,mode='same') + scipy.signal.convolve2d(im2,kernel_t*-0.25,mode='same')

    # # Derivatives as in Barron 2005
    # kernel = np.array([-1,8,0,-8,1])*1/12
    # fx = scipy.signal.convolve2d(im1,kernel,mode='same')
    # fy = scipy.signal.convolve2d(im1,kernel,mode='same')
    # ft = scipy.signal.convolve2d(im1,np.ones((2,2))*0.25,mode='same') + scipy.signal.convolve2d(im2,np.ones((2,2))*-0.25,mode='same')

    # # An alternative way to compute the spatiotemporal derivatives is to use simple finite difference masks.
    # fx = scipy.signal.convolve2d(im1,np.array([1,-1]),mode='same')
    # fy = scipy.signal.convolve2d(im1,np.array([1],[-1]),mode='same')
    # ft = im2 - im1

    # derivative used by Bruhn et al "combing "IJCV05' page218
    h = np.array([[-1,9,-45,0,45,-9,1]])/60
    # h = np.array([1,-8,0,8,-1])/12 # used in Wedel et al "improved TV L1"
    ft = im2 - im1

    if boundaryCondition == 'periodical':
        bounMargin = 3
        im1_temp = np.row_stack([im1[-1-bounMargin+1:,:],im1,im1[:bounMargin,:]])
        im1_period = np.column_stack([im1_temp[:,-1-bounMargin+1:],im1_temp,im1_temp[:,:bounMargin]])

        im2_temp = np.row_stack([im2[-1-bounMargin+1:,:],im2,im2[:bounMargin,:]])
        im2_period = np.column_stack([im2_temp[:,-1-bounMargin+1:],im2_temp,im2_temp[:,:bounMargin]])

        fx = 0.5*(scipy.ndimage.correlate(im1_period,h, mode='reflect')+ scipy.ndimage.correlate(im2_period, h, mode='reflect'))
        fy = 0.5*(scipy.ndimage.correlate(im1_period,h.T, mode='reflect')+ scipy.ndimage.correlate(im2_period, h.T, mode='reflect'))
        fx = fx[bounMargin:-bounMargin,bounMargin:-bounMargin]
        fy = fy[bounMargin:-bounMargin,bounMargin:-bounMargin]

    else:
        fx = 0.5*(scipy.ndimage.correlate(im1,h, mode='reflect')+ scipy.ndimage.correlate(im2, h, mode='reflect'))
        fy = 0.5*(scipy.ndimage.correlate(im1,h.T, mode='reflect')+ scipy.ndimage.correlate(im2, h.T, mode='reflect'))
    
    # laplace operator
    w = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    fxy = 0.5*(scipy.ndimage.correlate(im1,w, mode='reflect')+ scipy.ndimage.correlate(im2, w, mode='reflect'))

    f_avg = smoothImg(im2,1) #sigma=1
    f_avg_avg = smoothImg(f_avg,1) #sigma=1
    

    return fx, fy, ft, fxy

def flowAngErr(tu,tv,u,v,bord=0):
    """
    returns the Barron et al angular error
    bord (int) pixel width of the border to be ignored
    """

    smallflow = 0
    nrow, ncol = tu.shape[0], tu.shape[1]
    # ignore the border of image if necessary
    stu = tu[bord:nrow-bord,bord:ncol-bord]
    stv = tv[bord:nrow-bord,bord:ncol-bord]
    su = u[bord:nrow-bord,bord:ncol-bord]
    sv = v[bord:nrow-bord,bord:ncol-bord]

    # ignore the points whose velocities are zero
    # ignore a pixel if both u and v are zero 
    stu_abs = np.abs(stu).flatten('F') #flatten the arrays column wise
    stv_abs = np.abs(stv).flatten('F')
    ind2 = np.argwhere(stu_abs>smallflow | stv_abs>smallflow)
    ind2 = ind2.T[0] #all the indices will be arranged in (n,1). transpose to (1,n) then just change it to one row
    ind2 = np.unravel_index(ind2,(stu.shape[0],stu.shape[1]),order='F')
    # compute RMSE
    DIS = (stu-su)**2 + (stv-sv)**2
    DIS = DIS[ind2]
    DIS_sum = np.mean(DIS)
    RMSE = np.sqrt(DIS_sum)

    return RMSE

def gaussFilter(kSize,sigma=1):
    """
    Creates a 1-D Gaussian kernel of a standard deviation 'segma' and a size
    of 'kSize'. 

    In theory, the Gaussian distribution is non-zero everywhere. In practice,
    it's effectively zero at places further away from about three standard
    deviations. Hence the reason why the kernel is suggested to be truncated
    at that point.

    The 2D Gaussian filter is a complete circular symmetric operator. It can be
    seperated into x and y components. The 2D convolution can be performed by
    first convolving with 1D Gaussian in the x direction and the same in the
    y direction.

    Author: Mohd Kharbat at Cranfield Defence and Security
    mkharbat(at)ieee(dot)org , http://mohd.kharbat.com
    Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/

    October 2008
    """

    kSize = 2*(sigma*3)

    x = np.linspace(-(kSize/2),kSize/2,kSize,endpoint=True)
    G = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp (-(x**2)/(2*sigma**2))
    return G

def smoothGT(w,sigma,boundaryCondition='periodical',bounMargin=3):
    """
    smooth ground truth
    """
    if boundaryCondition=='periodical':
        w_temp = np.row_stack([w[-1-bounMargin+1:,:],w,w[:bounMargin,:]])
        w_period = np.column_stack([w_temp[:,-1-bounMargin+1:],w_temp,w_temp[:,:bounMargin]])
        w_smooth = smoothImg(w_period,sigma)
        w_smooth = w_smooth[bounMargin:-bounMargin,bounMargin:-bounMargin]
    else:
        w_smooth = smoothImg(w,sigma)
    
    return w_smooth

def smoothImg(im,sigma=1):
    """
    Gaussian filtering
    Convolving an image with a Gaussian kernel.

    The degree of smoothing is determined by the Gaussian's standard
    deviation 'segma'. The Gaussian outputs a 'weighted average' of each
    pixel's neighborhood, with the average weighted more towards the value of
    the central pixels. The larger the standard deviation, the less weight
    towards the central pixels and more weight towards the pixels away, hence
    heavier blurring effect and less details in the output image.

    Author: Mohd Kharbat at Cranfield Defence and Security
    mkharbat(at)ieee(dot)org , http://mohd.kharbat.com
    Published under a Creative Commons Attribution-Non-Commercial-Share Alike
    3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/

    October 2008
    """
    kSize = 2*(sigma*3)
    G = gaussFilter(kSize,sigma)
    smoothedImg = scipy.signal.convolve2d(im,G,mode='same')
    smoothedImg = scipy.signal.convolve2d(smoothedImg,G.T,mode='same')

    return smoothedImg


def HS_estimation(im1,im2,uLast,vLast,lamb=1,iter=200,boundaryCondition='periodical',estimation_method='HS'):
    """
    Adapted from Shengze Cai
    https://github.com/shengzesnail/coarse_to_fine_HS_PIV/blob/master/HS_Estimation.m
    im1,im2 (np.array): two subsequent frames or images.
    lamb (float): lambda, a parameter that reflects the influence of the smoothness term.
    iter (int): number of iterations.
    uLast, vLast (float): the flow field estimates of last pyramid; default is zero.
    estimationMethod (str): either HS or TE (where transport equation is considered in the constraint)
    """
    uInitial = uLast
    vInitial = vLast
    u = uInitial
    v = vInitial

    # Estimate Spatiotemporal derivatives of images
    fx, fy, ft, fxy = computeDerivatives_f(im1, im2, boundaryCondition)

    # averaging kernel
    kernel_1=np.array([[0, 1/4, 0],[1/4, 0 ,1/4],[0, 1/4, 0]])

    for i in range(iter):
        # Compute local averages of the flow vectors
        uAvg = computeAvg(u,kernel_1,boundaryCondition)
        vAvg = computeAvg(v,kernel_1,boundaryCondition)
        if estimation_method == 'HS':
            Diffu = 0
        else:
            Diffu = -(1/Re/Sc)*fxy
        
        #Compute flow vectors constrained by its local average and the optical flow constraints
        data = ( fx * (uAvg-uLast) ) + ( fy * (vAvg-vLast) ) + ft + Diffu
        u = uAvg - ( fx * ( data ) ) / ( lamb + fx**2 + fy**2)
        v = vAvg - ( fy * ( data ) )/ ( lamb + fx**2 + fy**2)
    
    u = np.nan_to_num(u) #converts nan to 0
    v = np.nan_to_num(v) #converts nan to 0

    return u,v

def expand2(ori,interpolation_method='linear'):
    """
    The Expansion Function for pyramid
    Project the velocity field to next pyramid level
    """
    m,n = ori.shape
    m1 = m * 2 
    n1 = n * 2
    result1 = np.zeros((m1,n1))
    mid = np.zeros((m,n1))
    w = np.array([[0.5, 1, 0.5]])

    for j in range(m): #iterate across row
        t = np.zeros((1,n1))
        for i in range(0,n1,2): #expand the columns
            t[0,i] = ori[j,i//2] 
        conv_t = np.column_stack((0,t,ori[j,0])) #pad with 0 and first column
        tmp = scipy.signal.convolve(conv_t,w,mode='same') #to average out between across alternate columns
        mid[j,:n1] = tmp[:,1:-1]
    
    for i in range(n1): # iterate across the expanded columns of mid
        t = np.zeros((1,m1))
        for j in range(0,m1,2): #expand the rows now
            t[0,j] = mid[j//2,i]
        conv_t = np.column_stack((0,t,mid[0,i])) #pad with 0 and first row
        tmp = scipy.signal.convolve(conv_t,w,mode='same')
        result1[:,i] = tmp[:,1:-1]

    m,n = result1.shape[0], result1.shape[1]
    x1 = np.linspace(1,n,n,endpoint=True)# - 1/2
    y1 = np.linspace(1,m,m,endpoint=True)# - 1/2
    x,y = np.meshgrid(x1,y1)
    
    #define new grid coordinates to interpolate subpixel
    x = x - 1/2
    y = y - 1/2

    for i in range(m):
        for j in range(n):
            if x[i,j]> n:
                x[i,j] = n
            if x[i,j] < 1:
                x[i,j] = 1
            if y[i,j] > m:
                y[i,j] = m
            if y[i,j] < 1:
                y[i,j] = 1

    if interpolation_method == 'bi-cubic':
        h = np.array([[1, -8,0,8,-1]])/12 # used in Wedel etal "improved TV L1"
        result = interp2_bicubic(result1,x,y,h)
    else:
        result = scipy.interpolate.griddata(np.column_stack((x1.ravel(),y1.ravel())), result1.ravel(), (x, y), method='linear')
    
    return result

def interp2_bicubic(Z,XI,YI,Dxfilter=None):
    """
    Author:  Stefan Roth, Department of Computer Science, TU Darmstadt
    Contact: sroth@cs.tu-darmstadt.de
    $Date$
    $Revision$

    Copyright 2004-2007, Brown University, Providence, RI. USA
    Copyright 2007-2010 TU Darmstadt, Darmstadt, Germany.
    modified by dqsun
    """
    if Dxfilter is None: 
        Dxfilter = np.array([[-0.5,0,0.5]])

    Dyfilter = Dxfilter.T
    Dxyfilter = scipy.signal.convolve2d(Dxfilter,Dyfilter,mode='full')

    input_size = XI.shape
    # Reshape input coordinates into a vector
    XI = XI.flatten('F') #flatten column wise
    YI = YI.flatten('F')

    # Bound coordinates to valid region
    sx = Z.shape[1]
    sy = Z.shape[0]

    # Neighbor coordinates
    fXI = XI.astype(int) #same as taking floor
    cXI = fXI + 1
    fYI = YI.astype(int) #same as taking floor
    cYI = fYI + 1

    indx = (fXI<1) | (cXI>sx) | (fYI<1) | (cYI>sy) #boolean array
    # indx = indx.astype(int) #logical array

    matlab_min = lambda m,vec: np.array([v if v < m else m for v in vec])
    matlab_max = lambda m, vec: np.array([m if v < m else v for v in vec])

    fXI = matlab_max(1, matlab_min(sx, fXI))
    cXI = matlab_max(1, matlab_min(sx, cXI))
    fYI = matlab_max(1, matlab_min(sy, fYI))
    cYI = matlab_max(1, matlab_min(sy, cYI))

    # Image at 4 neighbors
    z00_idx = fYI + sy * (fXI - 1) #dtype is still int
    z00_idx = np.unravel_index(z00_idx,(Z.shape[0],Z.shape[1]),order='F')
    z01_idx = cYI + sy * (fXI - 1)
    z01_idx = np.unravel_index(z01_idx,(Z.shape[0],Z.shape[1]),order='F')
    z10_idx = fYI + sy * (cXI - 1)
    z10_idx = np.unravel_index(z10_idx,(Z.shape[0],Z.shape[1]),order='F')
    z11_idx = cYI + sy * (cXI - 1)
    z11_idx = np.unravel_index(z11_idx,(Z.shape[0],Z.shape[1]),order='F')

    idx_list = [z00_idx,z10_idx,z11_idx,z01_idx]
    Z_list = [Z[idx] for idx in idx_list]

    # x-derivative at 4 neighbors
    DX = scipy.ndimage.correlate(Z, Dxfilter,mode='reflect')
    DX_list = [DX[idx] for idx in idx_list]
    # y-derivative at 4 neighbors
    DY = scipy.ndimage.correlate(Z, Dyfilter,mode='reflect')
    DY_list = [DY[idx] for idx in idx_list]
    # xy-derivative at 4 neighbors
    DXY = scipy.ndimage.correlate(Z, Dxyfilter,mode='reflect')
    DXY_list = [DXY[idx] for idx in idx_list]

    mat_list = Z_list + DX_list + DY_list + DXY_list
    V = np.row_stack(mat_list)

    W = np.array([ 
        [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
        [-3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1,  0,  0,  0,  0],  
        [2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
        [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0,  0, -3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1],
        [0,  0,  0,  0,  2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1],
        [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
        [9, -9,  9, -9,  6,  3, -3, -6,  6, -6, -3,  3,  4,  2,  1,  2],
        [-6,  6, -6,  6, -4, -2,  2,  4, -3,  3,  3, -3, -2, -1, -1, -2],
        [2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
        [-6,  6, -6,  6, -3, -3,  3,  3, -4,  4,  2, -2, -2, -2, -1, -1],
        [4, -4,  4, -4,  2,  2, -2, -2,  2, -2, -2,  2,  1,  1,  1,  1]
        ])

    C = W*V

    alpha_x = XI - fXI
    alpha_x = alpha_x.reshape(input_size,order='F')
    alpha_y = YI - fYI
    alpha_y = alpha_y.reshape(input_size,order='F')

    # Clip out-of-boundary pixels to boundary
    # Modified by Deqing Sun (7-10-2008)
    alpha_x[indx] = 0
    alpha_y[indx] = 0

    fXI = fXI.reshape(input_size,order='F')
    fYI = fYI.reshape(input_size,order='F')

    # interpolation
    ZI = np.zeros(input_size)
    ZXI = np.zeros(input_size)
    ZYI = np.zeros(input_size)

    idx = 0
    for i in range(0,4):
        for j in range(0,4):
            ZI = ZI + C[idx,:].reshape(input_size,order='F')*(alpha_x**i)*(alpha_y**j)
            if (i>0):
                ZXI = ZXI + i*C[idx,:].reshape(input_size,order='F')*(alpha_x**(i-1))*(alpha_y**j)
            if (j>0):
                ZYI = ZYI + j*C[idx,:].reshape(input_size,order='F')*(alpha_x**i)*(alpha_y**(j-1))
            idx += 1

    ZI[indx] = np.nan

    return ZI, ZXI, ZYI, C, alpha_x, alpha_y, fXI, fYI

def reduce2(ori,interpolation_method='linear'):

    #Step 1: low-pass filtering
    w = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    mid = scipy.signal.convolve2d(ori,w,mode='same')

    #Step 2: interpretation
    m,n = ori.shape[0], ori.shape[1]
    x1 = np.linspace(1,n,n,endpoint=True)# - 1/2
    y1 = np.linspace(1,m,m,endpoint=True)# - 1/2
    x,y = np.meshgrid(x1,y1)
    x = x + 1/2
    y = y + 1/2

    for i in range(m):
        for j in range(n):
            if x[i,j] > n:
                x[i,j] = n
            if x[i,j] < 1:
                x[i,j] = 1
            if y[i,j] > m:
                y[i,j] = m
            if y[i,j] < 1:
                y[i,j] = 1

    if interpolation_method == 'bi-cubic':
        h = np.array([[1,-8,0,8,-1]])/12
        temp = interp2_bicubic(mid,x,y,h)

    else:
        temp = scipy.interpolate.griddata(np.column_stack((x1.ravel(),y1.ravel())), mid.ravel(), (x, y), method='linear')
    
    result = temp[::2,::2] #take every odd row,column (downsampling the array)
    
    return result

def resample_flow(uv,sz,method='bilinear'):
    """
    RESAMPLE_FLOW   Resample flow field
    RESAMPLE_FLOW(IN, FACTOR[, METHOD]) resamples (resizes) the flow
    field IN using a factor of FACTOR.  The optional argument METHOD
    specifies the interpolation method ('bilinear' (default) or
    'bicubic'). 
    sz (tuple): (nrow,ncol)
    
    This is a private member function of the class 'clg_2d_optical_flow'. 

    Author:  Stefan Roth, Department of Computer Science, TU Darmstadt
    Contact: sroth@cs.tu-darmstadt.de
    """
    ratio = sz[0] / uv.shape[0]
    if method == 'bilinear':
        u = cv.resize(uv[:,:,0],(sz[1],sz[0]),interpolation=cv.INTER_LINEAR)*ratio
        v = cv.resize(uv[:,:,1],(sz[1],sz[0]),interpolation=cv.INTER_LINEAR)*ratio
        out = np.stack((u,v),axis=2)
    else: #bicubic interpolation
        u = cv.resize(uv[:,:,0],(sz[1],sz[0]),interpolation=cv.INTER_CUBIC)*ratio
        v = cv.resize(uv[:,:,1],(sz[1],sz[0]),interpolation=cv.INTER_CUBIC)*ratio
        out = np.stack((u,v),axis=2)
    
    return out

def warp_forward(img, Dx, Dy, interpolation_method='spline'):
    """
    Symmetric Warping and Interpolation
    """
    m,n = img.shape[0], img.shape[1]
    x1 = np.linspace(1,n,n,endpoint=True)# - 1/2
    y1 = np.linspace(1,m,m,endpoint=True)# - 1/2
    x,y = np.meshgrid(x1,y1)
    x = x - 1/2*Dx[:m,:n]
    y = y - 1/2*Dy[:m,:n]

    for i in range(m):
        for j in range(n):
            if x[i,j] > n:
                x[i,j] = n
            if x[i,j] < 1:
                x[i,j] = 1
            if y[i,j] > m:
                y[i,j] = m
            if y[i,j] < 1:
                y[i,j] = 1
            
    if interpolation_method == 'bi-cubic':
        h = np.array([[1,-8,0,8,-1]])/12
        result = interp2_bicubic(img,x,y,h)
    else:
        result = scipy.interpolate.griddata(np.column_stack((x1.ravel(),y1.ravel())), img.ravel(), (x, y), method='linear')
    
    return result

def warp_inverse(img,Dx,Dy,interpolation_method='spline'):
    """
    Symmetric Warping and Interpolation
    """
    m,n = img.shape[0], img.shape[1]
    x1 = np.linspace(1,n,n,endpoint=True)# - 1/2
    y1 = np.linspace(1,m,m,endpoint=True)# - 1/2
    x,y = np.meshgrid(x1,y1)

    x = x + 1/2*Dx[:m,:n]
    y = y + 1/2*Dy[:m,:n]

    for i in range(m):
        for j in range(n):
            if x[i,j] > n:
                x[i,j] = n
            if x[i,j] < 1:
                x[i,j] = 1
            if y[i,j] > m:
                y[i,j] = m
            if y[i,j] < 1:
                y[i,j] = 1
            
    if interpolation_method == 'bi-cubic':
        h = np.array([[1,-8,0,8,-1]])/12
        result = interp2_bicubic(img,x,y,h) #used in Wedel etal "improved TV L1"
    else:
        result = scipy.interpolate.griddata(np.column_stack((x1.ravel(),y1.ravel())), img.ravel(), (x, y), method='linear')
    
    return result

def HS_pyramids(im1,im2,lamb,PARA,uvInitial):
    """
    Horn-Schunck Motion Estimation Using Multi-Pyramids (Multi-Resolution)
    Ref:	
    Ruhnau, P., et al. (2005) Experiments in Fluids 38(1):21-32.
    Heitz, D., et al. (2010) Experiments in Fluids, 48(3):369-393.
    Sun, D., et al. (2010). Computer Vision & Pattern Recognition.

    Usage: [u, v] = HS_Pyramids(img1,img2,lambda,PARA)
    ********** inputs ***********
    im1,im2 (np.array): two subsequent frames or images (greyscale).
    lambda (float): a parameter that reflects the influence of the smoothness term.
    PARA: parameters
    ********** outputs ************
    u, v: the velocity components

    Shengze Cai, 2016/03
    """
    # initialise the velocity field
    if uvInitial is None:
        uInitial = np.zeros(im1.shape)
        vInitial = np.zeros(im1.shape)
        uvInitial = np.stack([uInitial,vInitial],axis=2)

    if len(im1.shape) > 2:
        im1 = rgb2gray(im1)

    if len(im2.shape) > 2:
        im2 = rgb2gray(im2)

    def Estimate(im1,im2,lamb,uvInitial,PARA):
        # Run Horn_schunck on all levels and interpolate
        warp_iter = PARA.warp_iter
        sizeOfMF = PARA.sizeOfMF
        isMedianFilter = PARA.isMedianFilter
        Dx = uvInitial[:,:,0] #delta x
        Dy = uvInitial[:,:,1] #de;ta y

        #consruct image pyramid for gnc stage 1
        pyramid_level = PARA.pyramid_level
        G1 = image_pyramid(im1,pyramid_level)
        G2 = image_pyramid(im2,pyramid_level)
        # iteration
        level = pyramid_level
        for l in range(level+1):
            small_im1 = G1[l]
            small_im2 = G2[l]
            sz = small_im1.shape
            uv = resample_flow(np.stack([Dx,Dy],axis=2),sz)
            Dx = uv[:,:,0]
            Dy = uv[:,:,1]

            for iwarp in range(warp_iter):
                W1 = warp_forward(small_im1,Dx,Dy,PARA.interpolation_method)
                W2 = warp_inverse(small_im2, Dx,Dy,PARA.interpolation_method)
                Dx,Dy = HS_estimation(W1,W2,lamb, PARA.iter,Dx,Dy,PARA.boundaryCondition)

                if (isMedianFilter is True):
                    Dx = scipy.ndimage.median_filter(Dx,sizeOfMF,mode='reflect')
                    Dy = scipy.ndimage.median_filter(Dy,sizeOfMF,mode='reflect')

        return Dx, Dy
    
    # Run HS with multi-pyramids
    u, v = Estimate(im1, im2, lamb, uvInitial,PARA)

    return u,v
    
class PARA:
    """
    parameters settings
    """
    def __init__(self,pyramid_level,warp_iter,iter,boundaryCondition='periodical',interpolation_method='spline',isMedianFilter=True,sizeOfMF=(5,5)):
        self.pyramid_level = pyramid_level
        self.warp_iter = warp_iter
        self.iter = iter
        #Boundary conditions
        self.boundaryCondition = boundaryCondition
        self.interpolation_method = interpolation_method
        #Divergence_free decomposition and the settings
        self.isMedianFilter = isMedianFilter
        self.sizeOfMF = sizeOfMF
    




