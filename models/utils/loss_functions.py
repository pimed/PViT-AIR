
'''
https://github.com/loli/medpy/blob/66265de8aedf6259feac00b897a22d0cf173d2e2/medpy/metric/binary.py#L1189

'''
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

import cv2 
import scipy.cluster.hierarchy as hcluster
from scipy import interpolate
import math


####################################################################################################################################
def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance.
    
    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.
    
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
        
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
        
    See also
    --------
    :func:`assd`
    :func:`asd`
    
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd
####################################################################################################################################
def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95
####################################################################################################################################
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds
####################################################################################################################################


####################################################################################################################################
# Stitching Loss
####################################################################################################################################
def distance(mask_tensor1, mask_tensor2, dpi, num_samples=None):
    contour1 = get_contour(mask_tensor1)
    contour2 = get_contour(mask_tensor2)
    line1, line2 = get_matching_lines(contour1, contour2, num_samples)
    dist_mm = np.mean( [np.min( np.linalg.norm((p1-line2)/dpi, axis=1)) for p1 in line1] ) * Inch2MM

    return dist_mm
####################################################################################################################################
def get_contour(mask_tensor, size=500): 
    if mask_tensor.ndim == 3:
        mask = mask_tensor.sum(dim=0).numpy().astype('uint8') 
    else:
        mask = mask_tensor.numpy().astype('uint8') 
        
    scale = size/mask.shape[0]
    mask = cv2.resize(mask, None, fx=scale, fy=scale)
    
    kernel = np.ones((11,11),np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cv2.convexHull( merge_contours(contours))
    return np.array(contour / scale).astype('int')    
####################################################################################################################################
def get_matching_lines(contour1, contour2, num_samples):
    '''return two close corners from each box and in correpsonding order '''
    rect1 = get_corners(contour1)
    rect2 = get_corners(contour2)
    
    TL1, TR1, BR1, BL1 = rect1
    TL2, TR2, BR2, BL2 = rect2
    
    dx, dy = (rect1 - rect2).mean(axis=0)
    
    if abs(dx) > abs(dy):

        if dx < 0:     # corners1: left, corners2: right        
            line1 = crop_contour(contour1, TR1, BR1)
            line2 = np.flip( crop_contour(contour2, BL2, TL2), axis=0 )
        else:         # corners1: right, corners2: left        
            line1 = np.flip( crop_contour(contour1, BL1, TL1), axis=0 )
            line2 = crop_contour(contour2, TR1, BR1)
            
        line1 = resample_line(line1, num_samples, cut='v')
        line2 = resample_line(line2, num_samples, cut='v')
    
    else:
        if dy < 0:   # corners1: top, corners2: bottom                
            line1 = crop_contour(contour1, BR1, BL1)
            line2 = np.flip( crop_contour(contour2, TL2, TR2), axis=0 )

        else:       # corners1: bottom, corners2: top                    
            line1 = np.flip( crop_contour(contour1, TL1, TR1), axis=0 )
            line2 = crop_contour(contour2, BR2, BL2)

        line1 = resample_line(line1, num_samples, cut='h')
        line2 = resample_line(line2, num_samples, cut='h')
        
    return [line1, line2]
####################################################################################################################################
def get_corners(contour, k_param=15):
    W, H = contour.max(axis=(0,1))
    mask = cv2.drawContours(np.zeros((H+10, W+10)), [contour], -1, 255, -1)  # create mask from extracte convex mask
    dst = cv2.cornerHarris( np.float32(mask) , k_param, 3, 0.05)
    Y, X = np.where(dst>0.01*dst.max()) 
    corners = np.array([[x,y] for (x,y) in zip(X,Y)])
    
    clusters = hcluster.fclusterdata(corners, k_param, criterion="distance") # cluster corners to find limited number of corners
    corners = [np.mean(corners[clusters==c], axis=0).astype('int') for c in np.unique(clusters)] # match each corner with a point on the contour
    cnt = contour.reshape(-1,2)
    corners = [cnt[np.argmin(np.sum((cnt-point)**2, axis=1)**.5)] for point in corners]
    
    # find closest corners to each rect_corner (topleft-Clockwise)
    box = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
    corners = [corners[np.argmin(np.sum((corners-point)**2, axis=1)**.5)] for point in box]
    corners = order_points(corners)

    return corners
####################################################################################################################################
def order_points(pts):
    ''' order points from top-left - clockwise '''
    points = np.array(pts)
    rect = np.zeros((4, 2), dtype = "int")
    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect
####################################################################################################################################
def crop_contour(contour, p1, p2):
    a = np.argmin(np.sum((contour.reshape(-1,2)-p1)**2, axis=1)**.5)
    b = np.argmin(np.sum((contour.reshape(-1,2)-p2)**2, axis=1)**.5)
    if a < b : return np.array(contour[a:b+1]).reshape(-1,2)
    else: return np.array([*contour[a:], *contour[:b+1]]).reshape(-1,2)
####################################################################################################################################
def resample_line(line, num_samples, cut='h'):
    if num_samples is None: 
        return line
    
    if cut == 'h':
        f = interpolate.interp1d(line[:,0], line[:,1], kind='nearest')
        x_new = np.linspace(line[0,0],line[-1,0], num_samples)
        y_new = f( x_new )
    
    elif cut =='v':
        f = interpolate.interp1d(line[:,1], line[:,0], kind='nearest')
        y_new = np.linspace(line[0,1],line[-1,1], num_samples)
        x_new = f( y_new )
        
    return np.array([[x,y] for (x,y) in zip(x_new,y_new)]).astype('int')
####################################################################################################################################
class clockwise_angle_and_distance():
    def __init__(self, origin):
        self.origin = origin
    # ------------------------------------------------------------------------------------
    def __call__(self, point, refvec = [0, 1]):
        if self.origin is None:
            raise NameError("clockwise sorting needs an origin. Please set origin.")
        
        vector = [point[0]-self.origin[0], point[1]-self.origin[1]]
        lenvector = np.linalg.norm(vector[0] - vector[1])
        if lenvector == 0: return - math.pi, 0

        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1] # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)

        if angle < 0:
            return 2*math.pi+angle, lenvector

        return angle, lenvector
####################################################################################################################################
def merge_contours(cnts):
    if len(cnts) > 1: 
        list_of_pts = np.vstack(cnts).reshape(-1,2)
        clock_ang_dist = clockwise_angle_and_distance( np.mean(list_of_pts, axis=0) ) # set origin
        cnts = sorted(list_of_pts, key=clock_ang_dist) # use to sort

    return np.array(cnts, dtype=np.int32).reshape((-1,1,2))
####################################################################################################################################

