import os
import math
import numpy as np
from numba import jit
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def refl_to_int(ref):
    """
    calculate convolutions
    """
    return np.floor(ref.copy()/5)*5

def _distance(a, b):
    """
    distance in pixel coordinates
    """
    return  math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

@jit
def lbp_filter(x):
    """
    Local Binary Pattern Filter
    Counts number of changes from 0 to 1 in x
    Described in Yuan et al. 2018 section 2.4.1
    """
    #remapping to circular order for LBP calculations 
    circular_remap3 = np.array([0,1,2,4,7,6,5,3])
    circular_remap5 = np.array([0,1,2,3,4,6,8,10,15,14,13,12,11,9,7,5])
    #remap to circular array
    if len(x) == 8:
        x = x[circular_remap3]
    elif len(x)==16:
        x = x[circular_remap5]
    #wrap array
    wrap_x     = np.empty((len(x)+1))
    wrap_x[0]  = x[-1]
    wrap_x[1:] = x
    #use abs of diff to find changes
    change_x = np.abs(wrap_x[1:] - wrap_x[:-1])
    #sum changes
    change_count = np.sum(change_x)
    return change_count


def point_line_distance(point, start, end):
    """
    generate line of points between a start and end point
    """
    if (start == end):
        return _distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = math.sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d

def vector_pair_angle(x_list, y_list, step_size):
    """
    Returns the angles between list of i and j coordinates.

    based off https://stackoverflow.com/questions/14631776/calculate-turning-points-pivot-points-in-trajectory-path

    Parameters:
    dir is a 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    The return value is a 1D-array of values of shape (N-1,), with each value
    between 0 and pi.

    0 implies the vectors point in the same direction
    pi/2 implies the vectors are orthogonal
    pi implies the vectors point in opposite directions
    """
    
    vec_start = []
    vec_end = []
    
    centre_x_list = x_list[step_size:-step_size]
    start_x_list  = x_list[step_size*2:]
    end_x_list    = x_list[:-step_size*2]
    
    centre_y_list = y_list[step_size:-step_size]    
    start_y_list    = y_list[step_size*2:]    
    end_y_list  = y_list[:-step_size*2]
    
    for i,_ in enumerate(centre_x_list):
        vec_start.append([centre_x_list[i] - start_x_list[i], centre_y_list[i] - start_y_list[i]])
        vec_end.append([end_x_list[i] - centre_x_list[i], end_y_list[i] - centre_y_list[i]])
    
    dir2 = np.array(vec_start)
    dir1 = np.array(vec_end)
    
    angle_calc = np.arccos((dir1*dir2).sum(axis=1)/(
        np.sqrt((dir1**2).sum(axis=1)*(dir2**2).sum(axis=1))))
    
    
    return 180 - np.degrees(angle_calc)

def order_points(points):
    """
    https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
    """
    
    clf = NearestNeighbors(2).fit(points) #calc nearest neighbour
    G = clf.kneighbors_graph() #create sparse matrix
    T = nx.from_scipy_sparse_matrix(G) #construct graph from sparse matrix
    # order paths
    paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]
    mindist = np.inf
    minidx = 0
    for i in range(len(points)):
        p = paths[i]           # order of nodes
        ordered = points[p]    # ordered nodes
        # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
        if cost < mindist:
            mindist = cost
            minidx = i
    
    return paths[minidx]

def line_angle(points, max_calc_pts):
    
    """
    #calculate the mean angle from the first point in a line to the subsequent points (up to 5)
    
    """
    
    #find halfway index and limit to 5 pixels
    half_idx = len(points)//2
    if half_idx > max_calc_pts:
        half_idx = max_calc_pts
    
    #calculate i and j unit vectors between other points up to half_idx and the end point    
    i_vec_list = []
    j_vec_list = []
    end_point = points[0] 
    for i in range(1, half_idx + 1):
        i_dist  = end_point[0] - points[i,0]
        j_dist  = end_point[1] - points[i,1]
        ij_dist = math.sqrt(i_dist**2 + j_dist**2)
        i_vec_list.append(i_dist/ij_dist)
        j_vec_list.append(j_dist/ij_dist)

    #calc mean unit vector
    i_vec_mean = np.mean(i_vec_list)
    j_vec_mean = np.mean(j_vec_list)
    vec_angle  = math.degrees(math.atan2(i_vec_mean, j_vec_mean)) #x,y order
    
    #return angle of unit vector
    return vec_angle
    
def points_angle(points_1, points_2):
    
    """
    #calculate the angle between two points
    
    """
                 
    i_dist = points_2[0] - points_1[0]
    j_dist = points_2[1] - points_1[1]
    angle  = math.degrees(math.atan2(i_dist, j_dist)) #y,x order
    
    #return angle of unit vector
    return angle

def cosine_test(a, b):
    """
    calculates the cosine similarity using the unit vector of provided angles a and b (in deg)
    
    Description from wiki:  It is thus a judgment of orientation and not magnitude: two vectors
    with the same orientation have a cosine similarity of 1, two vectors oriented at 90° relative
    to each other have a similarity of 0, and two vectors diametrically opposed have a similarity 
    of -1, independent of their magnitude. 
    """
    try:
        a_vec = np.array([math.sin(math.radians(a)), math.cos(math.radians(a))])
        b_vec = np.array([math.sin(math.radians(b)), math.cos(math.radians(b))])

        dot = np.dot(a_vec, b_vec)
        norma = np.linalg.norm(a_vec)
        normb = np.linalg.norm(b_vec)
        cos_lib = dot / (norma * normb)
        if np.isnan(cos_lib):
            cos_lib = 1
    except Exception as e:
        print(e)
    return cos_lib


def _edge_filter_worker(k_pos_list, point_list, refl_img, nvec):
    """
    Applies the filter edge technique from Yuan et al. 2018 for finding line segments which are in reflectivity areas and not true ridges
    """
    
    filter_vec = np.zeros_like(k_pos_list)
    
    for point in point_list:
        #origin point reflectivity value
        origin_value = refl_img[point[0], point[1]]
        
        for a, k_pos in enumerate(k_pos_list):
            #filter point reflectivity value
            try:
                filter_value  = refl_img[int(point[0]+nvec[0]*k_pos), int(point[1]+nvec[1]*k_pos)]
            except:
                #out of image
                filter_value = origin_value
            #equality test
            if origin_value > filter_value:
                continue
            elif origin_value == filter_value:
                filter_vec[a] += 1
            else:  #it's not a ridge!
                filter_vec[a] += 2
    #normalise filter
    filter0 = np.mean(filter_vec)/np.shape(point_list)[0]            
    return filter0

def _edge_filter_worker2(k_pos_list, point_list, refl_img, nvec):
    """
    Applies the filter edge technique from Yuan et al. 2018 for finding line segments which are in reflectivity areas and not true ridges
    """
    acc_origin = 0
    filter_vec = np.zeros_like(k_pos_list)
    
    for point in point_list:
        #origin point reflectivity value
        origin_value = refl_img[point[0], point[1]]
        acc_origin += origin_value
        
        for a, k_pos in enumerate(k_pos_list):
            #filter point reflectivity value
            try:
                filter_value = refl_img[int(point[0]+nvec[0]*k_pos), int(point[1]+nvec[1]*k_pos)]
            except:
                #out of image
                filter_value = origin_value
            
            filter_vec[a] += filter_value

    #normalise filter
    filter_vec0 = filter_vec/np.shape(point_list)[0]   
    filter0 = acc_origin/np.shape(point_list)[0] - np.mean(filter_vec0)
         
    return filter0

def edge_filter_wrapper(refl_img, nvec, point_list):
    """
    wrapper for edge filter
    """
    #filter value for positive side of line
    ak_list = np.array([2, 3, 4, 5, 6])
    a0 = _edge_filter_worker2(ak_list, point_list, refl_img, nvec)
    #filter value for negative size of line
    bk_list = np.array([-2, -3, -4, -5, -6])
    b0 = _edge_filter_worker2(bk_list, point_list, refl_img, nvec)
    #return filter values
    return a0, b0

def coor_intersect_test(arrA, arrB):
    return not set(map(tuple, arrA)).isdisjoint(map(tuple, arrB))