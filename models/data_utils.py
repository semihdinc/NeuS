import os
import boto3
import json
import imageio.v2 as imageio
import numpy as np
from io import BytesIO
from scipy.spatial.transform import Rotation as SR
import cv2

def getROICornerPixels(c2w,fl,cx,cy,w,Pw):
    """Calculates the projected points of ROI corner points (Pw) in image space defined by intrinsics and c2w

    Args:
        c2w (_type_): camera to world matrix for image
        fl (_type_): focal length of the camera in pixels
        cx (_type_): principal point x
        cy (_type_): principal point y
        w (_type_): width of the image
        Pw (_type_): 3D world points to project into 2D space 

    Returns:
        pts: 2D pixel coordinates of Pw in specific order to be used in opencv masking
    """
    #intrinsic camera parameters of the image
    K = np.array([ [ fl,    0.000,   cx,      0.000], 
                    [ 0.000, fl,      cy,      0.000], 
                    [ 0.000, 0.000,   1.000,   0.000], 
                    [ 0.000, 0.000,   0.000,   1.000]])
                    
    Pe = np.linalg.inv(c2w) @ Pw #transform tile corners to camera space

    #image projection
    p = K @ Pe
    p[0] = w - p[0]/p[2]
    p[1] = p[1]/p[2]
    pts = p[0:2,[0,1,3,2]].astype(int)
    pts = pts.T.reshape(1,4,2)
    return pts

def maskImage(image, pts):
    """Masks image by keeping only pixels in a polygon defined by pts. All other pixels converted into transparent.
    """
    mask = np.zeros(image.shape, dtype=np.uint8)
    ignore_mask_color = (255,255,255)
    cv2.fillPoly(mask, pts, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    masked_image = np.stack((masked_image[:,:,0],masked_image[:,:,1],masked_image[:,:,2],mask[:,:,0]),axis=2)

    alive_fraction = np.count_nonzero(mask[...,0]) / mask[...,0].size #find ratio of non-empty pixels
    return masked_image, alive_fraction


def sphericalPoses(numberOfFrames, radius = 2000):
   """
   We first move the camera to [0,0,tz] in the world coordinate space. 
   Then we rotate the camera pos 45 degrees wrt X axis.
   Finally we rotate the camera wrt Z axis numberOfFrames times.
   Note: Camera space and world space (ENU) is actually aligned 
      X_c == X_w or E (east)
      Y_c == Y_w or N (north)
      Z_c == Z_w or U (up)
      Camera is positioned at [0,0,tz] it is actually looking down to -Z direction 
   """
   transMat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,radius],[0,0,0,1]]).astype(float) #move camera to 0,0,1500
   
   #rotate camera 45 degrees wrt X axis
   rotMatX = np.identity(4)
   rotMatX[0:3,0:3] = SR.from_euler('X',np.pi/4).as_matrix()
   
   #first translate then rotate
   transMat = rotMatX @ transMat
   
   poses = []
   for angle in np.linspace(0,2*np.pi,numberOfFrames+1)[:-1]:

      rotMatZ = np.identity(4)
      rotMatZ[0:3,0:3] = SR.from_euler('Z',angle).as_matrix()

      myPose = rotMatZ @ transMat
      # myPose[1,3] += 130 #We needed this for non-masked shuttle imageset to move rotation center
      poses.append(myPose)

   poses = np.stack(poses, axis=0)
   return poses


def generateTestPoses(root_fp: str, scene: str, numberOfFrames: int, ds: int):
   """This function generates test poses for rendering from a trained model

   Args:
       root_fp (str): Root folder of all scenes
       scene (str): The name of the scene
       numberOfFrames (int): Desired number of frames in test poses
       ds (int): Downscale factor from original image size

   Returns:
       It returns intrinsics and other scene related data
   """

   #open the transforms.json for reading intrinsics of first camera
   with open(os.path.join(os.path.join(root_fp, scene), 'transforms.json'), 'r') as fp:
      meta = json.load(fp)

   frame = meta["frames"][15]
   num_train_img = len(meta["frames"]) #number of images used in training
   
   #image intrinsics
   focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
   w, h = frame['w'], frame['h']
   
   #downscale intrinsics
   w, h = int(w/ds), int(h/ds)
   cx, cy = cx/ds, cy/ds
   focal = focal/ds
   
   #intrinsics, bounding box, c2w, and ROI enu_center (in lat/lon)
   K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
   aabb = list(np.concatenate(meta['aabb']).flat)
   camtoworlds = sphericalPoses(numberOfFrames)
   enu_center = meta["enu_center"]

   return camtoworlds, K, int(w), int(h), aabb, num_train_img, enu_center


def read_reproj_jsons(transforms_path: str, reproj_path: str, downscale_factor: int=4):
    '''
    Inputs:
        transforms_path: path to transforms.json in instant-ngp form with aabb added
        reproj_path: path to reproj_err.json organized as:
            3D_point_name
            |-> image_name
                |-> distance: from point to image (m)
                |-> obs_error: x, y reprojection error (pixels)
                |-> obs_pixel: x, y reprojected pixel location on image
        downscale: Scale factor of the images relative to the pointcloud
        
    Outputs:
        aabb: flatten reconstruction boundaries, [xmin, ymin, zmin, xmax, ymax, zmax]
        K: Instrinsic matrix [#image, 3, 3] 
        c2w: Rotation-translation matrix [#image, 3, 3]
        rgba: RGBA images [#image, w, h, 4]
        points: 3D points [point_id, image_id, u, v, depth [px], avg_error[m]] 
    '''
    IM_NAME_DICT = {} # image name as key, image_id as values
    #PT3_ID_DICT = {} # point names as keys, point_id as values
    #IM_PT3_DICT = {} # image_id as keys, point_ids as values

    print('Loading Images')
    with open(transforms_path, "r") as fid:
        transforms = json.load(fid)

    basepath = os.path.dirname(transforms_path)
    aabb = list(np.concatenate(transforms['aabb']).flat)

    K, c2w, rgba  = [], [], []
    for i, meta in enumerate(transforms['frames']):
        K.append(np.array([[meta['fl_x'], 0, meta['cx']],
                      [0, meta['fl_x'], meta['cy']],
                      [0, 0, 1]]))
        c2w.append(np.array(meta['transform_matrix']))                  
        rgba.append(imageio.imread(os.path.join(basepath, meta['file_path'])))

        IM_NAME_DICT[os.path.basename(meta['file_path'])[:-4]] = i # image name as key, image_id as values
        #IM_PT3_DICT[i] = [] # image_id as keys, point_ids as values
    
    K = np.stack(K)
    c2w = np.stack(c2w)
    rgba = np.stack(rgba)
    
    print('Loading 3D points')
    with open(reproj_path, "r") as fid:
        reproj = json.load(fid)

    xyds, errs, im_ids, pt3_ids, points, gsds = [], [], [], [], [], []
    for i, id in enumerate(reproj):
        xyd, err, im_id, pt3_id, point, gsd = [], [], [], [], [], []
        for im_name in reproj[id]:
            _im_id = IM_NAME_DICT[im_name]
            #im_id.append(_im_id)
            #xyd.append(reproj[id][im_name]['obs_pixel'] + [reproj[id][im_name]['distance']])
            err.append(reproj[id][im_name]['obs_error'])
            gsd.append(reproj[id][im_name]['gsd'])
            #pt3_id.append(i)
            point.append([i] + [_im_id] + reproj[id][im_name]['obs_pixel'] + [reproj[id][im_name]['distance']])

            #IM_PT3_DICT[_im_id] += [i] # image_id as keys, point_ids as values

        #xyds += xyd
        #im_ids += im_id
        #pt3_ids += pt3_id
        err = np.array(err) * np.array(gsd)[:, np.newaxis]
        #rmse = np.sqrt(np.mean(err**2))
        mae = np.mean(abs(err))
        errs += [mae] * len(err)
        points += point

        #PT3_ID_DICT[id] = i # point names as keys, point_id as values

    points = np.stack(points)
    points[:, 2:4] = points[:, 2:4]//downscale_factor
    errs = np.stack(errs)
    points = np.hstack((points, errs[:, None]))
    
    return aabb, K, c2w, rgba, points 


def load_renderings(root_fp: str, subject_id: str):
    data_dir = os.path.join(root_fp, subject_id)
    
    with open(os.path.join(data_dir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    images = []
    camtoworlds = []
    intrinsics = []
    depths = np.zeros([4,0], dtype=float)
    
    scene_scale_factor = meta["aabb"][1][0] #we use xmax in aabb to downscale scene to unit cube [-1,1] 

    aabb = meta['aabb']
    Pw = np.array([[aabb[0][0],aabb[1][0],aabb[0][0],aabb[1][0]],[aabb[1][1],aabb[1][1],aabb[0][1],aabb[0][1]],[0,0,0,0],[1,1,1,1]])

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame['file_path'])
        rgba = imageio.imread(fname)
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w[0:3,3] /= scene_scale_factor # scale to unit cube
        camtoworlds.append(c2w)
        
        
        #mask the image
        c2w = np.array(frame["transform_matrix"])
        pts = getROICornerPixels(c2w,frame['fl_x'],frame['cx'],frame['cy'],frame['w'],Pw)
        rgba, _ = maskImage(rgba, pts)

        images.append(rgba)
        #per image intrinsics
        focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        intrinsics.append(K)

    images = np.stack(images, axis=0) #assume all images have same size
    camtoworlds = np.stack(camtoworlds, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    aabb = list(np.concatenate(meta['aabb']).flat)
    num_images = images.shape[0]

    return images, depths, camtoworlds, intrinsics, aabb, num_images
    

def load_renderings_s3(bucket_name: str, scene_folder: str):
    
    client = boto3.client('s3')

    transforms = os.path.join(scene_folder,'transforms.json')
    file_content = client.get_object(Bucket=bucket_name, Key=transforms)['Body'].read().decode('utf-8')
    meta = json.loads(file_content)

    images = []
    camtoworlds = []
    intrinsics = []
    depths = np.empty([4,0], dtype=float)
    
    
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(scene_folder, frame['file_path'][2:])
        
        #retrieve image from s3 bucket
        image_byte_string = client.get_object(Bucket=bucket_name, Key=fname)['Body'].read()
        
        rgba = imageio.imread(image_byte_string)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)
        
        #per image intrinsics
        focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        intrinsics.append(K)

        #depth images
        depth_path = os.path.join(scene_folder,'depth',os.path.basename(fname).split('.')[0]+'.npy')
        file_content = client.get_object(Bucket=bucket_name, Key=depth_path)['Body'].read()
        pts = np.load(BytesIO(file_content))
        
        idx = np.ones([1, pts.shape[1]]) * i
        pts = np.vstack((idx,pts))

        depths = np.append(depths,pts,axis=1)

    images = np.stack(images, axis=0) #assume all images have same size
    camtoworlds = np.stack(camtoworlds, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    aabb = list(np.concatenate(meta['aabb']).flat)

    return images, depths, camtoworlds, intrinsics, aabb