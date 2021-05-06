import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import cv2 
from copy import deepcopy
from itertools import chain

import numpy as np
import tensorflow as tf

def extract_bboxes(y_batch, anchors, image_shape, max=False):
    """
    Loads a batch of prediction and returns bboxes.

    Arguments:
    ---------
    y_batch:     [tf.Tensor] Tensor of shape (batch_size, grid_w, grid_h, nb_anchors, 5+nb_classes)
    anchors:     [np.array] Array of shape (nb_anchors, 2) specifying anchors used for y_batch (width and height)
    image_shape: [np.array] [tuple] [list] Input Image shape (width and height)
    max:         [bool] If max is set to True, only the class with max score will be returned

    Return:
    ------
    bboxes:      [List] List of shape (batch_size) containing an np.array of shape [nb_bbox_in_img, nb_classes]
                  where each bbox is defined by (x, y, w, h, [scores]) where [scores] = [Pc1, ..., Pcn]. 
                  If max is True, [scores] will consist of [max(Pc), argmax(Pc)] 
    """
    _, nb_grids_y, nb_grids_x, nb_anchors, nb_classes = y_batch.shape
    nb_classes -= 5

    grid_width = image_shape[0]/nb_grids_x
    grid_height = image_shape[1]/nb_grids_y

    # Indices to use after reshape
    grid_y_offsets = np.array(list(chain(*([x]*nb_grids_x*nb_anchors for x in range(nb_grids_y)))))
    grid_x_offsets = np.array(list(chain(*([x]*nb_anchors for x in range(nb_grids_x))))*nb_grids_y)
    anchor_index = np.array([range(nb_anchors)]*nb_grids_y*nb_grids_x).reshape(-1)

    anchor_w_offsets = anchors[anchor_index, 0]
    anchor_h_offsets = anchors[anchor_index, 1]

    bboxes = []
    scores = []
    for instance in y_batch:
        scores_tf = tf.expand_dims(instance[:,:,:,4], axis=3)*instance[:,:,:,5:]
        scores_tf = tf.reshape(scores_tf, (-1, nb_classes))
        # We need to rescale bboxes from gridcell's position and anchor dimension back to rescaled
        # image of config['image_shape']

        bboxes_tf = tf.reshape(instance[:,:,:,0:4], (-1, 4))
        scaled_back_x = tf.reshape((bboxes_tf[:,0]+grid_x_offsets)*grid_width, (-1,1))
        scaled_back_y = tf.reshape((bboxes_tf[:,1]+grid_y_offsets)*grid_height, (-1,1))
        scaled_back_w = tf.reshape(bboxes_tf[:,2]*anchor_w_offsets, (-1,1))
        scaled_back_h = tf.reshape(bboxes_tf[:,3]*anchor_h_offsets, (-1,1))

        bboxes_tf = tf.concat([scaled_back_x, 
                               scaled_back_y,
                               scaled_back_w,
                               scaled_back_h], axis=1)
        bboxes.append(bboxes_tf.numpy())
        scores.append(scores_tf.numpy())

    return bboxes, scores
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                   COMMON UTILS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  

def load_config(filename):
    """
    Loads config file into a dict of numpy objects

    Arguments:
    ---------
    filename: [pathlib.Path] path to .yaml file containing the configurations of the model

    Return:
    ------
    config: [dict] dict containing model's configuration
    """
    assert(isinstance(filename, Path)), "Argument is not a pathlib.Path"
    assert(filename.suffix==".yaml"), "File extension is not .yaml"

    with open(filename, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    
    if not set(["image_shape", "batch_size", "anchors", "labels"]).issubset(config.keys()):
        raise ValueError("Critical element missing in config file.\n'anchors'       \
                          'batch_size' 'image_shape' and 'labels' must be defined")

    for key, value in config.items():
        # Convert list to np arrays
        if type(value)==list:
            config[key] = np.array(value)
    
    # YOLO v2 reduction factor is 32
    if np.sum(config['image_shape']%32, axis=0):
        raise ValueError("Invalid image shape. Only multiples of 32 are accepted as YOLO's \
                          reduction factor is 32).")
        
    config['grid'] = np.array((config['image_shape']/32).astype(int))

    if config['anchors'].shape[1] != 2:
        raise ValueError("Invalid anchors shape. Shape should be (nb_anchors, 2).")

    return config

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                   IMAGES OPERATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def parse_annotations(ann_dir, img_dir, labels=[]):
    """
    Parses info from labeled data

    Arguments:
    ---------
    ann_dir: [pathlib.Path] path to the dir containing the annotations of the whole dataset (.xml)
    img_dir: [pathlib.Path] path to the dir containing all the images files (.jpg)

    Return:
    ------
    all_imgs: [list] List of the objects [dict] in the dataset containing:
        - filename: [str] path to .jpeg in img_dir 
        - height:   [int] heigh of image in pixels
        - width:    [int] width of image in pixels
        - object:   [list] list containing all labeled objects [dict] present in the image 
                            with containing the following info for each object:
            - name: class name
            - xmax: bbox max x coordinate (px)
            - xmin: bbox min x coordinate (px)
            - ymax: bbox max y coordinate (px)
            - ymin: bbox min y coordinate (px)
    seen_labels: [dict] dict of the classes with the nb of times labeled
    """

    assert(isinstance(ann_dir, Path) and isinstance(img_dir, Path)), "One argument is not of class pathlib.Path"
            
    assert(ann_dir.is_dir() and img_dir.is_dir()), "Not found directory!.." 

    all_imgs = []
    seen_labels = {}

    for ann in sorted(ann_dir.iterdir()):
        if ann.suffix != ".xml":
            continue
        img = {'object':[]}

        # tree = ET.parse(ann_dir + ann)
        tree = ET.parse(ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                path_to_image = img_dir.joinpath(elem.text)
                img['filename'] = str(path_to_image)

                ## make sure that the image exists:
                if not path_to_image.exists():
                    assert False, "file does not exist!\n{}".format(path_to_image)

            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']]  = 1
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels    

def normalize(img):
    return img/255

class ImageReader(object):
    """
    Class to process images and return a resized image as well as resize annotations for
    objects in it
    """
    def __init__(self, img_width, img_height, norm=None, **kwargs):
        self.img_w = img_width
        self.img_h = img_height
        self.norm = norm

        self.grid = kwargs.get('grid', (7, 7)) # Grid shape 
    

    def encodeCore(self, img, reorder_rgb=True):
        img = cv2.resize(img, (self.img_w, self.img_h))
        if reorder_rgb:
            img = img[:,:,::-1]
        if self.norm is not None:
            img = self.norm(img)
        return img

    def fit_data(self, train_instance):
        '''
        Read in and resize the image, annotations are resized accordingly.
        
        Argument:
        --------        
        train_instance : dictionary containing filename, height, width and object
        
        {'filename': 'ObjectDetectionRCNN/VOCdevkit/VOC2012/JPEGImages/2008_000054.jpg',
         'height':   333,
         'width':    500,
         'object': [{'name': 'bird',
                     'xmax': 318,
                     'xmin': 284,
                     'ymax': 184,
                     'ymin': 100},
                    {'name': 'bird', 
                     'xmax': 198, 
                     'xmin': 112, 
                     'ymax': 209, 
                     'ymin': 146}]
        }
        
        '''

        if not isinstance(train_instance, dict):
            train_instance = {'filename' : train_instance}

        img_name = train_instance['filename']
        img = cv2.imread(img_name)
        h, w, c = img.shape

        if img is None:
            print('Cannot find {}'.format(img_name))

        img = self.encodeCore(img, reorder_rgb=True)

        if 'object' in train_instance.keys():
            all_objs = deepcopy(train_instance['object'])
            
            for obj in all_objs:
                # Rule of three applied to the coordinate of detected objects in width
                # as well as in height
                for attr in ['xmin', 'xmax']:
                        obj[attr] = int(obj[attr] * float(self.img_w) / w)
                        obj[attr] = max(min(obj[attr], self.img_w), 0)

                for attr in ['ymin', 'ymax']:
                    obj[attr] = int(obj[attr] * float(self.img_h) / h)
                    obj[attr] = max(min(obj[attr], self.img_h), 0)
        else:
            return img
        return (img, all_objs)

    def abs2grid(self, obj):
        '''
        Rescale center relative to number of gridcells

        Arguments:
        ---------
        obj:     [dict] dict containing xmin, xmax, ymin, ymax
        
        Return:
        ------
        center: [tup] Tupple containing (x, y, w, h) in gridcells coordinate
                 i.e: (12.3, 6.4, 2.3, 4) 
                      Means an object which centers lays 30% right to 12th cell origin in x,
                      40% up (or down depending reference to origin) of 6th cell with a width
                      of 'w' times grid's width and a height of 'h' times grid's height
        '''

        center_x = (obj['xmin'] + obj['xmax'])/2
        center_x = center_x / (float(self.img_w) / self.grid[0])
        
        center_y = (obj['ymin'] + obj['ymax'])/2
        center_y = center_y / (float(self.img_h) / self.grid[1])

        # unit: grid cell
        center_w = (obj['xmax'] - obj['xmin']) / (float(self.img_w) / self.grid[0]) 
        # unit: grid cell
        center_h = (obj['ymax'] - obj['ymin']) / (float(self.img_h) / self.grid[1]) 

        return (center_x, center_y, center_w, center_h)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                   IOU AND NON MAX CALCULATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def non_max_suppression_v2(bboxes, scores, class_ids, iou_threshold, score_threshold):
    """
    Calculates Non Max suppression in a numpy way

    Arguments:
    ---------
    bboxes:          [np.array] Numpy array containing bboxes (x, y, w, h) in absolute 
                                coordinates
    scores:          [np.array] Array of scores corresponding to each bbox
    iou_threshold:   [float] IOU threshold (each bbox with IOU >= IOU threshold will be discarted)
    score_threshold: [float] Score threshold (each bbox with score < score_threshold will be 
                             discarted)

    Return:
    ------
    result_bboxes:   [np.array] Array of resulting bboxes 
    result_scores:
    result_classes:
    """
    result_bboxes = []
    result_scores = []
    result_classes = []

    # Drop non confident bboxes
    index = np.where(scores >= score_threshold)

    bboxes = bboxes[index]
    scores = scores[index]
    class_ids = class_ids[index]

    for clss in np.unique(class_ids):
        index = np.where(class_ids == clss)
        c_scores = scores[index]
        c_bboxes = bboxes[index]
        while len(c_scores):
            curr_best_id = np.argmax(c_scores)
            curr_best_bbox = c_bboxes[curr_best_id].reshape(-1,4)
            
            result_bboxes.append(curr_best_bbox)
            result_scores.append(c_scores[curr_best_id])
            result_classes.append(clss)

            c_bboxes = np.delete(c_bboxes, curr_best_id, axis=0)
            c_scores = np.delete(c_scores, curr_best_id)

            ious = calculate_IOU_nag(curr_best_bbox, c_bboxes).reshape(-1)
            
            index_to_eliminate = np.where(ious >= iou_threshold)
            # print('{}Logging\nIOUs: {}'.format('*'*15, ious))
            c_bboxes = np.delete(c_bboxes, index_to_eliminate, axis=0)
            c_scores = np.delete(c_scores, index_to_eliminate)

    result_bboxes = np.array(result_bboxes).reshape(-1,4)
    result_scores = np.array(result_scores).reshape(-1)
    result_classes = np.array(result_classes).reshape(-1)

    return result_bboxes, result_scores, result_classes

def non_max_suppression(y_batch, iou_threshold, prob_threshold):
    count = 0
    new_y_batch = np.zeros(y_batch.shape)
    _, grid_y, grid_x, nb_anchors, nb_classes = y_batch.shape 
    nb_classes -= 5 # (x, y, w, h)
    nb_bboxes = grid_y*grid_x*nb_anchors

    iou_vfunc = np.frompyfunc(lambda x1, x2, x3, x4: calculate_IOU(
                            np.array([x1, x2]), np.array([x3, x4])), 
                            4, 1)
    for instance in y_batch:
        # For each instance we'll need to calculate the scores for each bounding box predicted
        # As we're predicting Pc and C1, C2, ..., Cn as the scores will be calculated as follow
        # score = Pc*Ci = P(Classi|Object)*P(Object)
        scores = tf.multiply(tf.reshape(instance[:,:,:,5:], (-1, nb_classes)),
                            tf.reshape(instance[:,:,:,4], (-1,1))).numpy()
        
        scores = np.where(scores == np.max(scores, axis=1, keepdims=True), scores, 0)

        bboxes = tf.reshape(instance[:,:,:,2:4], (-1, 2)).numpy()

        for class_id in range(nb_classes):
            # mask = self.instance_non_max_suppression(bboxes, scores[:, class_id])
            ids = np.array(range(nb_bboxes))
            mask = np.zeros((nb_bboxes), dtype=bool)
            c_scores = scores[:, class_id]
            c_bboxes = bboxes

            # Eliminate low confidence predictions
            index = np.where(c_scores >= prob_threshold)[0]
            ids = ids[index]
            c_bboxes = c_bboxes[index]
            c_scores = c_scores[index]

            while len(ids):
                best_index = np.argmax(c_scores)
                best_id = ids[best_index]
                best_bbox = c_bboxes[best_index]

                # Mark current bbox in mask
                mask[best_id] = True

                ids = np.delete(ids, best_index)
                c_bboxes = np.delete(c_bboxes, best_index, axis=0)

                if not len(ids):
                    break
                    
                ious = iou_vfunc(best_bbox[0], best_bbox[1], c_bboxes[:,0], c_bboxes[:,1])
                remove_index = np.where(ious >= iou_threshold)[0]

                ids = np.delete(ids, remove_index)
                c_bboxes = np.delete(c_bboxes, remove_index, axis=0)
                c_scores = np.delete(c_scores, np.concatenate((remove_index, [best_index])))

            mask = mask.reshape(grid_y, grid_x, nb_anchors)

            new_y_batch[count, mask] = instance[mask]

        count += 1

    return tf.constant(new_y_batch)    

def tf_non_max_suppression(y_batch, iou_threshold, prob_threshold):
    count = 0
    batch_size, grid_y, grid_x, nb_anchors, nb_classes = y_batch.shape
    nb_classes -= 5 # (x, y, w, h, pr)
    new_y_batch = np.zeros(y_batch.shape, dtype=float)
    # nms_vfunc = np.frompyfunc(lambda scrs: self.instance_non_max_suppression(bboxes, scrs), 
    #                           self.grid[0]*self.grid[1]*self.nb_anchors, 1)
    nb_bboxes = grid_y*grid_x*nb_anchors
    dummy_center = tf.zeros((nb_bboxes, 2), dtype=tf.float32)
    for instance in y_batch:
        # For each instance we'll need to calculate the scores for each bounding box predicted
        # As we're predicting Pc and C1, C2, ..., Cn as the scores will be calculated as follow
        # score = Pc*Ci = P(Classi|Object)*P(Object)
        scores = tf.multiply(tf.reshape(instance[:,:,:,5:], (-1, nb_classes)),
                            tf.reshape(instance[:,:,:,4], (-1,1)))
        
        scores = tf.cast(tf.where(scores == tf.reduce_max(scores, axis=1, keepdims=True), scores, 0),
                        tf.float32)
    
        bboxes = tf.cast(tf.reshape(instance[:,:,:,2:4], (-1, 2)), tf.float32)
        bboxes = tf.concat([dummy_center, bboxes], axis=1)


        for class_id in range(nb_classes):
            mask = np.zeros(nb_bboxes, dtype=bool)
            index = tf.image.non_max_suppression(bboxes, scores[:, class_id], 30,
                    iou_threshold=iou_threshold, score_threshold=prob_threshold) # 30 max objects detected per image
            mask[index.numpy()] = True
            mask = mask.reshape(grid_x, grid_y, nb_anchors)

            new_y_batch[count, mask] = instance[mask]

        count += 1

    return tf.constant(new_y_batch)

def calculate_IOU_nag(bbox1, bbox2):
    """
    Non agnostic calculation of IOU (depends on the position xy)

    Arguments:
    ---------
    bbox1: [np.array] 2D Array containing (x, y, w, h) with shape (nb_bboxes, 4)
    bbox2: [np.array] 2D Array containing (x, y, w, h) with shape (nb_bboxes, 4)
    """
    epsilon = 1e-5 # Numerical hack
    x1, y1, w1, h1 = np.split(bbox1, 4, axis=1)
    x2, y2, w2, h2 = np.split(bbox2, 4, axis=1)

    x_max1 = x1 + w1/2
    x_min1 = x1 - w1/2
    x_max2 = x2 + w2/2
    x_min2 = x2 - w2/2

    y_max1 = y1 + h1/2
    y_min1 = y1 - h1/2
    y_max2 = y2 + h2/2
    y_min2 = y2 - h2/2

    intersect_x = np.maximum(np.minimum(x_max1, x_max2.T) - np.maximum(x_min1, x_min2.T), 0)
    intersect_y = np.maximum(np.minimum(y_max1, y_max2.T) - np.maximum(y_min1, y_min2.T), 0)
    intersect = intersect_x * intersect_y

    bboxes_areas_1 = (x_max1 - x_min1)*(y_max1 - y_min1)
    bboxes_areas_2 = (x_max2 - x_min2)*(y_max2 - y_min2)
    
    union = (bboxes_areas_1 + bboxes_areas_2.T) - intersect
    return np.divide(intersect, union, out=np.zeros_like(intersect), where=(union >= epsilon))

def _interval_overlap(interval_a, interval_b):
        """
        Helper function for IOU calculation
        """
        x1, x2 = interval_a
        x3, x4 = interval_b
        assert((x1 <= x2) and (x3 <= x4)), "Interval's 1st component larger than 2nd one!"

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

def calculate_IOU(bbox1, bbox2):
    """
    This function is agnostic of where the center is located! 
    
    Arguments:
    ---------
    bbox1: [np.array] or [list] Array containing (w,h) for bbox 1
    bbox2: [np.array] or [list] Array containing (w,h) for bbox 2

    Return:
    ------
    iou: [float] Resulting IOU
    """
    w1, h1 = bbox1
    w2, h2 = bbox2
    intersect_w = _interval_overlap([0, w1], [0, w2])
    intersect_h = _interval_overlap([0, h1], [0, w2])

    intersection = intersect_w*intersect_h

    union = w1*h1 + w2*h2 - intersection # -intersection because otherwise we're counting it twice

    if union < 1e-10: # Avoid division by zero
        return 0

    return float(intersection)/union

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                   LOAD WEIGHTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_bytes(weights, offset, size):
    offset += size
    return (weights[offset-size:offset], offset)

class WeightsReader(object):
    def __init__(self, weights_file):
        assert(isinstance(weights_file, Path)), "Weights file must be given as [pathlib.Path]"
        self.offset = 4
        self.all_weights = np.fromfile(weights_file, dtype='float32')

    def read_bytes(self, size):
        self.offset += size
        return self.all_weights[self.offset-size:(self.offset)]
        
    def reset(self):
        self.offset = 4
    
    def load_weights(self, model):
        """
        Loads saved weights to model 

        Arguments:
        ---------
        model: [tf.keras.Model] Model
        nb_grids: [int] # of gridcells
        """
        # TODO: Should implement some sort of assertion to verify the weights file and the
        #       model are compatible (same amount)

        layers_names = np.array([l.name.split('_')[0] for l in model.layers], dtype=str)
        nb_convs = len(layers_names[layers_names=='conv'])
        nb_grids = np.prod(model.grid)
        self.reset()

        for layer in model.layers:

            weights_list = layer.get_weights() 

            if 'conv' in layer.name:
                if len(weights_list) > 1: # Kernel and bias
                    bias = self.read_bytes(np.prod(weights_list[1].shape))
                    kernel = self.read_bytes(np.prod(weights_list[0].shape))
                    kernel = kernel.reshape(list(reversed(weights_list[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0]) # TODO: Investigate this transpose (why that order of axis, look how kernel are stored in weigths file)

                    layer.set_weights([kernel, bias])
                else: # just kernel
                    kernel = self.read_bytes(np.prod(weights_list[0].shape))
                    kernel = kernel.reshape(list(reversed(weights_list[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0]) # TODO: Investigate this transpose (why that order of axis, look how kernel are stored in weigths file)

                    layer.set_weights([kernel])

            if 'norm' in layer.name:

                size = np.prod(weights_list[0].shape)

                beta = self.read_bytes(size)
                gamma = self.read_bytes(size)
                mean = self.read_bytes(size)
                var = self.read_bytes(size)

                layer.set_weights([gamma, beta, mean, var])
        
        # Rescale last conv kernel to ouput image
        layer = model.get_layer('conv_{}'.format(nb_convs))

        weights_list = layer.get_weights()

        new_kernel = np.random.normal(size=weights_list[0].shape)/nb_grids
        new_bias = np.random.normal(size=weights_list[1].shape)/nb_grids
        layer.set_weights([new_kernel, new_bias])  
