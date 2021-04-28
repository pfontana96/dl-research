import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import cv2 
from copy import deepcopy

import numpy as np

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
    
    for key, value in config.items():
        # Convert list to np arrays
        if type(value)==list:
            config[key] = np.array(value)

    return config

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

    def rescale_center_rel_grids(self, obj):
        '''
        Rescale center relative to number of gridcells

        Arguments:
        ---------
        obj:     [dict] dict containing xmin, xmax, ymin, ymax
        config : [dict] dict containing IMAGE_W, GRID_W, IMAGE_H and GRID_H
        
        Return:
        ------
        center: 
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



class BBox(object):
    def __init__(self, xmin, xmax, ymin, ymax, confidence=None, classes=None):
        assert((xmin < xmax) and (ymin < ymax)), "BBox dimensions do not make sense"
        self.xmin, self.xmax = (xmin, xmax)
        self.ymin, self.ymax = (ymin, ymax)

        self.confidence = confidence

        # classes probabilities [C1, C2, .., Cn]
        self.set_class(classes)

    def set_class(self, classes):
        self.classes = classes
        self.label = np.argmax(self.classes)

    def get_label(self):
        return self.label

    def get_score(self):
        return self.classes[self.label]

    def __str__(self):
        return "BBox:\n\txmin: {}   xmax:{}\n\tymin: {}   ymax:{}".format(self.xmin,
                                                                          self.xmax,
                                                                          self.ymin,
                                                                          self.ymax)

class BestAnchorBoxFinder(object):
    """
    Finds the best anchor box for a particular object. This is done by finding the anchor box 
    with the highest IOU(Intersection over Union) with the bounding box of the object.
    """
    def __init__(self, anchors):
        """
        Argument:
        --------
        anchors: [list] list containing dimensions for each bbox. Example:
            anchors = [[1, 3], [2, 5], [3, 4]] --> 3 bboxes with width and height:  (1, 3)
                                                                                    (2, 5)
                                                                                    (3, 4)
        """
        self.anchors = [BBox(0, width, 0, height) for width, height in anchors]

    def _interval_overlap(self, interval_a, interval_b):

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

    def _IOU(self, bbox1, bbox2):
        """
        Calculates Intersection over Union of 2 Bounding boxes
        """
        intersect_w = self._interval_overlap([bbox1.xmin, bbox1.xmax], [bbox2.xmin, bbox2.xmax])
        intersect_h = self._interval_overlap([bbox1.ymin, bbox1.ymax], [bbox2.ymin, bbox2.ymax])

        intersection = intersect_w*intersect_h

        w1, h1 = (bbox1.xmax - bbox1.xmin, bbox1.ymax - bbox1.ymin)
        w2, h2 = (bbox2.xmax - bbox2.xmin, bbox2.ymax - bbox2.ymin)

        union = w1*h1 + w2*h2 - intersection # -intersection because otherwise we're counting it twice
        # print('Int: {} and Union: {}'.format(intersection, union))

        if union < 1e-10: # Avoid division by zero
            return 0

        return float(intersection)/union

    def find(self, center_w, center_h):
        """
        Find the anchor box that best predicts this bbox
        """
        target = BBox(0, center_w, 0, center_h)
        ious = np.frompyfunc(self._IOU, 2, 1)(self.anchors, target)

        index = np.argmax(ious)

        return index, ious[index]
