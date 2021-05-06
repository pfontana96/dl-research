import numpy as np

import tensorflow as tf

from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, LeakyReLU, concatenate
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import get_value, set_value

from pathlib import Path

from .utils import ImageReader, WeightsReader, read_bytes, calculate_IOU, calculate_IOU_nag

def adjust_yolo_output(y_pred):
    """
    As stated in the original paper, YOLO predicts a bounding box as, relative to the gridcell's
    origin (cx, cy) and anchor box chosen (pw, ph):
        (tx, ty, tw, th, to)
    where:
        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        bw = pw*exp(tw)
        bh = ph*exp(th)
        Pr(obj)*IOU(b,obj) = sigmoid(to)
        Pr(Ci) = softmax([C1 C2 .. Cn])
    """
    batch_size, grid_y, grid_x, nb_anchors, output_dim = y_pred.shape
    y_pred = tf.reshape(y_pred, (-1, output_dim))
    pred_xy = tf.sigmoid(y_pred[:, 0:2])
    pred_wh = tf.exp(y_pred[:, 2:4])
    pred_p = tf.reshape(tf.sigmoid(y_pred[:, 4]), (-1,1))
    pred_classes = tf.nn.softmax(y_pred[:, 5:])

    output = tf.concat([pred_xy, pred_wh, pred_p, pred_classes], axis=1)
    
    return tf.reshape(output, (batch_size, grid_y, grid_x, nb_anchors, output_dim))

def yolo_loss(lambda_coord, lambda_noobj):
    # Hack to allow for additional parameters when defining a keras loss function
    def calculate_loss(y_true, y_pred):
        """
        Computes YOLO Loss
        
        Arguments:
        ---------
        y_pred : [tf.Tensor] Tensor of shape (batch_size, grid_h, grid_w, nb_anchors, 5+nb_classes)
                containing predicted values
        y_pred : [tf.Tensor] Tensor of shape (batch_size, grid_h, grid_w, nb_anchors, 5+nb_classes)
                containing ground truth values
        """
        # Convert network's output (see adjust_yolo_ouput for further info)
        y_pred = adjust_yolo_output(y_pred)

        # Get common values
        batch_size, grid_h, grid_w, nb_anchors, _ = y_true.shape
        nb_bboxes = batch_size*grid_h*grid_w*nb_anchors
        epsilon = 1e-6 # Epsilon to avoid division by 0

        # Get mask for indexing where there're objects detected on ground truth
        gt_obj_mask = tf.where(y_true[:,:,:,:,4] == 1, True, False).numpy()

        # ------------------------------------------------------------------------
        # Localizaton LOSS (i.e. differences in true bboxes and the predicted ones)

        # Calculate x,y loss
        pred_xy = y_pred[gt_obj_mask][:,0:2]
        true_xy = y_true[gt_obj_mask][:,0:2]

        loss_xy = tf.reduce_sum(tf.reduce_sum(tf.math.square(pred_xy - true_xy), 0), 0)
        loss_xy = loss_xy / (nb_bboxes + epsilon)

        # Calculate w,h loss
        pred_wh = y_pred[gt_obj_mask][:,2:4]
        true_wh = y_true[gt_obj_mask][:,2:4]

        # As defined in original paper's loss, we take de square roots
        pred_wh = tf.math.sqrt(pred_wh)
        true_wh = tf.math.sqrt(true_wh)

        loss_wh = tf.reduce_sum(tf.reduce_sum(tf.math.square(pred_wh - true_wh), 0), 0)
        loss_wh = loss_wh / (nb_bboxes + epsilon)

        # ------------------------------------------------------------------------
        # Confidence LOSS (i.e. differences between anchor confidence vs ground truth(1) )

        pred_c_obj = y_pred[gt_obj_mask][:,4]
        true_c_obj = y_true[gt_obj_mask][:,4]   #TODO: Shouldn't this be always 1 for ground truth?
        
        loss_c = tf.reduce_sum(tf.math.square(pred_c_obj - true_c_obj), 0)
        
        # We also account for images with no objects in them
        pred_c_noobj = y_pred[~gt_obj_mask][:,4]
        true_c_noobj = y_true[~gt_obj_mask][:,4]

        loss_c += lambda_noobj*tf.reduce_sum(tf.math.square(pred_c_obj - true_c_obj), 0)

        loss_c = loss_c/(nb_bboxes + epsilon)

        # ------------------------------------------------------------------------
        #           Classification LOSS (i.e. class probabilities discrepancies)

        # We use cross_entropy
        pred_class = y_pred[gt_obj_mask][:,5:]
        true_class = tf.argmax(y_true[gt_obj_mask][:,5:], -1) # Index for cross-entropy calculation
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_class, logits=pred_class)
        loss_class = tf.reduce_sum(loss_class) / (nb_bboxes + epsilon)

        # ------------------------------------------------------------------------
        #                               Total LOSS

        return lambda_coord*(loss_xy + loss_wh) + loss_c + loss_class
    return calculate_loss

class LearningRateScheduler(Callback):
    def __init__(self, schedule):
        """
        Arguments:
        ---------
        schedule: 
        """
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError("Optimizer must have a 'lr' attribute.")
        # Get current learning rate
        lr = float(get_value(self.model.optimizer.learning_rate))
        # Get scheduled learning rate
        scheduled_lr = self.get_scheduled_lr(epoch, lr)
        if lr != scheduled_lr:
            set_value(self.model.optimizer.learning_rate, scheduled_lr)
            print('Epoch {}: Learning rate changed from {} to {}'.format(epoch, lr, 
                                                                        scheduled_lr))
    def get_scheduled_lr(self, epoch, lr):
        """
        Helper function to get scheduled learning_rate
        """
        if epoch < self.schedule[0, 0] or epoch > self.schedule[-1, 0]:
            return lr
        for s_epoch, s_lr in self.schedule:
            if s_epoch == epoch:
                return s_lr
        return lr

# the function to implement the orgnization layer (thanks to https://github.com/allanzelener/YAD2K)
def space_to_depth_x2(x):
    import tensorflow as tf 
    return tf.nn.space_to_depth(x, block_size=2)

class YOLO_v2(Model):
    """
    YOLO V2 Model as defined in original paper: https://arxiv.org/abs/1612.08242
    """
    def __init__(self, config):
        super(YOLO_v2, self).__init__()
        self.convs = []
        self.maxpools = []
        self.norms = []
        self.acts = []

        self.prob_threshold = config.get('prob_t', 0.6) # Threshold for discarting predictions
        self.iou_threshold = config.get('iou_t', 0.5) # Threshold for discarting multiple predictions of same 
                                                      # object in non-max suppression

        self.nb_anchors = len(config['anchors'])
        self.nb_classes = len(config['labels'])

        self.img_shape = config['image_shape']
        self.grid = config['grid']

        # Specifications taken from original paper (based on Darknet-19)
        filters = [32, 64, 128, 64, 128, 256, 128, 256, 512, 256, 512, 256, 512, 
                   1024, 512, 1024, 512, 1024, 1024, 1024, 64, 1024, self.nb_anchors*(5 + self.nb_classes)]
        kernels = [(3,3), (3,3), (3,3), (1,1), (3,3), (3,3), (1,1), (3,3), (3,3), (1,1), (3,3),
                   (1,1), (3,3), (3,3), (1,1), (3,3), (1,1), (3,3), (3,3), (3,3), (1,1), (3,3),
                   (1,1)]
        strides = [(1, 1)]*len(filters)
        padding = ['same']*len(filters)

        biases = np.zeros(23, dtype=bool)
        biases[22] = True # Only last Conv2D layer uses bias!

        self.max_pooling = np.zeros(23, dtype=bool)
        self.max_pooling[[0, 1, 4, 7, 12]] = True # Layers (1,2,5,8,13) use maxpooling
        pool_size = (2, 2)

        self.skipped_connections = [12, 20] # Fwd input from layer 13 to layer 21

        l_id = 1
        for f, k, s, p, b, max_p in zip(filters, kernels, strides, padding, biases, self.max_pooling):
            self.convs.append(Conv2D(f, k, strides=s, padding=p, use_bias=b,
                                     name='conv_{}'.format(l_id)))
            if l_id != 23: #
                self.norms.append(BatchNormalization(name="norm_{}".format(l_id)))
                self.acts.append(LeakyReLU(alpha=0.1, name="leaky_relu_{}".format(l_id)))
            
            if max_p:
                self.maxpools.append(MaxPooling2D(pool_size=pool_size))

            l_id += 1
        
    def call(self, inputs, training=False):
        # x, true_boxes = inputs
        x = inputs
        mp_id = 0
        for l_id in range(len(self.convs)-1):
            if l_id != self.skipped_connections[1]:
                x = self.convs[l_id](x)
                x = self.norms[l_id](x)
                x = self.acts[l_id](x)

                if l_id == self.skipped_connections[0]:
                    skip_connection = x

                if self.max_pooling[l_id]:
                    x = self.maxpools[mp_id](x)
                    mp_id += 1
            else:
                # Layer 21
                skip_connection = self.convs[l_id](skip_connection)
                skip_connection = self.norms[l_id](skip_connection)
                skip_connection = self.acts[l_id](skip_connection)
                # skip_connection = self.lambda_1(skip_connection)
                skip_connection = Lambda(space_to_depth_x2)(skip_connection)
                x = concatenate([skip_connection, x])

        # 23rd Layer does not have an activation nor normalization
        x = self.convs[len(self.convs)-1](x)

        output = Reshape((self.grid[1], self.grid[0], self.nb_anchors, 5+self.nb_classes))(x)
        
        return output   

    def load_weights_from_file(self, filename):
        """
        Loads saved weights to model 

        Arguments:
        ---------
        model: [tf.keras.Model] Model
        nb_grids: [int] # of gridcells
        """
        assert(isinstance(filename, Path)), "Weights file must be given as [pathlib.Path]"

        # TODO: Should implement some sort of assertion to verify the weights file and the
        #       model are compatible (same amount)
        all_weights = np.fromfile(filename, dtype='float32')
        offset = 4

        nb_convs = len(self.convs)
        nb_grids = np.prod(self.grid)

        for layer in self.layers:

            weights_list = layer.get_weights() 

            if 'conv' in layer.name:
                if len(weights_list) > 1: # Kernel and bias
                    bias, offset = read_bytes(all_weights, offset, np.prod(weights_list[1].shape))
                    kernel, offset = read_bytes(all_weights, offset, np.prod(weights_list[0].shape))
                    kernel = kernel.reshape(list(reversed(weights_list[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0]) # TODO: Investigate this transpose (why that order of axis, look how kernel are stored in weigths file)

                    layer.set_weights([kernel, bias])
                else: # just kernel
                    kernel, offset = read_bytes(all_weights, offset, np.prod(weights_list[0].shape))
                    kernel = kernel.reshape(list(reversed(weights_list[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0]) # TODO: Investigate this transpose (why that order of axis, look how kernel are stored in weigths file)

                    layer.set_weights([kernel])

            if 'norm' in layer.name:

                size = np.prod(weights_list[0].shape)

                beta, offset = read_bytes(all_weights, offset, size)
                gamma, offset = read_bytes(all_weights, offset, size)
                mean, offset = read_bytes(all_weights, offset, size)
                var, offset = read_bytes(all_weights, offset, size)

                layer.set_weights([gamma, beta, mean, var])
        
        # Rescale last conv kernel to ouput image
        layer = self.get_layer('conv_{}'.format(nb_convs))

        weights_list = layer.get_weights()

        new_kernel = np.random.normal(size=weights_list[0].shape)/nb_grids
        new_bias = np.random.normal(size=weights_list[1].shape)/nb_grids
        layer.set_weights([new_kernel, new_bias])


class BatchGenerator(Sequence):
    def __init__(self, images, config, norm=None, shuffle=True):
        
        self.images = images

        self.batch_size = config['batch_size']
        self.anchors = config['anchors']
        self.nb_anchors = len(config['anchors'])
        
        self.img_w, self.img_h = config['image_shape']
        self.grid = config['grid']
        self.img_encoder = ImageReader(img_width=self.img_w, img_height=self.img_h, 
                                       norm=norm, grid=self.grid)

        self.labels = np.array(config['labels'])

        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.images)

    def __getitem__(self, idx):
        '''
        Arguments:
        ---------
        idx : [int] non-negative integer value e.g., 0
        
        Return:
        ------
        x_batch: [np.array] Array of shape  (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels).
            
            x_batch[iframe,:,:,:] contains a iframeth frame of size  (IMAGE_H,IMAGE_W).
            
        y_batch: [np.array] Array of shape  (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes). 
                            BOX = The number of anchor boxes.

            y_batch[iframe,igrid_h,igrid_w,ianchor,:4] contains (center_x,center_y,center_w,center_h) 
            of ianchorth anchor at  grid cell=(igrid_h,igrid_w) if the object exists in 
            this (grid cell, anchor) pair, else they simply contain 0.
            Bbox's center coordinates (x,y) are given between 0 and 1 relative to cell's origin 
            (i.e. 0.4 means 40% from cell's origin) and its dimensions relative to the cell's size 
            (i.e. 3.4 means 3.4 times the cell's grid)

            y_batch[iframe,igrid_h,igrid_w,ianchor,4] contains 1 if the object exists in this 
            (grid cell, anchor) pair, else it contains 0.

            y_batch[iframe,igrid_h,igrid_w,ianchor,5 + iclass] contains 1 if the iclass^th 
            class object exists in this (grid cell, anchor) pair, else it contains 0.
        '''
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.batch_size

        instance_count = 0

        # Prepare storage for outputs
        x_batch = np.zeros((r_bound - l_bound, self.img_h, self.img_w, 3)) # Input images
        y_batch = np.zeros((r_bound - l_bound, self.grid[1], self.grid[0],
                            self.nb_anchors, 5+len(self.labels)))
          
        grid_width = float(self.img_w)/self.grid[0] 
        grid_height = float(self.img_h)/self.grid[1]

        iou_vfunc = np.frompyfunc(lambda w1, h1, w2, h2: calculate_IOU(
                                np.array([w1, h1]), np.array([w2, h2])), 4, 1)

        for train_instance in self.images[l_bound:r_bound]:
            # Resize image
            img, all_objs = self.img_encoder.fit_data(train_instance)

            # Construct output from object's x, y, w, h
            true_box_index = 0
            for obj in all_objs:
                if (obj['xmax'] > obj['xmin']) and (obj['ymax'] > obj['ymin']) and (obj['name'] in self.labels):
                    center_x, center_y, center_w, center_h = self.img_encoder.abs2grid(obj)
                    
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    # Now we save in y_batch, center position relative to cell's origin
                    center_x -= grid_x
                    center_y -= grid_y

                    if (grid_x < self.grid[0]) and (grid_y < self.grid[1]):
                        obj_idx = self.labels.tolist().index(obj['name'])

                        ious = iou_vfunc(self.anchors[:,0], self.anchors[:,1], center_w, center_h)
                        best_anchor_id = np.argmax(ious)

                        # Assign ground truth x, y, w, h, confidence and class probs to y_batch
                        # it could happen that the same grid cell contain 2 similar shape objects
                        # as a result the same anchor box is selected as the best anchor box by the multiple objects
                        # in such ase, the object is over written
                        
                        # As stated in paper, width and height are predicted
                        # relative to anchor's dimensions 
                        
                        center_w = center_w*(grid_width/self.anchors[best_anchor_id, 0])
                        center_h = center_h*(grid_height/self.anchors[best_anchor_id, 1])
                        
                        bbox = [center_x, center_y, center_w, center_h]

                        # center_x, center_y, w, h and 1 because ground truth confidence is always 1
                        y_batch[instance_count, grid_y, grid_x, best_anchor_id, 0:4] = bbox
                        y_batch[instance_count, grid_y, grid_x, best_anchor_id, 4] = 1
                        # Class' probability for detected object
                        y_batch[instance_count, grid_y, grid_x, best_anchor_id, 5+obj_idx] = 1

                else:
                    print("Omitting image {} because of inconsistent labeling..".format(train_instance['filename']))

            x_batch[instance_count] = img
            instance_count += 1

        return x_batch, y_batch

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)        
