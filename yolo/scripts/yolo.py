import numpy as np

from tensorflow.nn import space_to_depth
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, LeakyReLU, concatenate
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.backend import get_value, set_value

from pathlib import Path

from utils import BestAnchorBoxFinder, ImageReader, WeightsReader, read_bytes

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
    return space_to_depth(x, block_size=2)

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

        self.nb_anchors = len(config['anchors'])
        self.nb_classes = len(config['labels'])
        self.true_box_buffer = config['true_box_buffer']

        self.img_shape = config['image_shape']
        self.grid = config['grid']

        # Specifications taken of original paper (based on Darknet-19)
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
        
        self.lambda_1 = Lambda(space_to_depth_x2)
        # small hack to allow true_boxes to be registered when Keras build the model 
        # for more information: https://github.com/fchollet/keras/issues/2790
        self.lambda_2 = Lambda(lambda args: args[0])
        
    def call(self, inputs, training=False):
        x, true_boxes = inputs
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
                skip_connection = self.lambda_1(skip_connection)
                x = concatenate([skip_connection, x])

        # 23rd Layer does not have an activation nor normalization
        x = self.convs[len(self.convs)-1](x)

        output = Reshape((self.grid[1], self.grid[0], self.nb_anchors, 5+self.nb_classes))(x)
        output = self.lambda_2([output, true_boxes])
        
        return output   

    def summary(self):
        # Hack for being able to see built model
        x = Input(shape=(self.img_shape[1], self.img_shape[1], 3))
        true_boxes = Input(shape=(1, 1, 1, self.true_box_buffer , 4))

        return Model(inputs=[x, true_boxes], outputs=self.call([x, true_boxes])).summary()

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

        self.true_box_buffer = config['true_box_buffer'] # Maximun objects per box!!
        self.batch_size = config['batch_size']
        self.nb_anchors = len(config['anchors'])

        self.best_anc_finder = BestAnchorBoxFinder(config['anchors'])
        
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

            y_batch[iframe,igrid_h,igrid_w,ianchor,4] contains 1 if the object exists in this 
            (grid cell, anchor) pair, else it contains 0.

            y_batch[iframe,igrid_h,igrid_w,ianchor,5 + iclass] contains 1 if the iclass^th 
            class object exists in this (grid cell, anchor) pair, else it contains 0.


        b_batch: [np.array] Array of shape (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4).

            b_batch[iframe,1,1,1,ibuffer,ianchor,:] contains ibufferth object's 
            (center_x,center_y,center_w,center_h) in iframeth frame.

            If ibuffer > N objects in iframeth frame, then the values are simply 0.

            TRUE_BOX_BUFFER has to be some large number, so that the frame with the 
            biggest number of objects can also record all objects.

            The order of the objects do not matter.

            This is just a hack to easily calculate loss. 
        
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
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.true_box_buffer, 4))

        for train_instance in self.images[l_bound:r_bound]:
            # Resize image
            img, all_objs = self.img_encoder.fit_data(train_instance)

            # Construct output from object's x, y, w, h
            true_box_index = 0
            for obj in all_objs:
                if (obj['xmax'] > obj['xmin']) and (obj['ymax'] > obj['ymin']) and (obj['name'] in self.labels):
                    center_x, center_y, center_w, center_h = self.img_encoder.rescale_center_rel_grids(obj)
                    
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if (grid_x < self.grid[0]) and (grid_y < self.grid[1]):
                        obj_idx = self.labels.tolist().index(obj['name'])

                        bbox = [center_x, center_y, center_w, center_h]
                        best_anchor_id, max_iou = self.best_anc_finder.find(center_w, center_h)
                        # Assign ground truth x, y, w, h, confidence and class probs to y_batch
                        # it could happen that the same grid cell contain 2 similar shape objects
                        # as a result the same anchor box is selected as the best anchor box by the multiple objects
                        # in such ase, the object is over written

                        # center_x, center_y, w, h and 1 because ground truth confidence is always 1
                        y_batch[instance_count, grid_y, grid_x, best_anchor_id, 0:4] = bbox
                        y_batch[instance_count, grid_y, grid_x, best_anchor_id, 4] = 1
                        # Class' probability for detected object
                        y_batch[instance_count, grid_y, grid_x, best_anchor_id, 5+obj_idx] = 1

                        # Assign the true bbox to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = bbox
                        true_box_index = (true_box_index + 1) % self.true_box_buffer

                else:
                    print("Omitting image {} because of incorrect labeling..".format(train_instance['filename']))

            x_batch[instance_count] = img
            instance_count += 1

        return [x_batch, b_batch], y_batch

    def __len__(self):
        return np.ceil(float(len(self.images))/self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.images)

def main():
    from utils import load_config

    from pathlib import Path

    p = Path(__file__).resolve().parent 
    config = load_config(p.joinpath('config.yaml'))

    model = YOLO_v2(config)
    model.summary()
    model.load_weights_from_file(p.joinpath('yolov2.weights'))
    # yolov2 = YOLO_v2(config)
    # model = yolov2.get_model()
    # model.summary()
    
    # w_reader = WeightsReader(p.joinpath('yolov2.weights'))
    # w_reader.load_weights(model)

    # import yaml
    # from pathlib import Path
    # from matplotlib import pyplot as plt

    # from utils import parse_annotations, normalize, load_config
    # from graphic_tools import plot_img_with_gridcell, plot_bbox

    # root_dir = Path(__file__).resolve().parent.parent
    # img_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/JPEGImages')
    # ann_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/Annotations')
    
    # config = load_config(root_dir.joinpath('scripts/config.yaml'))

    # all_imgs, _ = parse_annotations(ann_dir, img_dir, labels=config['labels'])
    # print("First 3 images:\n{}\n{}\n{}".format(all_imgs[0], all_imgs[1], all_imgs[2]))
    # print(".."*40)

    # train_batch_generator = BatchGenerator(all_imgs, config, norm=normalize, shuffle=True)
    # [x_batch, b_batch], y_batch = train_batch_generator.__getitem__(idx=3)
    # print("x_batch: (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels)           = {}".format(x_batch.shape))
    # print("y_batch: (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes) = {}".format(y_batch.shape))
    # print("b_batch: (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4)            = {}".format(b_batch.shape))

    # print(".."*40)
    # for i in range(5):
    #     print('Image {}:'.format(i))
    #     img = x_batch[i]
    #     grid_y, grid_x, anchor_id = np.where(y_batch[i,:,:,:,4]==1) # BBoxes with 100% confidence
        
    #     plot_img_with_gridcell(img, config['grid'])
    #     plot_bbox(y_batch[i], config['image_shape'], config['labels'])
    #     plt.tight_layout()
    #     plt.show()

if __name__ == '__main__':
    main()
