import numpy as np 
import yaml
from pathlib import Path
from matplotlib import pyplot as plt

from utils import parse_annotations, ImageReader, normalize, BestAnchorBoxFinder, load_config
from yolo import YOLO_v2, BatchGenerator
from graphic_tools import plot_image


def main():
    root_dir = Path(__file__).resolve().parent.parent
    img_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/JPEGImages')
    ann_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/Annotations')
    
    # Extract hyperparameters
    config = load_config(root_dir.joinpath('scripts/config.yaml'))

    all_imgs, seen_labels = parse_annotations(ann_dir, img_dir, labels=config['labels'])

    train_batch_generator = BatchGenerator(all_imgs, config, norm=normalize, shuffle=True)
    [x_batch, b_batch], y_batch = train_batch_generator.__getitem__(idx=3)
    
    print("x_batch: (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels)           = {}".format(x_batch.shape))
    print("y_batch: (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes) = {}".format(y_batch.shape))
    print("b_batch: (BATCH_SIZE, 1, 1, 1, TRUE_BOX_BUFFER, 4)            = {}".format(b_batch.shape))

    # print(".."*40)
    # for i in range(5):
    #     print('Image {}:'.format(i))
    #     grid_y, grid_x, anchor_id = np.where(y_batch[i,:,:,:,4]==1) # BBoxes with 100% confidence
        
    #     plot_image(x_batch[i], y_batch[i], config['anchors'], config['labels'], True)
    #     plt.tight_layout()
    #     plt.show()

    model = YOLO_v2(config)
    # model.summary()
    nb_grids = 4 # Grids containing bboxes
    hardcode_batch = np.zeros(y_batch.shape)
    bboxes = np.random.rand(nb_grids*16).reshape(nb_grids, 4, -1) # 4 bboxes
    scores = (np.random.rand(nb_grids*4)*0.5 + 0.5).reshape(nb_grids, 4) # Confidence between(0.4 and 0.9)
    grids_h = np.random.randint(5, 5+int(nb_grids/2), size=nb_grids)
    grids_w = np.random.randint(5, 5+int(nb_grids/2), size=nb_grids)
    bboxes[:, :, 2:4] *= 16 # Increase bboxes dimensions
    hardcode_batch[0, grids_h, grids_w, :, 0:4] = bboxes # One bbox per anchor 
    hardcode_batch[0, grids_h, grids_w, :, 4] = scores
    clss = (np.random.rand(nb_grids*4*20)*0.5 + 0.5).reshape(4, 4, 20)
    # clss = np.zeros((4, 4, 20))
    # clss[:,:,5+10] = (np.random.rand(nb_grids*4)*0.5 +0.5).reshape(4, 4)
    hardcode_batch[0, grids_h, grids_w, :, 5:] = clss
    plot_image(x_batch[0], hardcode_batch[0], config['anchors'], config['labels'])
    plt.show()

    import tensorflow as tf
    from time import time
    hardcode_batch_t = tf.constant(hardcode_batch)
    s = time()
    new_batch = model.non_max_suppression(hardcode_batch_t)
    e = time()
    print('Time elapsed: {} s'.format(e-s))
    plot_image(x_batch[0], new_batch[0].numpy(), config['anchors'], config['labels'])
    plt.show()

if __name__ == '__main__':
    main()