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
        
    #     plot_image(x_batch[i], y_batch[i], config['labels'], True)
    #     plt.tight_layout()
    #     plt.show()

    model = YOLO_v2(config)
    # model.summary()
    nb_grids = 4 # Grids containing bboxes
    hardcode_batch = np.zeros(y_batch.shape)
    bboxes = np.random.rand(nb_grids*16).reshape(nb_grids, 4, -1) # 4 bboxes
    scores = (np.random.rand(nb_grids*4)*0.4 + 0.5).reshape(nb_grids, 4) # Confidence between(0.4 and 0.9)
    grids_h = np.random.randint(5, 5+int(nb_grids/2), size=nb_grids)
    grids_w = np.random.randint(5, 5+int(nb_grids/2), size=nb_grids)
    bboxes[:, :, 2:4] *= 4 # Increase bboxes dimensions
    hardcode_batch[0, grids_h, grids_w, :, 0:4] = bboxes # One bbox per anchor 
    hardcode_batch[0, grids_h, grids_w, :, 4] = scores
    clss = np.zeros(20)
    clss[14] = 1
    hardcode_batch[0, grids_h, grids_w, :, 5:] = clss
    import tensorflow as tf
    hardcode_batch_t = tf.constant(hardcode_batch)
    print(np.sum(hardcode_batch[0,:,:,:,5:].reshape((-1, len(clss))), axis=0))
    model.non_max_suppression(hardcode_batch)


    # plot_image(x_batch[0], hardcode_batch[0], config['labels'], True, 0.5)
    # plt.tight_layout()
    # plt.show()
    # print('Bboxes before reshape:\n{}\nBboxes after reshape:\n{}'.format(bboxes, bboxes.reshape(-1,4)))
    # mask = model.class_non_max_suppression(bboxes.reshape(-1,4), scores.reshape(-1))
    # print(mask.reshape(nb_grids, 4, -1))
    # hardcode_batch[0, grids_h, grids_w, :, 4] = mask.reshape(nb_grids, 4)*scores
    # plot_image(x_batch[0], hardcode_batch[0], config['labels'], True, 0.5)
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()