import numpy as np 
import yaml
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from pathlib import Path
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

import argparse

from yolo.utils import parse_annotations, ImageReader, normalize, load_config
from yolo.utils import tf_non_max_suppression, non_max_suppression, extract_bboxes
from yolo.core import YOLO_v2, BatchGenerator, adjust_yolo_output, LearningRateScheduler
from yolo.core import yolo_loss_v2, yolo_loss_v3
from yolo.graphic_tools import plot_image

parser = argparse.ArgumentParser(prog = "yolo")
parser.add_argument("action", choices=["train", "predict"], help="action to take.")
args = parser.parse_args()

def main(action):
    root_dir = Path(__file__).resolve().parent
    img_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/JPEGImages')
    ann_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/Annotations')
    
    # Extract hyperparameters
    config = load_config(root_dir.joinpath('config.yaml'))

    all_imgs, seen_labels = parse_annotations(ann_dir, img_dir, labels=config['labels'])

    # Train, val, test split
    N = len(all_imgs)
    train_dataset, val_dataset, test_dataset = np.split(all_imgs, [int(0.8*N), int(0.9*N)])

    # train_batch_generator = BatchGenerator(all_imgs, config, norm=normalize, shuffle=True)
    train_batch_generator = BatchGenerator(train_dataset, config, norm=normalize, shuffle=True)
    val_batch_generator = BatchGenerator(val_dataset, config, norm=normalize, shuffle=True)
    test_batch_generator = BatchGenerator(test_dataset, config, norm=normalize, shuffle=True)

    logs_dir = root_dir.joinpath('logs')     
    if not logs_dir.is_dir():
        print("Creating log dir..")
        logs_dir.mkdir()

    check_dir = root_dir.joinpath('checkpoints')
    if not check_dir.is_dir():
        print("Creating check dir..")
        check_dir.mkdir()

    model = YOLO_v2(config)
    model.build((None, model.img_shape[1], model.img_shape[0], 3))
    model.summary()

    if root_dir.joinpath('yolov2.weights').exists():
        print('Loading pre-trained weights..')
        model.load_weights_from_file(root_dir.joinpath('yolov2.weights'))

    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss=yolo_loss_v3(config), optimizer=optimizer)
    print("Model compiled successfully!")
    
    weights_file = root_dir.joinpath('checkpoints/yolo_v2_weights_voc_2012.h5')
    if weights_file.exists():
        print('Loading weights from checkpoint..')
        model.load_weights(str(weights_file))

    if action == "train":
        # Training

        tf.config.experimental_run_functions_eagerly(True)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=1, mode='min', 
                            verbose=1)
        
        checkpoint = ModelCheckpoint(str(check_dir.joinpath('yolo_v2_weights_voc_2012.h5')), 
                                    monitor='loss', save_best_only=True, mode='min', 
                                    save_freq=5*config['batch_size'], verbose=1)

        csv_logger = CSVLogger(str(logs_dir.joinpath('log.csv')), append=True, separator=';')

        scheduler = LearningRateScheduler(config['schedule'])

        model.fit   (x               = train_batch_generator,
                    validation_data  = val_batch_generator, 
                    epochs           = 50, 
                    verbose          = 1,
                    callbacks        = [early_stop, checkpoint, csv_logger, scheduler], 
                    max_queue_size   = 3)

    elif action == "predict":
        # # Prediction
        from yolo.graphic_tools import plot_img_with_gridcell, plot_bbox_abs

        x_batch, y_batch = train_batch_generator.__getitem__(idx=3)

        bboxes_p, scores_p, class_ids_p = model.predict_bboxes(x_batch)
        
        bboxes_batch_t, confs_batch_t, logits_batch_t = extract_bboxes(y_batch, config['anchors'], config['image_shape'])
        bboxes_t = []
        scores_t = []
        class_ids_t = []
        for bboxes, confs, logits in zip(bboxes_batch_t, confs_batch_t, logits_batch_t):
            boxes, scores, class_ids = non_max_suppression(bboxes, confs,
                                                           logits, 0.6, 
                                                           0.5)
            bboxes_t.append(boxes)
            scores_t.append(scores)
            class_ids_t.append(class_ids)

        for i in range(config['batch_size']):

            plot_img_with_gridcell(x_batch[i], config['grid'])
            plot_bbox_abs(bboxes_t[i], scores_t[i], class_ids_t[i], config['labels'])
            plt.show()

            plot_img_with_gridcell(x_batch[i], config['grid'])
            plot_bbox_abs(bboxes_p[i], scores_p[i], class_ids_p[i], config['labels'])
            plt.show()
    else:
        print('Not valid action')

if __name__ == '__main__':

    main(args.action)