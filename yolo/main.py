import numpy as np 
import tensorflow as tf
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from yolo.utils import parse_annotations, ImageReader, normalize, load_config
from yolo.utils import tf_non_max_suppression, non_max_suppression, extract_bboxes, non_max_suppression_v2
from yolo.core import YOLO_v2, BatchGenerator, yolo_loss, adjust_yolo_output, LearningRateScheduler
from yolo.graphic_tools import plot_image


def main():
    root_dir = Path(__file__).resolve().parent
    img_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/JPEGImages')
    ann_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/Annotations')
    
    # Extract hyperparameters
    config = load_config(root_dir.joinpath('config.yaml'))

    all_imgs, seen_labels = parse_annotations(ann_dir, img_dir, labels=config['labels'])
    all_imgs = all_imgs[:320]
    # Train, val, test split
    N = len(all_imgs)
    train_dataset, val_dataset, test_dataset = np.split(all_imgs, [int(0.8*N), int(0.9*N)])
    # print(val_dataset)
    # train_batch_generator = BatchGenerator(all_imgs, config, norm=normalize, shuffle=True)
    train_batch_generator = BatchGenerator(train_dataset, config, norm=normalize, shuffle=True)
    val_batch_generator = BatchGenerator(val_dataset, config, norm=normalize, shuffle=True)
    test_batch_generator = BatchGenerator(test_dataset, config, norm=normalize, shuffle=True)
    # x_batch, y_batch = train_batch_generator.__getitem__(idx=3)
    
    # print("x_batch: (BATCH_SIZE, IMAGE_H, IMAGE_W, N channels)           = {}".format(x_batch.shape))
    # print("y_batch: (BATCH_SIZE, GRID_H, GRID_W, BOX, 4 + 1 + N classes) = {}".format(y_batch.shape))

    # print(".."*40)
    # for i in range(5):
    #     print('Image {}:'.format(i))
    #     grid_y, grid_x, anchor_id = np.where(y_batch[i,:,:,:,4]==1) # BBoxes with 100% confidence
        
    #     plot_image(x_batch[i], y_batch[i], config['anchors'], config['labels'], True)
    #     plt.tight_layout()
    #     plt.show()

    test = val_batch_generator.__getitem__(idx=0)
    print(test)

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

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, mode='min', 
                                verbose=1)
    
    checkpoint = ModelCheckpoint(str(check_dir.joinpath('yolo_v2_weights_voc_2012.h5')), 
                                monitor='val_loss', save_best_only=True, mode='min', 
                                save_freq=5*config['batch_size'], verbose=1)

    csv_logger = CSVLogger(str(logs_dir.joinpath('log.csv')), append=True, separator=';')

    scheduler = LearningRateScheduler(config['schedule'])

    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss=yolo_loss(lambda_coord=5, lambda_noobj=0.5), optimizer=optimizer)
    print("Model compiled successfully!")

    tf.config.experimental_run_functions_eagerly(True)

    model.fit   (x               = train_batch_generator,
                validation_data  = val_batch_generator, 
                epochs           = 50, 
                verbose          = 1,
                callbacks        = [early_stop, checkpoint, csv_logger, scheduler], 
                max_queue_size   = 3)

    # Prediction
    # model.load_weights(str(root_dir.joinpath('checkpoints/yolo_v2_weights_voc_2012.h5')))
    # y_pred = model.predict(x_batch)

    # from yolo.graphic_tools import plot_img_with_gridcell, plot_bbox_abs
    # bboxes_true, scores = extract_bboxes(y_batch, config['anchors'], config['image_shape'])
    # class_ids = np.argmax(scores[0], axis=1)
    # scores = scores[0][range(len(class_ids)), class_ids]
    # plot_img_with_gridcell(x_batch[0], config['grid'])
    # plot_bbox_abs(bboxes_true[0], scores, class_ids, config['labels'])
    # plt.show()

    # # print(extract_bboxes(y_pred, config['anchors'], config['image_shape']))
    # y_pred_adj = adjust_yolo_output(y_pred)
    # bboxes_pred, scores = extract_bboxes(y_pred_adj, config['anchors'], config['image_shape'])
    # class_ids = np.argmax(scores[0], axis=1)
    # scores_0 = scores[0][range(len(class_ids)), class_ids]
    # bboxes_pred_0, scores_0, class_ids_0 = non_max_suppression_v2(bboxes_pred[0], scores_0, class_ids, 0.5, 0.6)

    # print(bboxes_pred_0)
    # print(scores_0)
    # plot_img_with_gridcell(x_batch[0], config['grid'])
    # plot_bbox_abs(bboxes_pred_0, scores_0, class_ids_0, config['labels'])
    # plt.show()

if __name__ == '__main__':
    main()