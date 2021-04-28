import numpy as np 
import yaml
from pathlib import Path
from matplotlib import pyplot as plt

from utils import parse_annotations, ImageReader, normalize, BestAnchorBoxFinder

def main():
    root_dir = Path(__file__).resolve().parent.parent
    img_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/JPEGImages')
    ann_dir = root_dir.joinpath('data/VOCdevkit/VOC2012/Annotations')
    
    # Extract hyperparameters
    with open(root_dir.joinpath('scripts/hyperparameters.yaml'), 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
        classes = config['labels']
        anchors = config['anchors']

    all_imgs, seen_labels = parse_annotations(ann_dir, img_dir, labels=classes)

    img_encoder = ImageReader(img_width=416, img_height=416, norm=normalize,
                              grids=(13, 13))
    print('Input..')
    test_input = all_imgs[0]
    for key, value in test_input.items():
        print("\t{} : {}".format(key, value))

    print('Ouput..')
    img, all_objs = img_encoder.fit_data(test_input)
    print('\n{}'.format(all_objs))

    plt.imshow(img)
    plt.title('Image shape: {}'.format(img.shape))
    plt.show()

    _ANCHORS01 = [[0.08285376, 0.13705531],
                  [0.20850361, 0.39420716],
                  [0.80552421, 0.77665105],
                  [0.42194719, 0.62385487]]
    print(".."*40)
    print("The three example anchor boxes:")
    count = 0
    for w, h in _ANCHORS01:
        print("anchor box index={}, w={}, h={}".format(count, w, h))
        count += 1
    print(".."*40)   
    print("Allocate bounding box of various width and height into the three anchor boxes:")  
    babf = BestAnchorBoxFinder(_ANCHORS01)
    for w in range(1,9,2):
        w /= 10.
        for h in range(1,9,2):
            h /= 10.
            best_anchor,max_iou = babf.find(w,h)
            print("bounding box (w = {}, h = {}):\n\tBbest anchor box index = {}, IOU = {:03.2f}".format(
                w,h,best_anchor,max_iou))

    obj    = {'xmin': 150, 'ymin': 84, 'xmax': 300, 'ymax': 294}
    # config = {"IMAGE_W":416,"IMAGE_H":416,"GRID_W":13,"GRID_H":13}
    # center_x, center_y = rescale_centerxy(obj,config)
    # center_w, center_h = rescale_cebterwh(obj,config)

    center_x, center_y, center_w, center_h = img_encoder.rescale_center_rel_grids(obj)
    grid_w, grid_h = img_encoder.grids

    print("cebter_x abd cebter_w should range between 0 and {}".format(grid_w))
    print("cebter_y abd cebter_h should range between 0 and {}".format(grid_h))

    print("center_x = {:06.3f} range between 0 and {}".format(center_x, grid_w))
    print("center_y = {:06.3f} range between 0 and {}".format(center_y, grid_h))
    print("center_w = {:06.3f} range between 0 and {}".format(center_w, grid_w))
    print("center_h = {:06.3f} range between 0 and {}".format(center_h, grid_h))

if __name__ == '__main__':
    main()