from matplotlib import pyplot as plt 
import numpy as np
import seaborn as sns

def plot_img_with_gridcell(img, grids, color="yellow"):
    """

    Arguments:
    ---------
    img : [np.array] (rows x columns x 3)
    grids: [tup] Tuple containing number of grids 
    """
    
    plt.figure(figsize=(15, 15))
    plt.imshow(img)

    img_h, img_w, _ = img.shape 
    grid_w, grid_h = grids
    

    # Vertical lines
    v_lines = np.array(range(grid_w))*img_w/grid_w
    [plt.axvline(line, color=color, alpha=0.3) for line in v_lines]
    plt.xticks([(i + 0.5)*img_w/grid_w for i in range(grid_w)],
                ["iGRIDW={}".format(i) for i in range(grid_w)])

    # Horizontal lines
    h_lines = np.array(range(grid_h))*img_h/grid_h
    [plt.axhline(line, color=color, alpha=0.3) for line in h_lines]
    plt.yticks([(i + 0.5)*img_h/grid_h for i in range(grid_h)],
                ["iGRIDH={}".format(i) for i in range(grid_h)])

def plot_bbox(output_instance, img_shape, labels):
    color_palette = list(sns.xkcd_rgb.values())

    grid_y, grid_x, anchor_id = np.where(output_instance[:,:,:,4] == 1)

    confident_outputs = output_instance[grid_y, grid_x, anchor_id] # Boxes with confidence 1
    predicted_bboxes = confident_outputs[:, 0:4] # x, y, w, h

    nb_grids_y, nb_grids_x, _, _ = output_instance.shape
    grid_width = img_shape[0]/nb_grids_x
    grid_height = img_shape[1]/nb_grids_y

    offsets_x = grid_x*grid_width
    offsets_y = grid_y*grid_height

    bbox_ids, labels_ids = np.where(confident_outputs[:, 5:])
    names = labels[labels_ids]

    id_obj = 0
    for bbox in predicted_bboxes:
        x, y, w, h = bbox

        # We need to rescale as x,y are given in px respecto to the 0,0 of current gridcell
        # and w, h are given times*gridcell_width and height respectively
        x = x + offsets_x[id_obj]
        y = y + offsets_y[id_obj]
        w = w*grid_width
        h = h*grid_height

        c = color_palette[id_obj]
        
        xmin = x - 0.5*w
        xmax = x + 0.5*w
        ymin = y - 0.5*h
        ymax = y + 0.5*h

        plt.text(x, y, "X",color=c,fontsize=15)
        plt.text(xmin, ymin, names[id_obj], color ='black', fontsize=7, backgroundcolor=c)
        plt.plot(np.array([xmin,xmin]),
                    np.array([ymin,ymax]),color=c,linewidth=7)
        plt.plot(np.array([xmin,xmax]),
                    np.array([ymin,ymin]),color=c,linewidth=7)
        plt.plot(np.array([xmax,xmax]),
                    np.array([ymax,ymin]),color=c,linewidth=7)  
        plt.plot(np.array([xmin,xmax]),
                    np.array([ymax,ymax]),color=c,linewidth=7)
        id_obj += 1