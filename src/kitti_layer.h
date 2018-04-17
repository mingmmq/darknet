#ifndef KITTI_LAYER_H
#define KITTI_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_kitti_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void forward_kitti_layer(const layer l, network net);
void backward_kitti_layer(const layer l, network net);
void resize_kitti_layer(layer *l, int w, int h);
int kitti_num_detections(layer l, float thresh);

#ifdef GPU
void forward_kitti_layer_gpu(const layer l, network net);
void backward_kitti_layer_gpu(layer l, network net);
#endif

#endif
