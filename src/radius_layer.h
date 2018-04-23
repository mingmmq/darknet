#ifndef RADIUS_LAYER_H
#define RADIUS_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_radius_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void forward_radius_layer(const layer l, network net);
void backward_radius_layer(const layer l, network net);
void resize_radius_layer(layer *l, int w, int h);
int radius_num_detections(layer l, float thresh);

#ifdef GPU
void forward_radius_layer_gpu(const layer l, network net);
void backward_radius_layer_gpu(layer l, network net);
#endif

#endif
