// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/rfcn_layers.hpp"
#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

  template <typename Dtype>
  __global__ void PSROIPoolingForward(
    const int nthreads,
    const Dtype* bottom_data,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
    const int output_dim,
    const int group_size,
    Dtype* top_data,
    int* mapping_channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom 
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph) * bin_size_h
                          + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
                          + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
                        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
                        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0),width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int gw = pw;
      int gh = ph;
      int c = (ctop*group_size + gh)*group_size + gw;

      bottom_data += (roi_batch_ind * channels + c) * height * width;
      Dtype out_sum = 0;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          out_sum += bottom_data[bottom_index];
        }
      }

      Dtype bin_area = (hend - hstart)*(wend - wstart);
      top_data[index] = is_empty? 0. : out_sum/bin_area;
      mapping_channel[index] = c;
    }
  }

  template <typename Dtype>
  __global__ void PSROIPoolingForward_RoiAlign(
      const int nthreads,
      const Dtype* bottom_data,
      const Dtype spatial_scale,
      const int channels,
      const int height, const int width,
      const int pooled_height, const int pooled_width,
      const Dtype* bottom_rois,
      const int output_dim,
      const int group_size,
      Dtype* top_data,
      int* mapping_channel) {
      CUDA_KERNEL_LOOP(index, nthreads) {
          // The output is in order (n, ctop, ph, pw)
          int pw = index % pooled_width;
          int ph = (index / pooled_width) % pooled_height;
          int ctop = (index / pooled_width / pooled_height) % output_dim;
          int n = index / pooled_width / pooled_height / output_dim;

          // [start, end) interval for spatial sampling
          bottom_rois += n * 5;
          int roi_batch_ind = bottom_rois[0];
          Dtype roi_start_w = bottom_rois[1] * spatial_scale;
          Dtype roi_start_h = bottom_rois[2] * spatial_scale;
          Dtype roi_end_w = bottom_rois[3] * spatial_scale;
          Dtype roi_end_h = bottom_rois[4] * spatial_scale;

          Dtype roi_width = roi_end_w - roi_start_w;
          Dtype roi_height = roi_end_h - roi_start_h;

          // Compute w and h at bottom
          Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
          Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

          // bin size is ratio of scaled roi height and 7,14, so it would be around 1,2,3

          //ph, pw is the position in the roi

          //therefore, we need use, bin size and ph to get the position of the 4 feature values

          int x1, x2, y1, y2;
          float px, py, pxmax, pymax, pxmin, pymin;
          pxmax = min(max(roi_start_w + static_cast<Dtype>(pw + 0.75) * bin_size_w, 0.001), width - 1.001);
          pymax = min(max(roi_start_h + static_cast<Dtype>(ph + 0.75) * bin_size_h, 0.001), height - 1.001);
          pxmin = min(max(roi_start_w + static_cast<Dtype>(pw + 0.25) * bin_size_w, 0.001), width - 1.001);
          pymin = min(max(roi_start_h + static_cast<Dtype>(ph + 0.25) * bin_size_h, 0.001), height - 1.001);

          Dtype out_sum = 0;
          int gw = pw;
          int gh = ph;
          int c = (ctop*group_size + gh)*group_size + gw;

          bottom_data += (roi_batch_ind * channels + c) * height * width;

          px = pxmin;
          py = pymin;

          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);

          out_sum += (px - x1)*(py - y1) * bottom_data[int(y2*width + x2)];
          out_sum += (px - x1)*(y2 - py) * bottom_data[int(y1*width + x2)];
          out_sum += (x2 - px)*(py - y1) * bottom_data[int(y2*width + x1)];
          out_sum += (x2 - px)*(y2 - py) * bottom_data[int(y1*width + x1)];

          px = pxmax;
          py = pymax;

          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);

          out_sum += (px - x1)*(py - y1) * bottom_data[int(y2*width + x2)];
          out_sum += (px - x1)*(y2 - py) * bottom_data[int(y1*width + x2)];
          out_sum += (x2 - px)*(py - y1) * bottom_data[int(y2*width + x1)];
          out_sum += (x2 - px)*(y2 - py) * bottom_data[int(y1*width + x1)];

          px = pxmin;
          py = pymax;

          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);

          out_sum += (px - x1)*(py - y1) * bottom_data[int(y2*width + x2)];
          out_sum += (px - x1)*(y2 - py) * bottom_data[int(y1*width + x2)];
          out_sum += (x2 - px)*(py - y1) * bottom_data[int(y2*width + x1)];
          out_sum += (x2 - px)*(y2 - py) * bottom_data[int(y1*width + x1)];

          px = pxmax;
          py = pymin;

          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);

          out_sum += (px - x1)*(py - y1) * bottom_data[int(y2*width + x2)];
          out_sum += (px - x1)*(y2 - py) * bottom_data[int(y1*width + x2)];
          out_sum += (x2 - px)*(py - y1) * bottom_data[int(y2*width + x1)];
          out_sum += (x2 - px)*(y2 - py) * bottom_data[int(y1*width + x1)];

          top_data[index] = out_sum / 4;
          mapping_channel[index] = c;
      }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_rois = bottom[1]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
    int count = top[0]->count();
    caffe_gpu_set(count, Dtype(0), top_data);
    caffe_gpu_set(count, -1, mapping_channel_ptr);
    if (roi_align_ == false){
        // NOLINT_NEXT_LINE(whitespace/operators)
        PSROIPoolingForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
            count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_,
            pooled_width_, bottom_rois, output_dim_, group_size_, top_data, mapping_channel_ptr);
    }
    else {
        PSROIPoolingForward_RoiAlign<Dtype> << <CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_,
            channels_, height_, width_, pooled_height_,
            pooled_width_, bottom_rois, output_dim_, group_size_,
            top_data, mapping_channel_ptr);
    }
    CUDA_POST_KERNEL_CHECK;

  }

  template <typename Dtype>
  __global__ void PSROIPoolingBackwardAtomic(
    const int nthreads,
    const Dtype* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const Dtype spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int output_dim, 
    Dtype* bottom_diff,
    const Dtype* bottom_rois) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // The output is in order (n, ctop, ph, pw)
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale;
      Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale;
      Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale;
      Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
      Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom 
      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int hstart = floor(static_cast<Dtype>(ph)* bin_size_h
        + roi_start_h);
      int wstart = floor(static_cast<Dtype>(pw)* bin_size_w
        + roi_start_w);
      int hend = ceil(static_cast<Dtype>(ph + 1) * bin_size_h
        + roi_start_h);
      int wend = ceil(static_cast<Dtype>(pw + 1) * bin_size_w
        + roi_start_w);
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      Dtype bin_area = (hend - hstart)*(wend - wstart);
      Dtype diff_val = is_empty ? 0. : top_diff[index] / bin_area;
      for (int h = hstart; h < hend; ++h){
        for (int w = wstart; w < wend; ++w){
          int bottom_index = h*width + w;
          caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
        }
      }
    }
  }

  template <typename Dtype>
  __global__ void PSROIPoolingBackwardAtomic_RoiAlign(
      const int nthreads,
      const Dtype* top_diff,
      const int* mapping_channel,
      const int num_rois,
      const Dtype spatial_scale,
      const int channels,
      const int height, const int width,
      const int pooled_height, const int pooled_width,
      const int output_dim,
      Dtype* bottom_diff,
      const Dtype* bottom_rois) {
      CUDA_KERNEL_LOOP(index, nthreads) {
          // The output is in order (n, ctop, ph, pw)
          int pw = index % pooled_width;
          int ph = (index / pooled_width) % pooled_height;
          int n = index / pooled_width / pooled_height / output_dim;

          // [start, end) interval for spatial sampling
          bottom_rois += n * 5;
          int roi_batch_ind = bottom_rois[0];
          Dtype roi_start_w = bottom_rois[1] * spatial_scale;
          Dtype roi_start_h = bottom_rois[2] * spatial_scale;
          Dtype roi_end_w = bottom_rois[3] * spatial_scale;
          Dtype roi_end_h = bottom_rois[4] * spatial_scale;

          Dtype roi_width = roi_end_w - roi_start_w;
          Dtype roi_height = roi_end_h - roi_start_h;

          // Compute w and h at bottom
          Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
          Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

          int x1, x2, y1, y2;
          float pxmin, pymin, pxmax, pymax, py, px;
          pxmax = min(max(roi_start_w + static_cast<Dtype>(pw + 0.75) * bin_size_w, 0.001), width - 1.001);
          pymax = min(max(roi_start_h + static_cast<Dtype>(ph + 0.75) * bin_size_h, 0.001), height - 1.001);
          pxmin = min(max(roi_start_w + static_cast<Dtype>(pw + 0.25) * bin_size_w, 0.001), width - 1.001);
          pymin = min(max(roi_start_h + static_cast<Dtype>(ph + 0.25) * bin_size_h, 0.001), height - 1.001);

          // Compute c at bottom
          int c = mapping_channel[index];
          Dtype* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
          Dtype diff_val = 0;
          diff_val = top_diff[index] / 4;

          px = pxmin;
          py = pymin;
          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);
          caffe_gpu_atomic_add(diff_val * (px - x1)*(py - y1), offset_bottom_diff + int(y2*width + x2));
          caffe_gpu_atomic_add(diff_val * (px - x1)*(y2 - py), offset_bottom_diff + int(y1*width + x2));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(py - y1), offset_bottom_diff + int(y2*width + x1));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(y2 - py), offset_bottom_diff + int(y1*width + x1));

          px = pxmax;
          py = pymax;
          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);
          caffe_gpu_atomic_add(diff_val * (px - x1)*(py - y1), offset_bottom_diff + int(y2*width + x2));
          caffe_gpu_atomic_add(diff_val * (px - x1)*(y2 - py), offset_bottom_diff + int(y1*width + x2));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(py - y1), offset_bottom_diff + int(y2*width + x1));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(y2 - py), offset_bottom_diff + int(y1*width + x1));

          px = pxmin;
          py = pymax;
          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);
          caffe_gpu_atomic_add(diff_val * (px - x1)*(py - y1), offset_bottom_diff + int(y2*width + x2));
          caffe_gpu_atomic_add(diff_val * (px - x1)*(y2 - py), offset_bottom_diff + int(y1*width + x2));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(py - y1), offset_bottom_diff + int(y2*width + x1));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(y2 - py), offset_bottom_diff + int(y1*width + x1));

          px = pxmax;
          py = pymin;
          x1 = floor(px);
          x2 = ceil(px);
          y1 = floor(py);
          y2 = ceil(py);
          caffe_gpu_atomic_add(diff_val * (px - x1)*(py - y1), offset_bottom_diff + int(y2*width + x2));
          caffe_gpu_atomic_add(diff_val * (px - x1)*(y2 - py), offset_bottom_diff + int(y1*width + x2));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(py - y1), offset_bottom_diff + int(y2*width + x1));
          caffe_gpu_atomic_add(diff_val * (x2 - px)*(y2 - py), offset_bottom_diff + int(y1*width + x1));
      }
  }

  template <typename Dtype>
  void PSROIPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (!propagate_down[0]) {
      return;
    }

    const Dtype* bottom_rois = bottom[1]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    const int* mapping_channel_ptr = mapping_channel_.gpu_data();
    caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());
    caffe_gpu_set(bottom_count, Dtype(0), bottom_diff);
    const int count = top[0]->count();
    if (roi_align_ == false){
        // NOLINT_NEXT_LINE(whitespace/operators)
        PSROIPoolingBackwardAtomic<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
            count, top_diff, mapping_channel_ptr, top[0]->num(), spatial_scale_,
            channels_, height_, width_, pooled_height_, pooled_width_, output_dim_,
            bottom_diff, bottom_rois);
    }
    else {
        PSROIPoolingBackwardAtomic_RoiAlign<Dtype> << <CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS >> >(count, top_diff, mapping_channel_ptr,
            top[0]->num(), spatial_scale_, channels_, height_, width_,
            pooled_height_, pooled_width_, output_dim_, bottom_diff,
            bottom_rois);
    }

    CUDA_POST_KERNEL_CHECK;
  }

  INSTANTIATE_LAYER_GPU_FUNCS(PSROIPoolingLayer);

}  // namespace caffe
