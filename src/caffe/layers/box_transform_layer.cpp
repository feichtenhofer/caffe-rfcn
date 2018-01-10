// ------------------------------------------------------------------
// D & T 
// Copyright (c) 2017 Graz University of Technology
// Written by Christoph Feichtenhofer [feichtenhofer@tugraz.at]
// Please email me if you find bugs
// ------------------------------------------------------------------

#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/box_transform_layer.hpp"

namespace caffe {

    template <typename Dtype>
    void BoxTransformLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        //CHECK_EQ(bottom[0]->num(), bottom[1]->num());
    }

    template <typename Dtype>
    void BoxTransformLayer<Dtype>::Reshape(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        const int num1 = bottom[0]->num();
        const int num2 = bottom[1]->num();

        //CHECK_EQ(bottom[0]->num(), bottom[1]->num());     
        top[0]->Reshape(std::min(num1,num2), 1, 1, 1);
        if (top.size() >= 2) {
            top[1]->Reshape(1, 1, 1, 1);
        }
    }

    template <typename Dtype>
    void BoxTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        //CHECK_EQ(bottom[0]->num(), bottom[1]->num());

        const Dtype* bbox_preds = bottom[0]->cpu_data();
        const int pred_dim = bottom[0]->channels();
        const int num1 = bottom[0]->num();

        // (batch_id, x1, y1, x2, y2)
        const Dtype* rois = bottom[1]->cpu_data();
        const int roi_dim = bottom[1]->channels();
        const int num2 = bottom[0]->num();

        const int num_out = std::min(num1, num2);



        // bbox mean and std
        bool do_bbox_norm = false;
        vector<float> bbox_means, bbox_stds;
        if (this->layer_param_.box_reg_param().bbox_mean_size() > 0
            && this->layer_param_.box_reg_param().bbox_std_size() > 0) {
            do_bbox_norm = true;
            int num_bbox_means = this->layer_param_.box_reg_param().bbox_mean_size();
            int num_bbox_stds = this->layer_param_.box_reg_param().bbox_std_size();
            CHECK_EQ(num_bbox_means, 4); CHECK_EQ(num_bbox_stds, 4);
            for (int i = 0; i < 4; i++) {
                bbox_means.push_back(this->layer_param_.box_reg_param().bbox_mean(i));
                bbox_stds.push_back(this->layer_param_.box_reg_param().bbox_std(i));
            }
        }

        // only regress specific batches
        bool do_batch_ind = false, batch_found = false;
        vector<int> batch_inds;
        if (this->layer_param_.box_reg_param().batch_ind_size() > 0) {
            do_batch_ind = true;
            int num_batch_ind = this->layer_param_.box_reg_param().batch_ind_size();
            for (int i = 0; i < num_batch_ind; i++) {
                batch_inds.push_back(this->layer_param_.box_reg_param().batch_ind(i));
            }
        }

        vector<vector<Dtype> > top_boxes;
        for (int i = 0; i < num_out; i++) {

            int label = 1; // 0 is background

            if (do_batch_ind)
            {
                batch_found = false;
                for (int j = 0; j < batch_inds.size(); j++)
                    if (rois[i*roi_dim] == batch_inds[j])
                    {
                        batch_found = true;
                        break;
                    }
                        
                if (!batch_found)
                    continue;
            }

            Dtype pred_x, pred_y, pred_w, pred_h;
            pred_x = bbox_preds[i*pred_dim + label * 4]; pred_y = bbox_preds[i*pred_dim + label * 4 + 1];
            pred_w = bbox_preds[i*pred_dim + label * 4 + 2]; pred_h = bbox_preds[i*pred_dim + label * 4 + 3];

            // bbox de-normalization
            if (do_bbox_norm) {
                pred_x *= bbox_stds[0]; pred_y *= bbox_stds[1];
                pred_w *= bbox_stds[2]; pred_h *= bbox_stds[3];
                pred_x += bbox_means[0]; pred_y += bbox_means[1];
                pred_w += bbox_means[2]; pred_h += bbox_means[3];
            }

            Dtype roi_x, roi_y, roi_w, roi_h;
            roi_x = rois[i*roi_dim + 1]; 
            roi_w = rois[i*roi_dim + 3] - roi_x + 1;

            roi_y = rois[i*roi_dim + 2]; 
            roi_h = rois[i*roi_dim + 4] - roi_y + 1;

            Dtype ctr_x, ctr_y, pred_ctr_x, pred_ctr_y;

            ctr_x = roi_x + 0.5*roi_w-1; ctr_y = roi_y + 0.5*roi_h-1;

            pred_ctr_x = pred_x*roi_w + ctr_x; pred_ctr_y = pred_y*roi_h + ctr_y;

            pred_w = roi_w*exp(pred_w); pred_h = roi_h*exp(pred_h);

            //pred_ctr_x = pred_ctr_x - pred_w / Dtype(2); 
            
            //pred_ctr_y = pred_ctr_y - pred_h / Dtype(2);
            vector<Dtype> bb(5);

            //bb[0] = rois[i*roi_dim];
            bb[0] = Dtype(0); // batch index is first
            bb[1] = pred_ctr_x - Dtype(0.5) * (pred_w - 1);
            bb[2] = pred_ctr_y - Dtype(0.5) * (pred_h - 1);
            bb[3] = pred_ctr_x + Dtype(0.5) * (pred_w - 1);
            bb[4] = pred_ctr_y + Dtype(0.5) * (pred_h - 1);

            top_boxes.push_back(bb);

        }

        top[0]->Reshape(top_boxes.size(), 5, 1, 1);
        Dtype* top_boxes_out = top[0]->mutable_cpu_data();
        for (int i = 0; i < top_boxes.size(); i++) {
            top_boxes_out[i*roi_dim] = top_boxes[i][0]; 
            top_boxes_out[i*roi_dim + 1] = top_boxes[i][1];
            top_boxes_out[i*roi_dim + 2] = top_boxes[i][2];
            top_boxes_out[i*roi_dim + 3] = top_boxes[i][3];
            top_boxes_out[i*roi_dim + 4] = top_boxes[i][4];
        }

    }


    INSTANTIATE_CLASS(BoxTransformLayer);
    REGISTER_LAYER_CLASS(BoxTransform);

}  // namespace caffe