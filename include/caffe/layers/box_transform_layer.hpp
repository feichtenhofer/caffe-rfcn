// ------------------------------------------------------------------
// D & T 
// Copyright (c) 2017 Graz University of Technology
// Written by Christoph Feichtenhofer [feichtenhofer@tugraz.at]
// Please email me if you find bugs
// ------------------------------------------------------------------

#ifndef CAFFE_BOX_TRANSFORM_LAYERS_HPP_
#define CAFFE_BOX_TRANSFORM_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    template <typename Dtype>
    class BoxTransformLayer : public Layer<Dtype> {
    public:
        explicit BoxTransformLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "BoxTransform"; }

        virtual inline int MinBottomBlobs() const { return 2; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
            for (int i = 0; i < propagate_down.size(); ++i) {
                if (propagate_down[i]) { NOT_IMPLEMENTED; }
            }
        }

    };

}  // namespace caffe

#endif  // CAFFE_BOX_TRANSFORM_LAYERS_HPP_