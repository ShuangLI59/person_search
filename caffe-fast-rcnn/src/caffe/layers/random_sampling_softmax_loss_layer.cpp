#include <algorithm>
#include <functional>
#include <utility>
#include <cfloat>
#include <vector>

#include "caffe/layers/random_sampling_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void RandomSamplingSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  rss_num_ = this->layer_param_.rss_loss_param().random_sampling_num();
  rss_policy_ = this->layer_param_.rss_loss_param().random_sampling_policy();
  if (rss_policy_ != "random" && rss_policy_ != "topk") {
    LOG(FATAL) << "Cannot recognize the random_sampling_policy parameter";
  }

  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&rss_bottom_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&rss_top_);
  vector<int> rss_dims = bottom[0]->shape();
  rss_dims[softmax_axis_] = rss_num_;
  rss_bottom_.Reshape(rss_dims);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void RandomSamplingSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  channels_ = bottom[0]->shape(softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  CHECK_GT(channels_, rss_num_)
      << "Number of channels must be greater than random sampling number.";
  vector<int> rss_dims = bottom[0]->shape();
  rss_dims[softmax_axis_] = rss_num_;
  rss_bottom_.Reshape(rss_dims);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  rss_index_.ReshapeLike(rss_bottom_);
  index_vec_.resize(channels_);
  for (int i = 0; i < channels_; ++i) {
    index_vec_[i] = i;
  }
}

template <typename Dtype>
Dtype RandomSamplingSoftmaxLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void RandomSamplingSoftmaxLossLayer<Dtype>::random_sampling(
    const vector<Blob<Dtype>*>& bottom) {
  // Randomly choose the bottom indices
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* rss_index_data = rss_index_.mutable_cpu_data();
  const int dim = channels_ * inner_num_;
  const int dim_rss = rss_num_ * inner_num_;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        // Just put 0 to rss_num_ - 1
        for (int k = 0; k < rss_num_; ++k) {
          rss_index_data[i * dim_rss + k * inner_num_ + j] = (Dtype)k;
        }
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, rss_top_.shape(softmax_axis_));
      // Here we hard coded two rules:
      //   1. Put the label_value to the first
      //   2. Put the channels_-1 (background class) to the second
      int k = 0;
      rss_index_data[i * dim_rss + k * inner_num_ + j] = (Dtype)label_value;
      k++;
      const int bg_label = channels_ - 1;
      if (label_value != bg_label) {
        rss_index_data[i * dim_rss + k * inner_num_ + j] = (Dtype)bg_label;
        k++;
      }
      if (rss_policy_ == "random") {
        shuffle(index_vec_.begin(), index_vec_.end());
        for (int x = 0; x < index_vec_.size(); ++x) {
          if (k >= rss_num_) break;
          const int index = index_vec_[x];
          if (index != label_value && index != bg_label) {
            rss_index_data[i * dim_rss + k * inner_num_ + j] = (Dtype)index;
            k++;
          }
        }
      } else {
        vector<std::pair<Dtype, int> > score_index_vector;
        for (int c = 0; c < channels_; ++c) {
          score_index_vector.push_back(std::make_pair(
              bottom_data[i * dim + c * inner_num_ + j], c));
        }
        std::partial_sort(score_index_vector.begin(),
                          score_index_vector.begin() + rss_num_,
                          score_index_vector.end(),
                          std::greater<std::pair<Dtype, int> >());
        for (int x = 0; x < rss_num_; ++x) {
          if (k >= rss_num_) break;
          const int index = score_index_vector[x].second;
          if (index == label_value || index == bg_label) continue;
          rss_index_data[i * dim_rss + k * inner_num_ + j] = (Dtype)index;
          k++;
        }
      }
    }
  }
  // Copy the data
  Dtype* rss_bottom_data = rss_bottom_.mutable_cpu_data();
  int index = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int k = 0; k < rss_num_; ++k) {
      for (int j = 0; j < inner_num_; ++j) {
        const int c = static_cast<int>(rss_index_data[index]);
        rss_bottom_data[index++] = bottom_data[i * dim + c * inner_num_ + j];
      }
    }
  }
}

template <typename Dtype>
void RandomSamplingSoftmaxLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  random_sampling(bottom);
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob = rss_top_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = rss_top_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      // Here we hard coded that when fill up the random sampling index,
      // we always put the target label to the first element (channel = 0).
      loss -= log(std::max(prob[i * dim + j], Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
}

template <typename Dtype>
void RandomSamplingSoftmaxLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), (Dtype)0, bottom_diff);
    const Dtype* prob = rss_top_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* rss_index_data = rss_index_.cpu_data();
    int dim = channels_ * inner_num_;
    int dim_rss = rss_num_ * inner_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        // Project back the gradients
        for (int k = 0; k < rss_num_; ++k) {
          const int index = i * dim_rss + k * inner_num_ + j;
          const int c = static_cast<int>(rss_index_data[index]);
          bottom_diff[i * dim + c * inner_num_ + j] = prob[index];
        }
        bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
        ++count;
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_CLASS(RandomSamplingSoftmaxLossLayer);
REGISTER_LAYER_CLASS(RandomSamplingSoftmaxLoss);

}  // namespace caffe
