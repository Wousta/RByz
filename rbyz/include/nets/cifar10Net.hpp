#pragma once

#include "nnet.hpp"

class Cifar10NetImpl : public NNet {
public:
  Cifar10NetImpl()
      : // First conv block
        conv1(torch::nn::Conv2dOptions(3, 32, 3).padding(1)),
        bn1(torch::nn::BatchNorm2dOptions(32)),
        conv2(torch::nn::Conv2dOptions(32, 32, 3).padding(1)),
        bn2(torch::nn::BatchNorm2dOptions(32)),

        // Second conv block
        conv3(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
        bn3(torch::nn::BatchNorm2dOptions(64)),
        conv4(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        bn4(torch::nn::BatchNorm2dOptions(64)),

        // Third conv block
        conv5(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        bn5(torch::nn::BatchNorm2dOptions(128)),
        conv6(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
        bn6(torch::nn::BatchNorm2dOptions(128)),

        // Pooling and dropout
        pool(torch::nn::MaxPool2dOptions({2, 2})),
        dropout(torch::nn::DropoutOptions(0.5)),

        // Fully connected layers
        fc1(torch::nn::LinearOptions(128 * 4 * 4, 512)),
        fc2(torch::nn::LinearOptions(512, 10)) {

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    register_module("conv4", conv4);
    register_module("bn4", bn4);
    register_module("conv5", conv5);
    register_module("bn5", bn5);
    register_module("conv6", conv6);
    register_module("bn6", bn6);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("dropout", dropout);
  }

  torch::Tensor forward(torch::Tensor x) override {
    // First conv block: 32x32x3 -> 16x16x32
    auto out = torch::relu(bn1->forward(conv1->forward(x)));
    out = torch::relu(bn2->forward(conv2->forward(out)));
    out = pool->forward(out);

    // Second conv block: 16x16x32 -> 8x8x64
    out = torch::relu(bn3->forward(conv3->forward(out)));
    out = torch::relu(bn4->forward(conv4->forward(out)));
    out = pool->forward(out);

    // Third conv block: 8x8x64 -> 4x4x128
    out = torch::relu(bn5->forward(conv5->forward(out)));
    out = torch::relu(bn6->forward(conv6->forward(out)));
    out = pool->forward(out);

    // Flatten and fully connected layers
    out = out.view({out.size(0), -1});
    out = dropout->forward(torch::relu(fc1->forward(out)));
    out = fc2->forward(out);

    return out;
  }

private:
  // Convolutional layers
  torch::nn::Conv2d conv1, conv2, conv3, conv4, conv5, conv6;

  // Batch normalization layers
  torch::nn::BatchNorm2d bn1, bn2, bn3, bn4, bn5, bn6;

  // Pooling and regularization
  torch::nn::MaxPool2d pool;
  torch::nn::Dropout dropout;

  // Fully connected layers
  torch::nn::Linear fc1, fc2;
};

TORCH_MODULE(Cifar10Net);