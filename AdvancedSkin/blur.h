// interface for the generation of blur shaders
//
// Author: Simon Green
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.

float *generateGaussianWeights(float s, int &n);
float *generateTriangleWeights(int width);

GLuint generate1DConvolutionFP(float *weights, int n, bool vertical, bool tex2D, int img_width, int img_height);
GLuint generate1DConvolutionFP_filter(float *weights, int width, bool vertical, bool tex2D, int img_width, int img_height);
GLuint generateLineFilterFP(int samples, float *weights, float length, float angle, int img_width, int img_height);
