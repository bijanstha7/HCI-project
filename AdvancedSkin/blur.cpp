// Generate shader code for Gaussian blur filters
//
// Author: Simon Green
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <string>
#include <sstream>
#include <math.h>
#include <GL/glew.h>
#include "nvMath.h"
#include "nvShaderUtils.h"

//==============================================================================
// 1d Gaussian distribution, s is standard deviation
//==============================================================================
float gaussian(float x, float s)
{
    return expf(-x*x/(2.0f*s*s)) / (s*sqrtf(2.0f*NV_PI));
}

//==============================================================================
// generate array of weights for Gaussian blur
//==============================================================================
float *
generateGaussianWeights(float s, int &width)
{
    width = (int) floor(3.0f*s)-1;
    int size = width*2+1;
    float *weight = new float [size];

    float sum = 0.0;
    int x;
    for(x=0; x<size; x++) {
        weight[x] = gaussian((float) x-width, s);
        sum += weight[x];
    }

    for(x=0; x<size; x++) {
        weight[x] /= sum;
    }
    return weight;
}

//==============================================================================
//
//==============================================================================
float *
generateTriangleWeights(int width)
{
    float *weights = new float [width];
    float sum = 0.0f;
    for(int i=0; i<width; i++) {
        float t = i / (float) (width-1);
        weights[i] = 1.0f - abs(t-0.5f)*2.0f;
        sum += weights[i];
    }
    for(int i=0; i<width; i++) {
        weights[i] /= sum;
    }
    return weights;
}

//==============================================================================
// generate fragment program code for separable convolution
//==============================================================================
GLuint generate1DConvolutionFP(float *weights, int width, bool vertical, bool tex2D, int img_width, int img_height)
{
    std::ostringstream ost;
    ost <<
        "!!ARBfp1.0\n"
        "OPTION NV_fragment_program2;\n"
        "SHORT TEMP H1, H2;\n"
        "TEMP R0;\n";

    int nsamples = 2*width+1;

    for(int i=0; i<nsamples; i++) {
        float x_offset = 0, y_offset = 0;
        if (vertical) {
            y_offset = (float) i-width;
        } else {
            x_offset = (float) i-width;
        }
        if (tex2D) {
            x_offset = x_offset / img_width;
            y_offset = y_offset / img_height;
        }
        float weight = weights[i];

        ost << "ADD R0, fragment.texcoord[0], {" << x_offset << ", " << y_offset << "};\n";
        if (tex2D) {
            ost << "TEX  H1, R0, texture[0], 2D;\n";
        } else {
            ost << "TEX  H1, R0, texture[0], RECT;\n";
        }

        if (i==-width) {
            ost << "MUL H2, H1, {" << weight << "}.x;\n";
        } else {
            ost << "MAD H2, H1, {" << weight << "}.x, H2;\n";
        }
    }

    ost << 
        "MOVH result.color, H2;\n"
        "END\n";

    return nv::CompileASMShader(GL_FRAGMENT_PROGRAM_ARB, ost.str().c_str());
}

//==============================================================================
//
//==============================================================================
/*
  Generate fragment program code for a separable convolution, taking advantage of linear filtering.
  This requires roughly half the number of texture lookups.

  We want the general convolution:
    a*f(i) + b*f(i+1)
  Linear texture filtering gives us:
    f(x) = (1-alpha)*f(i) + alpha*f(i+1);
  It turns out by using the correct weight and offset we can use a linear lookup to achieve this:
    (a+b) * f(i + b/(a+b))
  as long as 0 <= b/(a+b) <= 1
*/
GLuint generate1DConvolutionFP_filter(float *weights, int width, bool vertical, bool tex2D, int img_width, int img_height)
{
    // calculate new set of weights and offsets
    int nsamples = 2*width+1;
    int nsamples2 = (int) ceilf(nsamples/2.0f);
    float *weights2 = new float [nsamples2];
    float *offsets = new float [nsamples2];

    for(int i=0; i<nsamples2; i++) {
        float a = weights[i*2];
        float b;
        if (i*2+1 > nsamples-1)
            b = 0;
        else
            b = weights[i*2+1];
        weights2[i] = a + b;
        offsets[i] = b / (a + b);
    }

    std::ostringstream ost;
    ost <<
        "!!ARBfp1.0\n"
        "OPTION NV_fragment_program2;\n"
        "SHORT TEMP H1, H2;\n"
        "TEMP R0;\n";

    for(int i=0; i<nsamples2; i++) {
        float x_offset = 0, y_offset = 0;
        if (vertical) {
            y_offset = (i*2)-width+offsets[i];
        } else {
            x_offset = (i*2)-width+offsets[i];
        }
        if (tex2D) {
            x_offset = x_offset / img_width;
            y_offset = y_offset / img_height;
        }
        float weight = weights2[i];

        ost << "ADD R0, fragment.texcoord[0], {" << x_offset << ", " << y_offset << "};\n";
        if (tex2D) {
            ost << "TEX  H1, R0, texture[0], 2D;\n";
        } else {
            ost << "TEX  H1, R0, texture[0], RECT;\n";
        }

        if (i==-width) {
            ost << "MUL H2, H1, {" << weight << "}.x;\n";
        } else {
            ost << "MAD H2, H1, {" << weight << "}.x, H2;\n";
        }
    }

    ost << 
        "MOV result.color, H2;\n"
        "END\n";

    delete [] weights2;
    delete [] offsets;

    return nv::CompileASMShader(GL_FRAGMENT_PROGRAM_ARB, ost.str().c_str());
}
