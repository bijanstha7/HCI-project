// interface for cubemap tools
//
// Author: Simon Green
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.

#include "nvMath.h"

nv::vec3f getCubeMapVector(int face, int size, int x, int y);
void indexCubeMap(nv::vec3f d, int &face, float &s, float &t);
GLuint createCubemapTextureFromFunc(int size, GLuint format, nv::vec3f(*func)(int face, float s, float t));