// Cubemap utility code
//
// Author: Simon Green
// Email: sdkfeedback@nvidia.com
//
// Copyright (c) NVIDIA Corporation. All rights reserved.

#include <GL/glew.h>
#include "cubeMap.h"

using namespace nv;

//==============================================================================
// get cubemap direction vector for given face and pixel coordinate
//==============================================================================
vec3f getCubeMapVector(int face, int size, int x, int y)
{
    float s, t, sc, tc;
    vec3f v;

    s = ((float) x + 0.5f) / (float) size;
    t = ((float) y + 0.5f) / (float) size;
    sc = s*2.0f - 1.0f;
    tc = t*2.0f - 1.0f;

    switch (face) {
    case 0: // CUBE_POS_X
        v.x = 1.0;
        v.y = -tc;
        v.z = -sc;
        break;
    case 1: // CUBE_NEG_X
        v.x = -1.0;
        v.y = -tc;
        v.z = sc;
        break;
    case 2: // CUBE_POS_Y
        v.x = sc;
        v.y = 1.0;
        v.z = tc;
        break;
    case 3: // CUBE_NEG_Y
        v.x = sc;
        v.y = -1.0;
        v.z = -tc;
        break;
    case 4: // CUBE_POS_Z
        v.x = sc;
        v.y = -tc;
        v.z = 1.0;
        break;
    case 5: // CUBE_NEG_Z
        v.x = -sc;
        v.y = -tc;
        v.z = -1.0;
        break;
    }

    return normalize(v);
}

//==============================================================================
// convert cubemap direction vector to face index and (s, t) coordinates
//==============================================================================
void indexCubeMap(vec3f d, int &face, float &s, float &t)
{
    vec3f absd;
    float sc, tc, ma;

    absd.x = fabs(d.x);
    absd.y = fabs(d.y);
    absd.z = fabs(d.z);

    face = 0;

    if ((absd.x >= absd.y) && (absd.x >= absd.z)) {
        if (d.x > 0.0) {
            face = 0;
            sc = -d.z; tc = -d.y; ma = absd.x;
        } else {
            face = 1;
            sc = d.z; tc = -d.y; ma = absd.x;
        }
    }

    if ((absd.y >= absd.x) && (absd.y >= absd.z)) {
        if (d.y > 0.0) {
            face = 2;
            sc = d.x; tc = d.z; ma = absd.y;
        } else {
            face = 3;
            sc = d.x; tc = -d.z; ma = absd.y;
        }
    }

    if ((absd.z >= absd.x) && (absd.z >= absd.y)) {
        if (d.z > 0.0) {
            face = 4;
            sc = d.x; tc = -d.y; ma = absd.z;
        } else {
            face = 5;
            sc = -d.x; tc = -d.y; ma = absd.z;
        }
    }

    if (ma == 0.0) {
        s = 0.0;
        t = 0.0;
    } else {
        s = ((sc / ma) + 1.0f) / 2.0f;
        t = ((tc / ma) + 1.0f) / 2.0f;
    }
}

//==============================================================================
// create cubemap procedurally from function
//==============================================================================
GLuint createCubemapTextureFromFunc(int size, GLuint format, vec3f(*func)(int face, float s, float t))
{
    // generate texture object
    GLuint tex; 
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

    // load face images
    vec3f *data = new vec3f [size*size];

    for(int i=0; i<6; i++) {

        // generate face
        vec3f *ptr = data;
        for(int y=0; y<size; y++) {
            for(int x=0; x<size; x++) {
                float s = ((float) x + 0.5f) / (float) size;
                float t = ((float) y + 0.5f) / (float) size;
                s = s*2.0f - 1.0f;
                t = t*2.0f - 1.0f;
                *ptr++ = (*func)(i, s, t);
            }
        }

        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
                     format, size, size, 0, 
                     GL_RGB, GL_FLOAT, data);
    }

    delete data;

    return tex;
}
