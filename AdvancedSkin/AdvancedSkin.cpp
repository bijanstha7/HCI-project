
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <map>
#include <math.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <Cg/cgGL.h>

#include <nvImage.h>
#include <nvMath.h>
#include <nvGlutManipulators.h>
#define NV_REPORT_COMPILE_ERRORS
#include <nvShaderUtils.h>
#include <nvModel.h>
#include <nvGlutWidgets.h>

#include "RenderTextureFBO.h"
#include "blur.h"
#include "cubeMap.h"
#include "appPath.h"
#include "framerate.h"

using std::map;
using namespace nv;

// path utility
appPath g_AppPath;

GLuint hdr_tex;

// face model
char *model_filename = "../models/james_hi.obj";
nv::Model *model = 0;

// texture image file name
char *skin_filename		= "../textures/james.png";
char *normal_filename	= "../textures/james_normal.png";
char *spec_filename		= "../textures/skin_spec.dds";			// need to make my own
char *stretch_filename	= "../textures/skin_stretch.dds";		// need to make my own
char *rho_d_filename	= "../textures/rho_d.png";
char *cube_filename		= "../textures/cube.hdr";
char *cubediff_filename	= "../textures/cube_diff.hdr";
char *cubeconv_filename = "../textures/cube_conv.hdr";

// texture image object
nv::Image img_skin, img_normal, img_spec, img_stretch;
nv::Image img_rho, img_cube, img_cubediff, img_cubeconv;

// texture handle
GLuint skin_tex, normal_tex, spec_tex, stretch_tex;
GLuint rho_tex, cube_tex, cubediff_tex, cubeconv_tex;

// frame buffer object: shadow map & gaussian-sum
#define BUFFERSIZE 1024
#define IRR_BUFFERS 5
RenderTexture *first_buffer = 0;			// the first rendering: non blurred
RenderTexture *irr_buffer[IRR_BUFFERS];		// for gaussian-sum: store result
RenderTexture *temp_buffer = 0;				//     intermediate storage between u,v blur
RenderTexture *stretch_buffer[IRR_BUFFERS];	// stretch texture blur storage
RenderTexture *stretch_temp=0;				// stretch texture blur storage
RenderTexture *shadow_buffer = 0;			// shadow buffer

// shader programs
CGprogram flat_vprog, flat_fprog;			// texture space model rendering
CGprogram skin_vprog, skin_fprog;			// final skin rendering
CGprogram final_vprog, final_fprog;			// final skin rendering
CGprogram stretch_vprog, stretch_fprog;		// stretch uv map generator
CGprogram conv_vprog, convU_fprog, convV_fprog;
CGprogram convStretch_vprog, convStretchU_fprog, convStretchV_fprog;
CGparameter flat_model_param;
CGparameter stretch_model_param;
CGparameter stretch_scale_param;
CGparameter convolveU_width_param, convolveV_width_param;
CGparameter convStretchU_width_param, convStretchV_width_param;
CGparameter final_model1_param, final_viewproj_param, final_viewtarget_param;
CGparameter final_model2_param, final_litcolor_param, final_shadowcolor_param;
CGparameter final_lightpos_param, final_eye_pos, final_diffmix_param;
CGparameter final_gauss1_param, final_gauss2_param, final_gauss3_param;
CGparameter final_gauss4_param, final_gauss5_param, final_gauss6_param;

// number of debugging viewport rects
#define DEBUG_RECT_COLS 8
#define DEBUG_RECT_ROWS 5
int debug_mode = 0;

// parameters
float exposure = 1.0;
float aniso = 2.0;
float blur_width = 3.0; // width of blur kernel
float blur_amount = 0.5;
float effect_amount = 0.2;
float convolution_scale[5] = { 15.0f, 15.0f, 15.0f, 15.0f, 15.0f};

struct frustum
{
	float neard;
	float fard;
	float fov;
	float ratio;
	nv::vec3f point[8];
};

int depth_size = 1024;
float shad_cpm[16];
frustum f;

bool  bMenu = false;

float lpos[4] = {1.8, 0.9, 1.8, 0.0};
float cam_pos[3] = {0.0f, 0.0f, -4.0f};
float cam_view[3] = {0.0f, 0.0f, 2.0f};
#define FAR_DIST 100.0f

enum UIOption {
    OPTION_WIREFRAME,
    OPTION_GLOW,
	OPTION_DRAW_DEBUG,
	OPTION_MOVE_LIGHT,
    OPTION_COUNT,
};
bool options[OPTION_COUNT];
map<char, UIOption> optionKeyMap;

nv::GlutUIContext ui;

CGcontext context;
CGprofile cg_vprofile, cg_fprofile;

// shaders
CGprogram object_vprog, object_fprog;
CGparameter model_param1, model_param2;

CGprogram skybox_vprog, skybox_fprog;
CGprogram downsample_vprog, downsample_fprog;
CGprogram downsample4_vprog, downsample4_fprog;
CGprogram tonemap_fprog;
CGparameter blurAmount_param, effectAmount_param, windowSize_param, exposure_param;


CGparameter twoTexelSize_param;
GLuint blurh_fprog = 0, blurv_fprog = 0;

bool keydown[256];

int win_w = 768, win_h = 768;

// AA configuration info
struct aaInfo {
    int samples;
    int coverage_samples;
};

const aaInfo aaModes[] = { {0,0}, {2,2}, {4,4}, {4,8}, {8,8}, {8,16}};
const char* aaOptions[] = { "None", "2x", "4x", "8xCSAA", "8x", "16xCSAA"};
int currentAAMode = 2;

nv::GlutExamine camera, object;

#define DOWNSAMPLE_BUFFERS 2
#define BLUR_BUFFERS 2
RenderTexture *scene_buffer = 0, *downsample_buffer[DOWNSAMPLE_BUFFERS];
RenderTexture *blur_buffer[BLUR_BUFFERS];
RenderTexture *ms_buffer = 0;

bool fullscreen = 0;

GLuint buffer_format = GL_RGBA16F_ARB;
GLuint texture_format = GL_RGBA16F_ARB;



int have_CSAA = false;

//==============================================================================
// forward declaration
//==============================================================================
void initGL();
void setOrthoProjection(int w, int h);
void setPerspectiveProjection(int w, int h);
void initBlurCode(float blur_width);
void createBuffers(GLenum format);
void convolution(RenderTexture *src, RenderTexture *dest, int itr);
void convolutionStretch(RenderTexture *src, RenderTexture *dest, int itr);

//==============================================================================
// draw a quad with texture coordinate for texture rectangle
//==============================================================================
void drawQuad(int w, int h)
{
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0);
    glVertex2f(0.0, h);
    glTexCoord2f(1.0, 1.0);
    glVertex2f(w, h);
    glTexCoord2f(1.0, 0.0);
    glVertex2f(w, 0.0);
    glTexCoord2f(0.0, 0.0);
    glVertex2f(0.0, 0.0);
    glEnd();
}

//==============================================================================
// draw a quad on viewport: useful to debug
// shadow, blur0~4: total 6 slots
//==============================================================================
void drawViewport(RenderTexture *src)
{
	setOrthoProjection(win_w, win_h);
	glActiveTexture(GL_TEXTURE0);
	src->Bind();

	glDisable(GL_DEPTH_TEST);
	
	drawQuad(win_w, win_h);

	glEnable(GL_DEPTH_TEST);

	src->Release();
}

//==============================================================================
// draw a quad on viewport: useful to debug
// shadow, blur0~4: total 6 slots
//==============================================================================
void drawViewRect(RenderTexture *src, int col, int row=0)
{
	if (col < 0 || col >= DEBUG_RECT_COLS)
		return;
	if (row < 0 || row >= DEBUG_RECT_ROWS)
		return;
	
	setOrthoProjection(win_w, win_h);
	glActiveTexture(GL_TEXTURE0);
	src->Bind();

	float size;
	float offset = 5.0f;
	size = (win_w - (DEBUG_RECT_COLS + 1.0f) * offset) / (float)DEBUG_RECT_COLS;
	
	// rect coordinate
	float x0,x1,y0,y1;
	x0 = size*col + offset * (col+1);
	x1 = x0 + size;
	y0 = size*row + offset * (row+1);
	y1 = y0 + size;	

	glDisable(GL_DEPTH_TEST);
	
	glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0);
    glVertex2f(x0, y1);
    glTexCoord2f(1.0, 1.0);
    glVertex2f(x1, y1);
    glTexCoord2f(1.0, 0.0);
    glVertex2f(x1, y0);
    glTexCoord2f(0.0, 0.0);
    glVertex2f(x0, y0);
    glEnd();

	glEnable(GL_DEPTH_TEST);

	src->Release();
}

//==============================================================================
// draw a quad on viewport: useful to debug
// shadow, blur0~4: total 6 slots
//==============================================================================
void drawViewRect(GLuint tex, int col, int row=0)
{
	if (col < 0 || col >= DEBUG_RECT_COLS)
		return;
	if (row < 0 || row >= DEBUG_RECT_ROWS)
		return;

	setOrthoProjection(win_w, win_h);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);

	float size;
	float offset = 5.0f;
	size = (win_w - (DEBUG_RECT_COLS + 1.0f) * offset) / (float)DEBUG_RECT_COLS;
	
	// rect coordinate
	float x0,x1,y0,y1;
	x0 = size*col + offset * (col+1);
	x1 = x0 + size;
	y0 = size*row + offset * (row+1);
	y1 = y0 + size;	

	glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 1.0);
    glVertex2f(x0, y1);
    glTexCoord2f(1.0, 1.0);
    glVertex2f(x1, y1);
    glTexCoord2f(1.0, 0.0);
    glVertex2f(x1, y0);
    glTexCoord2f(0.0, 0.0);
    glVertex2f(x0, y0);
    glEnd();

	glEnable(GL_DEPTH_TEST);

	glBindTexture(GL_TEXTURE_2D, 0);

}


//==============================================================================
//
//==============================================================================
GLuint createCubemapTexture(nv::Image &img, GLint internalformat)
{
    GLuint tex; 
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

    // load face data
    for(int i=0; i<6; i++) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
                     internalformat, img.getWidth(), img.getHeight(), 0, 
                     GL_RGB, GL_FLOAT, img.getLevel(0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i));
    }

    return tex;
}

//==============================================================================
//
//==============================================================================
GLuint create2DTexture(nv::Image &img, GLint internalformat)
{
    GLuint tex; 
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri( GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

    // load face data
	glTexImage2D( GL_TEXTURE_2D, 0, img.getInternalFormat(), img.getWidth(), img.getHeight(), 0, img.getFormat(), img.getType(), img.getLevel(0));
    
	return tex;
}

//==============================================================================
// function for creating procedural test cubemap
//==============================================================================
vec3f testFunc(int face, float s, float t)
{
    vec3f col[] = {
        vec3f(1.0f, 0.0f, 0.0f),
        vec3f(0.0f, 1.0f, 1.0f),
        vec3f(0.0f, 1.0f, 0.0f),
        vec3f(1.0f, 0.0f, 1.0f),
        vec3f(0.0f, 0.0f, 1.0f),
        vec3f(1.0f, 1.0f, 0.0f)
    };
    float i = sqrt(s*s + t*t);
    i = powf(2.0f, -(i+1.0f)*16.0f);
    return col[face]*i;
}

//==============================================================================
// draw cubemap background
//==============================================================================
void drawSkyBox()
{
    glDisable(GL_CULL_FACE);

	cgGLBindProgram(skybox_vprog);
    cgGLEnableProfile(cg_vprofile);
    cgGLBindProgram(skybox_fprog);
    cgGLEnableProfile(cg_fprofile);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, hdr_tex);
    glTexEnvf(GL_TEXTURE_FILTER_CONTROL_EXT, GL_TEXTURE_LOD_BIAS_EXT, 0.0f);

    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
	nv::matrix4f m = camera.getTransform();
    nv::matrix4f mi = inverse(m);
    glMultMatrixf((float *) &mi);
    
	glutSolidCube(1.0);
    glPopMatrix();

    glEnable(GL_DEPTH_TEST);
    cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

	glEnable(GL_CULL_FACE);

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

//==============================================================================
//
//==============================================================================
void drawModel( const nv::Model *model)
{
    glVertexPointer( model->getPositionSize(), GL_FLOAT, model->getCompiledVertexSize() * sizeof(float), model->getCompiledVertices());
    glNormalPointer( GL_FLOAT, model->getCompiledVertexSize() * sizeof(float), model->getCompiledVertices() + model->getCompiledNormalOffset());
	glTexCoordPointer (model->getTexCoordSize(), GL_FLOAT, model->getCompiledVertexSize() * sizeof(float), model->getCompiledVertices() + model->getCompiledTexCoordOffset());
    
	glEnableClientState( GL_VERTEX_ARRAY);
    glEnableClientState( GL_NORMAL_ARRAY);
	glEnableClientState( GL_TEXTURE_COORD_ARRAY);

    glDrawElements( GL_TRIANGLES, model->getCompiledIndexCount( nv::Model::eptTriangles), GL_UNSIGNED_INT, model->getCompiledIndices( nv::Model::eptTriangles));

    glDisableClientState( GL_VERTEX_ARRAY);
    glDisableClientState( GL_NORMAL_ARRAY);
	glDisableClientState( GL_TEXTURE_COORD_ARRAY);
}

//==============================================================================
// read from float texture, apply tone mapping, render to regular 8/8/8 display
//==============================================================================
void toneMappingPass()
{
    // render to window
    scene_buffer->Deactivate();
    setOrthoProjection(win_w, win_h);

    // bind textures
    glActiveTexture(GL_TEXTURE0);
    scene_buffer->Bind();

    glActiveTexture(GL_TEXTURE1);
    blur_buffer[0]->Bind();

    cgGLBindProgram(tonemap_fprog);
    cgGLEnableProfile(cg_fprofile);

    if (options[OPTION_GLOW]) {
		cgGLSetParameter1f(blurAmount_param, blur_amount);
		cgGLSetParameter1f(effectAmount_param, effect_amount);
	} else {
		cgGLSetParameter1f(blurAmount_param, 0.0f);
		cgGLSetParameter1f(effectAmount_param, 0.0f);
	}

	cgGLSetParameter4f(windowSize_param, 2.0/win_w, 2.0/win_h, -1.0, -1.0);
    cgGLSetParameter1f(exposure_param, exposure);

    glDisable(GL_DEPTH_TEST);
    drawQuad(win_w, win_h);
	glEnable(GL_DEPTH_TEST);

	blur_buffer[0]->Release();
	scene_buffer->Release();

    cgGLDisableProfile(cg_fprofile);

    glutReportErrors();
}

//==============================================================================
//
//==============================================================================
void makeShadowMap()
{
	float shad_modelview[16];

	glDisable(GL_TEXTURE_2D);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	gluLookAt( lpos[0], lpos[1], lpos[2],	// Look from the light's position
               0.0f, 0.0f, 0.0f,							// Towards the teapot's position
               0.0f, 1.0f, 0.0f );
	glGetFloatv(GL_MODELVIEW_MATRIX, shad_modelview);

	// redirect rendering to the depth texture
	shadow_buffer->Activate();
	
	// store the screen viewport
	glPushAttrib(GL_VIEWPORT_BIT);
	
	// and render only to the shadowmap
	glViewport(0, 0, depth_size, depth_size);
	
	// draw all faces since our terrain is not closed.
	glDisable(GL_CULL_FACE);
	
	// clear the depth texture from last time
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw the scene
	glCallList(1);
	
	glMatrixMode(GL_PROJECTION);
	
	// store the product of all shadow matries for later
	glMultMatrixf(shad_modelview);
	glGetFloatv(GL_PROJECTION_MATRIX, shad_cpm);

	// revert to normal back face culling as used for rendering
	glEnable(GL_CULL_FACE);

	glPopAttrib(); 
	
	shadow_buffer->Deactivate();
	
	glEnable(GL_TEXTURE_2D);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

}

//==============================================================================
//
//==============================================================================
void makeStretchMap()
{

	glDisable(GL_TEXTURE_2D);

	// redirect rendering to the depth texture
	stretch_buffer[0]->Activate();
	
	// store the screen viewport
	glPushAttrib(GL_VIEWPORT_BIT);
	
	// and render only to the shadowmap
	glViewport(0, 0, BUFFERSIZE, BUFFERSIZE);
	
	// draw all faces
	glDisable(GL_CULL_FACE);
	
	// clear the depth texture from last time
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	cgGLBindProgram(stretch_vprog);
    cgGLEnableProfile(cg_vprofile);
	cgGLBindProgram(stretch_fprog);
    cgGLEnableProfile(cg_fprofile);
	cgGLSetMatrixParameterfc(stretch_model_param, object.getTransform().get_value());
	cgGLSetParameter1f(stretch_scale_param, 0.001f);

	// draw the scene
	glCallList(1);
	
	// revert to normal back face culling as used for rendering
	glEnable(GL_CULL_FACE);

	glPopAttrib(); 
	
	stretch_buffer[0]->Deactivate();
	
	glEnable(GL_TEXTURE_2D);

	cgGLDisableProfile(cg_fprofile);
	cgGLDisableProfile(cg_vprofile);

	////////////////////////////////////////////////
	// make convolved stretch map
	glDisable(GL_CULL_FACE);
	convolutionStretch(stretch_buffer[0], stretch_buffer[1], 0);
	convolutionStretch(stretch_buffer[1], stretch_buffer[2], 1);
	convolutionStretch(stretch_buffer[2], stretch_buffer[3], 2);
	convolutionStretch(stretch_buffer[3], stretch_buffer[4], 3);
	
	glutReportErrors();

}

//==============================================================================
// render scene to float pbuffer
//==============================================================================
void renderFlatModel()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLightfv(GL_LIGHT0, GL_POSITION, lpos);

	first_buffer->Activate();
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, BUFFERSIZE, BUFFERSIZE);

	// draw object
	cgGLBindProgram(flat_vprog);
    cgGLEnableProfile(cg_vprofile);

	cgGLBindProgram(flat_fprog);
    cgGLEnableProfile(cg_fprofile);
	
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
    glLoadIdentity();
	camera.applyTransform();
	
	cgGLSetMatrixParameterfc(flat_model_param, object.getTransform().get_value());

    glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, skin_tex);

    glEnable(GL_CULL_FACE);

	glEnable(GL_DEPTH_TEST);

	glCallList(1);

    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_MULTISAMPLE);
    cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glPopAttrib();
	first_buffer->Deactivate();

	glBindTexture(GL_TEXTURE_2D, 0);

    glutReportErrors();
}

//==============================================================================
// render subsurface scattering head
//==============================================================================
void renderSSS()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if ( options[OPTION_WIREFRAME])
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glLightfv(GL_LIGHT0, GL_POSITION, lpos);


	// draw object
	cgGLBindProgram(final_vprog);
    cgGLEnableProfile(cg_vprofile);

	cgGLBindProgram(final_fprog);
    cgGLEnableProfile(cg_fprofile);
	
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
	camera.applyTransform();

/*	uniform sampler2D irrad1Tex			: TEXUNIT0,
	uniform sampler2D irrad2Tex			: TEXUNIT1,
	uniform sampler2D irrad3Tex			: TEXUNIT2,
	uniform sampler2D irrad4Tex			: TEXUNIT3,
	uniform sampler2D irrad5Tex			: TEXUNIT4,
	uniform sampler2D irrad6Tex			: TEXUNIT5,

	uniform sampler2D normalTex			: TEXUNIT6,
	uniform sampler2D TSMTex			: TEXUNIT7,
	uniform sampler2D rhodTex			: TEXUNIT8,
	uniform sampler2D stretch32Tex		: TEXUNIT9,
	uniform sampler2D diffuseColorTex	: TEXUNIT10,
	uniform sampler2D specTex			: TEXUNIT11,
*/
	glActiveTexture(GL_TEXTURE0);
	first_buffer->Bind();
	glActiveTexture(GL_TEXTURE1);
	irr_buffer[0]->Bind();
	glActiveTexture(GL_TEXTURE2);
	irr_buffer[1]->Bind();
    glActiveTexture(GL_TEXTURE3);
	irr_buffer[2]->Bind();
	glActiveTexture(GL_TEXTURE4);
	irr_buffer[3]->Bind();
    glActiveTexture(GL_TEXTURE5);
	irr_buffer[4]->Bind();

	glActiveTexture(GL_TEXTURE6);
	glBindTexture(GL_TEXTURE_2D, normal_tex);

	glActiveTexture(GL_TEXTURE7);	// TSM
	shadow_buffer->Bind();

	glActiveTexture(GL_TEXTURE8);
	glBindTexture(GL_TEXTURE_2D, rho_tex);

	glActiveTexture(GL_TEXTURE9);
	stretch_buffer[4]->Bind();

	glActiveTexture(GL_TEXTURE10);
	glBindTexture(GL_TEXTURE_2D, skin_tex);

	glActiveTexture(GL_TEXTURE11);
	glBindTexture(GL_TEXTURE_2D, spec_tex);

	// parameter setting
	float mat[16];
	cgGLSetMatrixParameterfc(final_model1_param, object.getTransform().get_value());
	glGetFloatv(GL_PROJECTION_MATRIX, mat);
	cgGLSetMatrixParameterfc(final_viewproj_param, mat);
	cgGLSetMatrixParameterfc(final_viewtarget_param, shad_cpm);

	cgGLSetMatrixParameterfc(final_model2_param, object.getTransform().get_value());
	cgGLSetParameter3f(final_litcolor_param, 1.0f, 1.0f, 1.0f);
	cgGLSetParameter3f(final_shadowcolor_param, 0.1f, 0.1f, 0.1f);
	cgGLSetParameter3f(final_lightpos_param, lpos[0], lpos[1], lpos[2]);
	cgGLSetParameter3f(final_eye_pos, 0.0f, 0.0f, -3.0f);
	cgGLSetParameter1f(final_diffmix_param, 0.5f);
	cgGLSetParameter3f(final_gauss1_param, 0.233f, 0.455f, 0.649f);
	cgGLSetParameter3f(final_gauss2_param, 0.1f, 0.336f, 0.344f);
	cgGLSetParameter3f(final_gauss3_param, 0.118f, 0.198f, 0.0f);
	cgGLSetParameter3f(final_gauss4_param, 0.113f, 0.007f, 0.007f);
	cgGLSetParameter3f(final_gauss5_param, 0.358f, 0.004f, 0.0f);
	cgGLSetParameter3f(final_gauss6_param, 0.078f, 0.0f, 0.0f);




	glEnable(GL_CULL_FACE);

	glEnable(GL_DEPTH_TEST);

	glCallList(1);

    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_MULTISAMPLE);
    cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


	// release RTT
	stretch_buffer[4]->Release();
	shadow_buffer->Release();
	irr_buffer[4]->Release();
	irr_buffer[3]->Release();
	irr_buffer[2]->Release();
	irr_buffer[1]->Release();
	irr_buffer[0]->Release();

	glBindTexture(GL_TEXTURE_2D, 0);

    glutReportErrors();
}


//==============================================================================
// render scene to float pbuffer
//==============================================================================
void renderSkin()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if ( options[OPTION_WIREFRAME])
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    else
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glLightfv(GL_LIGHT0, GL_POSITION, lpos);

	// draw object
	cgGLBindProgram(object_vprog);
    cgGLEnableProfile(cg_vprofile);

	cgGLBindProgram(object_fprog);
    cgGLEnableProfile(cg_fprofile);
	
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
	camera.applyTransform();
	
	cgGLSetMatrixParameterfc(model_param1, object.getTransform().get_value());
	cgGLSetMatrixParameterfc(model_param2, object.getTransform().get_value());

    glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, skin_tex);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, normal_tex);

    glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	glCallList(1);

    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_MULTISAMPLE);

	cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glBindTexture(GL_TEXTURE_2D, 0);

    glutReportErrors();
}

//==============================================================================
// Gaussian Convolution
//==============================================================================
void convolution(RenderTexture *src, RenderTexture *dest, int itr)
{
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, BUFFERSIZE, BUFFERSIZE);
	
	// convolution U
	temp_buffer->Activate();
    setOrthoProjection(BUFFERSIZE, BUFFERSIZE);

    cgGLBindProgram(conv_vprog);
    cgGLEnableProfile(cg_vprofile);
    cgGLBindProgram(convU_fprog);
    cgGLEnableProfile(cg_fprofile);
	cgGLSetParameter1f(convolveU_width_param, convolution_scale[itr]);

    glActiveTexture(GL_TEXTURE0);
    src->Bind();
    glActiveTexture(GL_TEXTURE1);
    stretch_buffer[itr]->Bind();

	drawQuad(BUFFERSIZE, BUFFERSIZE);
    
	src->Release();
    
	cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

    temp_buffer->Deactivate();

	// convolution V
	dest->Activate();
    setOrthoProjection(BUFFERSIZE, BUFFERSIZE);

    cgGLBindProgram(conv_vprog);
    cgGLEnableProfile(cg_vprofile);
    cgGLBindProgram(convV_fprog);
    cgGLEnableProfile(cg_fprofile);
	cgGLSetParameter1f(convolveV_width_param, convolution_scale[itr]);

    glActiveTexture(GL_TEXTURE0);
    temp_buffer->Bind();
    glActiveTexture(GL_TEXTURE1);
    stretch_buffer[itr]->Bind();

	drawQuad(BUFFERSIZE, BUFFERSIZE);
    
	temp_buffer->Release();
    
	cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

    dest->Deactivate();

	glBindTexture(GL_TEXTURE_2D, 0);

	glPopAttrib();
}

//==============================================================================
// Gaussian Convolution
//==============================================================================
void convolutionStretch(RenderTexture *src, RenderTexture *dest, int itr)
{
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0, 0, BUFFERSIZE, BUFFERSIZE);
	
	// convolution U
	temp_buffer->Activate();
    setOrthoProjection(BUFFERSIZE, BUFFERSIZE);

    cgGLBindProgram(convStretch_vprog);
    cgGLEnableProfile(cg_vprofile);
    cgGLBindProgram(convStretchU_fprog);
    cgGLEnableProfile(cg_fprofile);
	cgGLSetParameter1f(convStretchU_width_param, convolution_scale[itr]);

    glActiveTexture(GL_TEXTURE0);
    src->Bind();

	drawQuad(BUFFERSIZE, BUFFERSIZE);
    
	src->Release();
    
	cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

    temp_buffer->Deactivate();

	// convolution V
	dest->Activate();

    cgGLBindProgram(convStretch_vprog);
    cgGLEnableProfile(cg_vprofile);
    cgGLBindProgram(convStretchV_fprog);
    cgGLEnableProfile(cg_fprofile);
	cgGLSetParameter1f(convStretchV_width_param, convolution_scale[itr]);

    glActiveTexture(GL_TEXTURE0);
    temp_buffer->Bind();

	drawQuad(BUFFERSIZE, BUFFERSIZE);
    
	temp_buffer->Release();
    
	cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

    dest->Deactivate();

	glBindTexture(GL_TEXTURE_2D, 0);

	glPopAttrib();
}

//==============================================================================
// downsample image 2x in each dimension
//==============================================================================
void downsample(RenderTexture *src, RenderTexture *dest)
{
    dest->Activate();
    setOrthoProjection(dest->GetWidth(), dest->GetHeight());

    cgGLBindProgram(downsample_vprog);
    cgGLEnableProfile(cg_vprofile);
    cgGLBindProgram(downsample_fprog);
    cgGLEnableProfile(cg_fprofile);

    glActiveTexture(GL_TEXTURE0);
    src->Bind();
    
	drawQuad(dest->GetWidth(), dest->GetHeight());
    
	src->Release();
    
	cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

    dest->Deactivate();
}

//==============================================================================
// downsample image 4x in each dimension
//==============================================================================
void downsample4(RenderTexture *src, RenderTexture *dest)
{
    dest->Activate();
    setOrthoProjection(dest->GetWidth(), dest->GetHeight());

    cgGLBindProgram(downsample4_vprog);
    cgGLEnableProfile(cg_vprofile);
    cgGLBindProgram(downsample4_fprog);
    cgGLSetParameter2f(twoTexelSize_param, 2.0 / src->GetWidth(), 2.0 / src->GetWidth());
    cgGLEnableProfile(cg_fprofile);

    glActiveTexture(GL_TEXTURE0);
    src->Bind();
    
	drawQuad(dest->GetWidth(), dest->GetHeight());
    
	src->Release();
    
	cgGLDisableProfile(cg_fprofile);
    cgGLDisableProfile(cg_vprofile);

    dest->Deactivate();
}

//==============================================================================
//  generic full screen processing function
//==============================================================================
void run_pass(GLuint prog, RenderTexture *src, RenderTexture *dest)
{
    dest->Activate();
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, prog);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glActiveTexture(GL_TEXTURE0);
    src->Bind();
    drawQuad(dest->GetWidth(), dest->GetHeight());
    src->Release();
    glDisable(GL_FRAGMENT_PROGRAM_ARB);
    dest->Deactivate();
}

//==============================================================================
//  generic full screen processing function
//==============================================================================
void run_pass(CGprogram prog, RenderTexture *src, RenderTexture *dest)
{
    dest->Activate();
    cgGLBindProgram(prog);
    cgGLEnableProfile(cg_fprofile);
    glActiveTexture(GL_TEXTURE0);
    src->Bind();
    drawQuad(dest->GetWidth(), dest->GetHeight());
    src->Release();
    cgGLDisableProfile(cg_fprofile);
    dest->Deactivate();
}

//==============================================================================
//  function used to apply the gaussian blur
//==============================================================================
void glow(RenderTexture *src, RenderTexture *dest, RenderTexture *temp)
{
    setOrthoProjection(dest->GetWidth(), dest->GetHeight());

    // horizontal blur
    run_pass(blurh_fprog, src, temp);

    // vertical blur
    run_pass(blurv_fprog, temp, dest);
}

//==============================================================================
//
//==============================================================================
inline void updateButtonState( const nv::ButtonState &bs, nv::GlutManipulator &manip, int button) {
    int modMask = 0;

    if (bs.state & nv::ButtonFlags_Alt) modMask |= GLUT_ACTIVE_ALT;
    if (bs.state & nv::ButtonFlags_Shift) modMask |= GLUT_ACTIVE_SHIFT;
    if (bs.state & nv::ButtonFlags_Ctrl) modMask |= GLUT_ACTIVE_CTRL;

    if (bs.state & nv::ButtonFlags_End)
        manip.mouse( button, GLUT_UP, modMask, bs.cursor.x, win_h - bs.cursor.y);
    if (bs.state & nv::ButtonFlags_Begin)
        manip.mouse( button, GLUT_DOWN, modMask, bs.cursor.x, win_h - bs.cursor.y);
}

//==============================================================================
//
//==============================================================================
void doUI()
{
	static nv::Rect none;
	static nv::Rect labelRect(12, 12);
	static nv::Rect glowButtonRect(300, 20, 0);
	static nv::Rect wireButtonRect(300, 60, 0);
	static nv::Rect sliderRect(0, 0, 100, 12);
	
	static const char* modelOptions[] = {"sphere", "tetra", "cube", "face"};
    static const char* bufferOptions[] = {"RGBA8", "RGBA16F", "RGBA32F", "R11F_G11F_B10F"};
    static const GLenum bufferTokens[] = { GL_RGBA8, GL_RGBA16F_ARB, GL_RGBA32F_ARB, GL_R11F_G11F_B10F_EXT};
    
	static int comboSelected = 0;

	ui.begin();

	if (bMenu)
	{
		ui.beginGroup( nv::GroupFlags_GrowDownFromLeft);

		ui.doLabel(none, "Advanced Skin Rendering");
		ui.doCheckButton(none, "Draw Wireframe", &options[OPTION_WIREFRAME]);
		ui.doCheckButton(none, "Moving Light", &options[OPTION_MOVE_LIGHT]);
		ui.doCheckButton(none, "Draw Debug Rect", &options[OPTION_DRAW_DEBUG]);

		ui.endGroup();
	}

    // Pass non-ui mouse events to the manipulator
    if (!ui.isOnFocus()) {
        const nv::ButtonState &lbState = ui.getMouseState( 0);
        const nv::ButtonState &mbState = ui.getMouseState( 1);
        const nv::ButtonState &rbState =  ui.getMouseState( 2);

        camera.motion( ui.getCursorX(), win_h - ui.getCursorY());

        updateButtonState( lbState, camera, GLUT_LEFT_BUTTON);
        updateButtonState( mbState, camera, GLUT_MIDDLE_BUTTON);
        updateButtonState( rbState, camera, GLUT_RIGHT_BUTTON);

    }

	ui.end();
}

//==============================================================================
//  display callback function
//==============================================================================
static bool bStretchMap = false;
void display()
{	
	// SSS 3 layer Rendering Procedure
	// 1. Specular Reflectance
	//		a. 
	//
	// 2. Subsurface Sacattering Diffuse
	//		a. render shadow map: render mesh to shadow map
	//		b. render stretch correcton map (precomputed)
	//		c. render irradiance into off-screen texture
	//			render mesh to 2D texture space
	//			this is the first non-blurred texture
	//		d. gaussian kernel (5 times)
	//			blur U, blur V
	//			generate 5 gaussian convolution textures
	//		e. render mesh in 3D
	//			access each gaussian convolution texture and combine linearly
	//			add specular for each light source (from step 1)
	
    setPerspectiveProjection(win_w, win_h);
    
	// 2.a draw shadow map
	makeShadowMap();    

	// 2.b draw stretch map: done
	if (!bStretchMap)
	{
		bStretchMap = true;
		makeStretchMap();
	}

	// 2.c non-blurred irr map: RTT
	renderFlatModel();

	// 2.d irr blur 1: RTT
	convolution(first_buffer, irr_buffer[0], 0);

	// 2.d irr blur 2: RTT
	convolution(irr_buffer[0], irr_buffer[1], 1);

	// 2.d irr blur 3: RTT
	convolution(irr_buffer[1], irr_buffer[2], 2);

	// 2.d irr blur 4: RTT
	convolution(irr_buffer[2], irr_buffer[3], 3);

	// 2.d irr blur 5: RTT
	convolution(irr_buffer[3], irr_buffer[4], 4);

	// 2.e final render pass: render to final framebuffer
	setPerspectiveProjection(win_w, win_h);
	renderSkin();
	//renderSSS();

	// debug viewport
	if ( options[OPTION_DRAW_DEBUG])
	{
		if (debug_mode == 0)
		{
			// show stretchUV and irr convolutions
			drawViewRect(skin_tex, 0);
			drawViewRect(shadow_buffer, 0, 1);
			drawViewRect(normal_tex, 0, 2);
			drawViewRect(spec_tex, 0, 3);
			drawViewRect(rho_tex, 0, 4);

			drawViewRect(first_buffer, 1);

			drawViewRect(stretch_buffer[0], 1, 1);
			drawViewRect(irr_buffer[0], 2);

			drawViewRect(stretch_buffer[1], 2, 1);
			drawViewRect(irr_buffer[1], 3);

			drawViewRect(stretch_buffer[2], 3, 1);
			drawViewRect(irr_buffer[2], 4);

			drawViewRect(stretch_buffer[3], 4, 1);
			drawViewRect(irr_buffer[3], 5);
			
			drawViewRect(stretch_buffer[4], 5, 1);
			drawViewRect(irr_buffer[4], 6);
		}
		else if (debug_mode == 1)
		{
			
		}
	}
	
	// for screenshot
	//drawViewport(irr_buffer[4]);

    //handle UI
    doUI();

    glutReportErrors();
    
	glutSwapBuffers();

	// update framerate
	framerateUpdate();
}

void testKeys()
{

}

//==============================================================================
//  idle callback function
//==============================================================================
void idle()
{
	static float last_time;
    float time = glutGet(GLUT_ELAPSED_TIME) / 1000.0;
    float dt = time - last_time;
    last_time = time;

	static float lightangle = 3.1415926535f * 0.25f;
	if ( options[OPTION_MOVE_LIGHT])
	{
		lightangle += dt*0.5f;
		lpos[0] = sin(lightangle) * 1.8f * sqrt(2.0);
		lpos[2] = cos(lightangle) * 1.8f * sqrt(2.0);
	}

    testKeys();
    glutPostRedisplay();
}

//==============================================================================
// keypress callback
//==============================================================================
void key(unsigned char k, int x, int y)
{
    ui.keyboard(k, x, y);

    k = tolower(k);

    if (optionKeyMap.find(k) != optionKeyMap.end())
        options[optionKeyMap[k]] = ! options[optionKeyMap[k]];

    switch (k) {
    case '\033':
    case 'q':
        if (fullscreen)
            glutLeaveGameMode();
        exit(0);
        break;	
    default:
        keydown[k] = 1;
        break;
    }
    printf("exposure = %f, blur = %f, blur_sigma = %f\n", exposure, blur_amount, blur_width);
    glutPostRedisplay();
}

//==============================================================================
//
//==============================================================================
void keyUp(unsigned char key, int x, int y)
{
    keydown[key] = 0;

	if ( tolower(key) == 'm')
		bMenu = !bMenu;
}

//==============================================================================
//
//==============================================================================
void mouse(int button, int state, int x, int y) {
    ui.mouse(button, state, glutGetModifiers(), x, y);
}

//==============================================================================
//
//==============================================================================
void motion(int x, int y) {
    ui.mouseMotion(x, y);
}

//==============================================================================
//
//==============================================================================
void passiveMotion(int x, int y) {
	ui.mouseMotion(x, y);
}

//==============================================================================
//
//==============================================================================
void setOrthoProjection(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1.0, 1.0);
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w, h);
}

//==============================================================================
//
//==============================================================================
void setPerspectiveProjection(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (float) w / (float) h, 0.1, 10.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, w, h);
}

//==============================================================================
//
//==============================================================================
void reshape(int w, int h)
{
    ui.reshape(w, h);

    win_w = w;
    win_h = h;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, win_w, 0, win_h, -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, win_w, win_h);

    createBuffers(buffer_format);
    initBlurCode(blur_width);

    camera.reshape(w, h);
    object.reshape(w, h);
}

//==============================================================================
//
//==============================================================================
void cgErrorCallback(void)
{
    CGerror lastError = cgGetError();
    if(lastError) {
        const char *listing = cgGetLastListing(context);
        printf("%s\n", cgGetErrorString(lastError));
        printf("%s\n", listing);
        exit(-1);
    }
}

//==============================================================================
//
//==============================================================================
CGprogram loadProgram(CGcontext context, char *path, char *filename, char *entry, CGprofile profile)
{
    char fullpath[256];
    sprintf(fullpath, "%s/%s", path, filename);
    std::string resolvedPath;

    CGprogram program = NULL;

	if (g_AppPath.getFilePath( fullpath, resolvedPath)) {
        program = cgCreateProgramFromFile(context, CG_SOURCE, resolvedPath.c_str(), profile, entry, NULL);
        if (!program) {
            fprintf(stderr, "Error creating program '%s'\n", fullpath);
        }
        cgGLLoadProgram(program);
    } else {
        fprintf(stderr, "Failed to locate program '%s'\n", fullpath);
    }

    return program;
}

//==============================================================================
//
//==============================================================================
void initCg()
{
    cgSetErrorCallback(cgErrorCallback);
    context = cgCreateContext();

    cg_vprofile = cgGLGetLatestProfile(CG_GL_VERTEX);
    cg_fprofile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
    
    char *path = "../shaders/"; //path relative to app root
    downsample_vprog = loadProgram(context, path, "downsample.cg", "downsample_vp", cg_vprofile);
    downsample_fprog = loadProgram(context, path, "downsample.cg", "downsample_fp", cg_fprofile);

    downsample4_vprog = loadProgram(context, path, "downsample.cg", "downsample4_vp", cg_vprofile);
    downsample4_fprog = loadProgram(context, path, "downsample.cg", "downsample4_fp", cg_fprofile);
    twoTexelSize_param = cgGetNamedParameter(downsample4_vprog, "twoTexelSize");

    tonemap_fprog = loadProgram(context, path, "tonemap.cg", "main", cg_fprofile);

    blurAmount_param = cgGetNamedParameter(tonemap_fprog, "blurAmount");
	effectAmount_param = cgGetNamedParameter(tonemap_fprog, "effectAmount");
	windowSize_param = cgGetNamedParameter(tonemap_fprog, "windowSize");
    exposure_param   = cgGetNamedParameter(tonemap_fprog, "exposure");

    skybox_vprog = loadProgram(context, path, "skybox.cg", "skybox_vp", cg_vprofile);
    skybox_fprog = loadProgram(context, path, "skybox.cg", "skybox_fp", cg_fprofile);

	// default render with single color texture
    object_vprog = loadProgram(context, path, "object.cg", "object_vp", cg_vprofile);
    model_param1	= cgGetNamedParameter(object_vprog, "model");
    object_fprog = loadProgram(context, path, "object.cg", "object_fp", cg_fprofile);
	model_param2	= cgGetNamedParameter(object_fprog, "model");

	// texture space model renderer
    flat_vprog = loadProgram(context, path, "objectflat.cg", "flat_vp", cg_vprofile);
    flat_model_param	= cgGetNamedParameter(flat_vprog, "model");
    flat_fprog = loadProgram(context, path, "objectflat.cg", "flat_fp", cg_fprofile);

	// final skin renderer
	skin_vprog = loadProgram(context, path, "skin.cg", "skin_vp", cg_vprofile);
    skin_fprog = loadProgram(context, path, "skin.cg", "skin_fp", cg_fprofile);

	final_vprog = loadProgram(context, path, "final.cg", "final_vp", cg_vprofile);
	final_model1_param	= cgGetNamedParameter(final_vprog, "model");
	final_viewproj_param	= cgGetNamedParameter(final_vprog, "viewProj");
	final_viewtarget_param	= cgGetNamedParameter(final_vprog, "viewProjWin_Target");
    final_fprog = loadProgram(context, path, "final.cg", "final_fp", cg_fprofile);
	final_model2_param	= cgGetNamedParameter(final_fprog, "model");
	final_litcolor_param	= cgGetNamedParameter(final_fprog, "lightColor");
	final_shadowcolor_param	= cgGetNamedParameter(final_fprog, "lightShadow");
	final_lightpos_param	= cgGetNamedParameter(final_fprog, "s_worldPointLightPos");
	final_eye_pos	= cgGetNamedParameter(final_fprog, "s_worldEyePos");
	final_diffmix_param	= cgGetNamedParameter(final_fprog, "s_diffColMix");
	final_gauss1_param	= cgGetNamedParameter(final_fprog, "gauss1w");
	final_gauss2_param	= cgGetNamedParameter(final_fprog, "gauss2w");
	final_gauss3_param	= cgGetNamedParameter(final_fprog, "gauss3w");
	final_gauss4_param	= cgGetNamedParameter(final_fprog, "gauss4w");
	final_gauss5_param	= cgGetNamedParameter(final_fprog, "gauss5w");
	final_gauss6_param	= cgGetNamedParameter(final_fprog, "gauss6w");

	// stretch map generator
	stretch_vprog = loadProgram(context, path, "stretch.cg", "stretch_vp", cg_vprofile);
	stretch_model_param	= cgGetNamedParameter(stretch_vprog, "model");
    stretch_fprog = loadProgram(context, path, "stretch.cg", "stretch_fp", cg_fprofile);
	stretch_scale_param	= cgGetNamedParameter(stretch_fprog, "scale");
	
	// blur irr map
	conv_vprog = loadProgram(context, path, "convolve.cg", "convolve_vp", cg_vprofile);
    convU_fprog = loadProgram(context, path, "convolve.cg", "convolveU_fp", cg_fprofile);
	convolveU_width_param	= cgGetNamedParameter(convU_fprog, "GaussWidth");
    convV_fprog = loadProgram(context, path, "convolve.cg", "convolveV_fp", cg_fprofile);
	convolveV_width_param	= cgGetNamedParameter(convV_fprog, "GaussWidth");

	// blur stretchUV map
	convStretch_vprog = loadProgram(context, path, "convolveStretch.cg", "convolveStretch_vp", cg_vprofile);
    convStretchU_fprog = loadProgram(context, path, "convolveStretch.cg", "convolveStretchU_fp", cg_fprofile);
	convStretchU_width_param	= cgGetNamedParameter(convStretchU_fprog, "GaussWidth");
    convStretchV_fprog = loadProgram(context, path, "convolveStretch.cg", "convolveStretchV_fp", cg_fprofile);
	convStretchV_width_param	= cgGetNamedParameter(convStretchV_fprog, "GaussWidth");

}

//==============================================================================
//
//==============================================================================
void createBuffers(GLenum format)
{
    // these are viewport dependent buffers
	if (scene_buffer) {
        delete scene_buffer;
    }
    for(int i=0; i<DOWNSAMPLE_BUFFERS; i++) {
        if (downsample_buffer[i])
            delete downsample_buffer[i];
    }
    for(int i=0; i<BLUR_BUFFERS; i++) {
        if (blur_buffer[i])
            delete blur_buffer[i];
    }
	
    GLenum target = GL_TEXTURE_2D;

    // create float pbuffers
    scene_buffer = new RenderTexture(win_w, win_h, target);
    scene_buffer->InitColor_Tex(0, format);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    scene_buffer->InitDepth_RB();
	
	// aa buffer
    if (aaModes[currentAAMode].samples > 0) {
        // create multisampled fbo
		ms_buffer = new RenderTexture(win_w, win_h, target, aaModes[currentAAMode].samples, aaModes[currentAAMode].coverage_samples);
		ms_buffer->InitColor_RB(0, format);
		ms_buffer->InitDepth_RB();
    }

    int w = win_w;
    int h = win_h;
	
	// downsample buffer
    for(int i=0; i<DOWNSAMPLE_BUFFERS; i++) {
        w /= 2;
        h /= 2;
        downsample_buffer[i] = new RenderTexture(w, h, target);
        downsample_buffer[i]->InitColor_Tex(0, format);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    // blur pbuffers
    for(int i=0; i<BLUR_BUFFERS; i++) {
        blur_buffer[i] = new RenderTexture(win_w / 4, win_h / 4, target);
        blur_buffer[i]->InitColor_Tex(0, format);
        glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
	
	// non-blur first render target
	if (!first_buffer)
	{
		first_buffer = new RenderTexture(BUFFERSIZE, BUFFERSIZE, target);
		first_buffer->InitColor_Tex(0, format);
		glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	}

	// shadow buffer
	if (!shadow_buffer)
	{
		shadow_buffer = new RenderTexture(depth_size, depth_size, target);
		glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(target, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
		//shadow_buffer->InitDepth_Tex();
		shadow_buffer->InitColor_Tex(0, format);
		shadow_buffer->InitDepth_RB();
	}

	// temp buffers
	if (!temp_buffer)
	{
		temp_buffer = new RenderTexture(BUFFERSIZE, BUFFERSIZE, target);
		temp_buffer->InitColor_Tex(0, format);
		glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	}

	// irr buffers
	if (!irr_buffer[0])
	{
		for(int i=0; i<IRR_BUFFERS; i++) {
			irr_buffer[i] = new RenderTexture(BUFFERSIZE, BUFFERSIZE, target);
			irr_buffer[i]->InitColor_Tex(0, format);
			glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		}
	}

	// stretch buffer
	if (!stretch_buffer[0])
	{
		for (int i=0; i<IRR_BUFFERS; i++)
		{
			stretch_buffer[i] = new RenderTexture(BUFFERSIZE, BUFFERSIZE, target);
			stretch_buffer[i]->InitColor_Tex(0, format);
			glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		}
		bStretchMap = false;
	}
}

//==============================================================================
// get extension pointers
//==============================================================================
void getExts()
{
    glewInit();

    if (!glewIsSupported(
        "GL_VERSION_2_0 "
        "GL_ARB_fragment_program "
        "GL_ARB_vertex_program "
        "GL_ARB_texture_float "
        "GL_ARB_color_buffer_float "
        "GL_EXT_framebuffer_object "
        ))
    {
        fprintf(stderr, "Error - required extensions were not supported\n");
        exit(-1);
    }

    if(!glewIsSupported("GL_EXT_framebuffer_blit "))
    {
        fprintf(stderr, "EXT_framebuffer_multisample not supported\n");
        currentAAMode = 0;
    }

    if (glewIsSupported("GL_EXT_packed_float")) {
        buffer_format = GL_R11F_G11F_B10F_EXT;
    }

    if (glewIsSupported("GL_EXT_texture_shared_exponent")) {
        texture_format = GL_RGB9_E5_EXT;
    }

    have_CSAA = glewIsSupported("GL_NV_framebuffer_multisample_coverage");
}

//==============================================================================
//
//==============================================================================
void initGL()
{
    getExts();
    initCg();
    createBuffers(buffer_format);

    glutReportErrors();
}

//==============================================================================
//
//==============================================================================
void initBlurCode(float blur_width)
{
    // delete old programs
    if (blurh_fprog) {
        glDeleteProgramsARB(1, &blurh_fprog);
    }
    if (blurh_fprog) {
        glDeleteProgramsARB(1, &blurh_fprog);
    }

    // generate weights for gaussian blur
    float *weights;
    int width;
    weights = generateGaussianWeights(blur_width, width);

    // generate blur fragment programs
    blurh_fprog = generate1DConvolutionFP_filter(weights, width, false, true, win_w / 2, win_h / 2);
    blurv_fprog = generate1DConvolutionFP_filter(weights, width, true, true, win_w / 2, win_h / 2);

    delete [] weights;
}

//==============================================================================
//
//==============================================================================
void loadTextures()
{
    std::string pathString;
	
	// skin diffuse texture: skin_filename, img_skin
	if ( g_AppPath.getFilePath(skin_filename, pathString)) {
        if (!img_skin.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", skin_filename);
        exit(-1);        
    }

	// skin normal texture: normal_filename, img_normal
	if ( g_AppPath.getFilePath(normal_filename, pathString)) {
        if (!img_normal.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", normal_filename);
        exit(-1);        
    }

	// skin spec texture: spec_filename, img_spec
	if ( g_AppPath.getFilePath(spec_filename, pathString)) {
        if (!img_spec.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", spec_filename);
        exit(-1);        
    }

	// skin stretch texture: stretch_filename, img_stretch
	if ( g_AppPath.getFilePath(stretch_filename, pathString)) {
        if (!img_stretch.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", stretch_filename);
        exit(-1);        
    }

	// rho_d texture: rho_d_filename, img_rho
	if ( g_AppPath.getFilePath(rho_d_filename, pathString)) {
        if (!img_rho.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", rho_d_filename);
        exit(-1);        
    }

	// cube texture: cube_filename, img_cube
	if ( g_AppPath.getFilePath(cube_filename, pathString)) {
        if (!img_cube.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
        if (!img_cube.convertCrossToCubemap()) {
            fprintf(stderr, "Error converting image to cubemap\n");
            exit(-1);        
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", cube_filename);
        exit(-1);        
    }

	// cube diff texture: cubediff_filename, img_cubediff
	if ( g_AppPath.getFilePath(cubediff_filename, pathString)) {
        if (!img_cubediff.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
        if (!img_cubediff.convertCrossToCubemap()) {
            fprintf(stderr, "Error converting image to cubemap\n");
            exit(-1);        
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", cubediff_filename);
        exit(-1);        
    }

	// cube convolution texture: cubeconv_filename, img_cubeconv
	if ( g_AppPath.getFilePath(cubeconv_filename, pathString)) {
        if (!img_cubeconv.loadImageFromFile(pathString.c_str())) {
            fprintf(stderr, "Error loading image file '%s'\n", pathString.c_str());
            exit(-1);
        }
        if (!img_cubeconv.convertCrossToCubemap()) {
            fprintf(stderr, "Error converting image to cubemap\n");
            exit(-1);        
        }
    }
    else {
        fprintf(stderr, "Filed to find image '%s'\n", cubeconv_filename);
        exit(-1);        
    }

	// skin texture
	skin_tex = create2DTexture(img_skin, texture_format);

	// skin normal
	normal_tex = create2DTexture(img_normal, texture_format);

	// skin specular
	spec_tex = create2DTexture(img_spec, texture_format);

	// skin stretch
	stretch_tex = create2DTexture(img_stretch, texture_format);

	// rho_d
	rho_tex = create2DTexture(img_rho, texture_format);

	// cube color
    cube_tex = createCubemapTexture(img_cube, texture_format);
    hdr_tex = cube_tex;

	// cube diff
	cubediff_tex = createCubemapTexture(img_cubediff, texture_format);

	// cube conv
	cubeconv_tex = createCubemapTexture(img_cubeconv, texture_format);

}

//==============================================================================
//
//==============================================================================
int main(int argc, char **argv)
{
    std::string pathString;
    glutInit(&argc, argv);

    char displayString[256];
    sprintf(displayString, "double rgb~8 depth~16");

    if (fullscreen) {
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
        char gamemode[256];
        sprintf(gamemode, "%dx%d:%d", win_w, win_h, 32);
        glutGameModeString(gamemode);
        int win = glutEnterGameMode();

    } else {
        // windowed
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

        glutInitWindowSize(win_w, win_h);
        (void) glutCreateWindow("Advanced Skin Rendering");
		framerateTitle("Advanced Skin Rendering");
    }

    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(passiveMotion);
    glutIdleFunc(idle);
    glutKeyboardFunc(key);
    glutKeyboardUpFunc(keyUp);
    glutReshapeFunc(reshape);

    camera.setTrackballActivate(GLUT_LEFT_BUTTON, GLUT_ACTIVE_ALT);
	camera.setDollyActivate( GLUT_RIGHT_BUTTON, GLUT_ACTIVE_ALT);
    camera.setPanActivate( GLUT_MIDDLE_BUTTON, GLUT_ACTIVE_ALT);
    camera.setDollyPosition( -3.0f);

	// set camera rotating
	nv::ExamineManipulator &manip = (nv::ExamineManipulator &) camera.getManipulator();
	manip.getIncrement().set_value(nv::vec3f(0.0, 1.0, 0), 0.01);

    optionKeyMap['w'] = OPTION_WIREFRAME;
    options[OPTION_WIREFRAME] = false;

    optionKeyMap['g'] = OPTION_GLOW;
    options[OPTION_GLOW] = false;

    optionKeyMap['d'] = OPTION_DRAW_DEBUG;
    options[OPTION_DRAW_DEBUG] = false;
	
	optionKeyMap['l'] = OPTION_MOVE_LIGHT;
    options[OPTION_MOVE_LIGHT] = false;


    printf( "Advanced Skin Rendering\n");
    printf( "   q/[ESC]    - Quit the app\n");
    printf( "      w       - Toggle displaying the object in wireframe\n");
    printf( "      g       - Toggle the glow pass\n");
	printf( "      d       - Toggle viewport debugging rect\n");
    printf( "   [MOUSE]    - Left + Alt - rotate, Middle + Alt - pan, Right + Alt - zoom\n");
    printf( "      m       - Toggle OnScreen Menu\n");
    

	// setting buffer to null at the beginning
    for(int i=0; i<DOWNSAMPLE_BUFFERS; i++)
        downsample_buffer[i] = 0;
    
	for(int i=0; i<BLUR_BUFFERS; i++)
        blur_buffer[i] = 0;
	
	for(int i=0; i<IRR_BUFFERS; i++)
	{
        irr_buffer[i] = 0;
		stretch_buffer[i] = 0;
	}

	
    initGL();
    initBlurCode(blur_width);

	loadTextures();

    // create the model and make it ready for display
    model = new nv::Model;
	if ( g_AppPath.getFilePath(model_filename, pathString)) {
        if (model->loadModelFromFile(pathString.c_str())) {

            // compute normal
            model->computeNormals();

            // get the bounding box to help place the model on-screen
            model->rescale(1.2f);

            // make the model efficient for rendering with vertex arrays
            model->compileModel( nv::Model::eptAll);

            glNewList(1, GL_COMPILE);
            drawModel(model);
            glEndList();

        } else {
            fprintf(stderr, "Error loading model '%s'\n", pathString.c_str());
            delete model;
            model = 0;
        }
    } else {
        fprintf(stderr, "Unable to locate model '%s'\n", model_filename);
        delete model;
        model = 0;
    }

    glutMainLoop();
    return 0;
}
