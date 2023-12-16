/*****************************************

 framerate.h

 ====================

 Usage: to add an on-screen frames per second counter to your program, save
 this file alongside your app as "frames.h", and add:

    #include "frames.h"

 add framerateUpdate() call in your display function.

 ====================

 Example:

    void display(void) {
      glClear(GL_COLOR_BUFFER_BIT);
 
      framerateUpdate();
     
      // all the graphics code

      glutSwapBuffers();
    }
*****************************************/

//#include <sys/time.h>
//#include <stdio.h>

//struct timeval frameStartTime, frameEndTime;
float frameStartTime, frameEndTime;
char	appTitle[64] = "OpenGL App";			// window title
float	refreshtime = 0.5f;						// fps refresh period
float	gElapsedTime = 0.0f;					// current frame elapsed time
float	gTimeAccum = 0.0f;						// time accumulator for refresh
int		gFrames = 0;							// frame accumulator
float	gFPS = 0.0f;							// framerate

void framerateTitle(char* title) {
	strcpy(appTitle, title);
	glutSetWindowTitle(title);
	frameStartTime = frameEndTime = 0.0f;
}

void framerateUpdate(void)
{
	frameEndTime = GetTickCount()/1000.0f;
	//gettimeofday(&frameEndTime, NULL);
	
	gElapsedTime = frameEndTime - frameStartTime;
	//gElapsedTime = frameEndTime.tv_sec - frameStartTime.tv_sec +
    //         ((frameEndTime.tv_usec - frameStartTime.tv_usec)/1.0E6);
    frameStartTime = frameEndTime;
    
    gTimeAccum += gElapsedTime;
    gFrames++;
    
	if (gTimeAccum > refreshtime)
	{  
		char title[64];
		gFPS = (float) gFrames / gTimeAccum;
		sprintf(title, "%s : %3.1f fps", appTitle, gFPS);
		glutSetWindowTitle(title);
		gTimeAccum = 0.0f;
		gFrames = 0;
	}

}
/* end of frames.h */
