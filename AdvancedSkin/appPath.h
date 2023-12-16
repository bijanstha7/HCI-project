#ifndef APPPATH_H
#define APPPATH_H

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <windows.h>

// application path handler...

class appPath {

public:
    appPath() {
		char rootPath[256];
		GetCurrentDirectoryA(2048, rootPath);
        
		//search local paths first, in-case someone has the SDK installed while hacking another copy 
        _pathList.push_back("./");  // present directory
        _pathList.push_back("../"); // back one
        _pathList.push_back("../../"); // back two

        if ( rootPath) {
            _pathList.push_back(std::string(rootPath) + "/");  // Path lacks a terminating slash
        }
    }

    void addPath( const std::string &path) {
        _pathList.push_back(path);
    }

    void clearPaths() {
        _pathList.clear();
    }

    bool getFilePath( const std::string &file, std::string &path) {
        std::string pathString;
        
        for ( std::vector<std::string>::iterator it = _pathList.begin(); it != _pathList.end(); it++) {
            pathString = *it + file;
            FILE *fp = fopen( pathString.c_str(), "rb");
            if (fp) {
                fclose(fp);
                path = pathString;
                return true;
            }
        }

        return false;
    }

    bool getPath( const std::string &file, std::string &path) {
        std::string pathString;
        
        for ( std::vector<std::string>::iterator it = _pathList.begin(); it != _pathList.end(); it++) {
            pathString = *it + file;
            FILE *fp = fopen( pathString.c_str(), "rb");
            if (fp) {
                fclose(fp);
                path = *it;
                return true;
            }
        }

        return false;
    }

private:
    std::vector<std::string> _pathList;

};

#endif