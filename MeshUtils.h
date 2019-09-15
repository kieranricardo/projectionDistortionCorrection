//
// Created by kieran on 15/9/19.
//

#ifndef PROJECTION_DISTORTION_MESHUTILS_H
#define PROJECTION_DISTORTION_MESHUTILS_H

#include <opencv2/opencv.hpp>

double getfocallength(int w, double fov);
void stereographic(cv::Mat yxmap, double f, double d, double imwidth, double imheight);
void quadmesh(cv::Mat yxmap, double imwidth, double imheight);

#endif //PROJECTION_DISTORTION_MESHUTILS_H
