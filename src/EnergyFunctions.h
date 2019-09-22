//
// Created by kieran on 15/9/19.
//

#ifndef PROJECTION_DISTORTION_ENERGYFUNCTIONS_H
#define PROJECTION_DISTORTION_ENERGYFUNCTIONS_H

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

void DefineProblem(ceres::Problem* problem, cv::Mat *quad_yxmap, cv::Mat *image,
        cv::Mat *stereo_yxmap, std::vector<cv::Rect> faces, double ra, double rb);

struct LineBending;
struct Regularizer;
struct FaceRegularizer;
struct BoundaryEnergy;
struct FaceEnergy;


#endif //PROJECTION_DISTORTION_ENERGYFUNCTIONS_H
