//
// Created by kieran on 15/9/19.
//

#ifndef PROJECTION_DISTORTION_ENERGYFUNCTIONS_H
#define PROJECTION_DISTORTION_ENERGYFUNCTIONS_H

#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>

void DefineProblem(ceres::Problem* problem, cv::Mat* quad_yxmap, cv::Mat *image, cv::Mat *stereo_yxmap, cv::Mat *last_mesh,
                   std::vector<cv::Rect> faces, double ra, double rb, int max_iter);

struct LineBending;
struct Regularizer;
struct FaceRegularizer;
struct BoundaryEnergy;
struct FaceEnergy;
struct TimeEnergy;


#endif //PROJECTION_DISTORTION_ENERGYFUNCTIONS_H
