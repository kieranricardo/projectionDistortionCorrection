//
// Created by kieran on 14/9/19.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "MeshUtils.h"

using namespace cv;
using namespace std;


double getfocallength(int w, double fov) {
    double f = double(w) / (2.0 * std::tan(fov / 2.0));
    return f;
}


void stereographic(Mat yxmap, double f, double d, double imwidth, double imheight) {

    double r0 = d / (2 * (std::tan(0.5 * std::atan(d / (2 * f)))));

    for(int y=0; y < yxmap.rows; y++){
        for(int x=0; x < yxmap.cols; x++){
            double x_ = double(imwidth) * (double(x-4) / double(yxmap.cols-8)) - (double(imwidth) / 2);
            double y_ = double(imheight) * (double(y-4) / double(yxmap.rows-8)) - (double(imheight) / 2);
            double rp = std::sqrt((x_ * x_) + (y_ * y_));
            double ru = r0 * std::tan(0.5 * std::atan(rp / f));
            double scaleFactor = ru / rp;
            yxmap.at<Vec2d>(y,x)[0] = (double(x_) * scaleFactor + (double(imwidth) / 2.0)); // / double(imwidth);
            yxmap.at<Vec2d>(y,x)[1] = (double(y_) * scaleFactor + (double(imheight) / 2.0));  // double(imheight);
        }
    }
}


void quadmesh(Mat yxmap, double imwidth, double imheight) {
    for(int y=0; y< yxmap.rows; y++) {
        for (int x = 0; x < yxmap.cols; x++) {
            double x_ = imwidth * (double(x-4) / double(yxmap.cols - 8));
            double y_ = imheight * (double(y-4) / double(yxmap.rows - 8));

            yxmap.at<Vec2d>(y,x)[0] = double(x_);
            yxmap.at<Vec2d>(y,x)[1] = double(y_);
        }
    }
}
