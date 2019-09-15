#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <math.h>
#include <ceres/ceres.h>
#include "EnergyFunctions.h"
#include "MeshUtils.h"

using namespace cv;
using namespace ceres;
using namespace std;


int main(int argc, char* argv[]) {

    //set fov of image
    double FOV = 100.0;
    //load image
    Mat image = imread("../photos/104_8445982227_0414221e99_o.jpg");

    //create empty arrays for original and steroegraphic mesh
    Mat stereo_yxmap = Mat(103 + 8, 78 + 8, CV_64FC2);
    Mat quad_yxmap = Mat(103 + 8, 78 + 8, CV_64FC2);

    //create empty array for output image
    Mat outImage = Mat(image.rows, image.cols, image.type());

    //calculate focal length fro FOV + some other parameters specified in paper
    double d = double(min(image.rows, image.cols));
    double f = getfocallength(image.cols, FOV);

    double r120 = f * std::tan(double(120.0 / 2));

    double rb = r120 / (std::log((1.0 / 0.01)-1.0)-std::log((1.0 / 0.99)-1.0));
    double ra = rb * std::log((1.0 / 0.01)-1.0);

    //fill in empty meshes
    stereographic(stereo_yxmap, f, d, image.cols, image.rows);
    quadmesh(quad_yxmap, image.cols, image.rows);

    //detect faces
    CascadeClassifier clf;
    clf.load("../cascade-classifiers/haarcascade_frontalface_alt.xml");
    Mat gray_img;
    cvtColor(image, gray_img, COLOR_BGR2GRAY );
    std::vector<Rect> faces;
    clf.detectMultiScale(gray_img, faces);

    //create and solve optimization problem
    Problem problem;
    DefineProblem(&problem, &quad_yxmap, &image, &stereo_yxmap, faces, ra, rb);

    //create the mesh used for warping
    Mat warpmesh = Mat(image.rows, image.cols, CV_64FC2);

    //up scale the optimized mesh to full size
    resize(quad_yxmap, warpmesh, warpmesh.size());

    Vec3b blackpixel = Vec3b({0, 0, 0});

    //poorly implemented forward warping -- no interpolation used
    for(int y=0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int x_ = max(int(std::round(warpmesh.at<Vec2d>(y,x)[0])), 0);
            int y_ = max(int(std::round(warpmesh.at<Vec2d>(y,x)[1])), 0);
            x_ = min(x_, image.cols - 1);
            y_ = min(y_, image.rows - 1);

            outImage.at<Vec3b>(y_, x_) = image.at<Vec3b>(y,x);

            // if any 4-way neighbouring pixels are undefined copy pixel value over (splatting)

            if (outImage.at<Vec3b>(y_+1, x_) == blackpixel) {
                outImage.at<Vec3b>(y_+1, x_) = image.at<Vec3b>(y,x);
            }

            if (outImage.at<Vec3b>(y_-1, x_) == blackpixel) {
                outImage.at<Vec3b>(y_-1, x_) = image.at<Vec3b>(y,x);
            }

            if (outImage.at<Vec3b>(y_, x_+1) == blackpixel) {
                outImage.at<Vec3b>(y_, x_+1) = image.at<Vec3b>(y,x);
            }

            if (outImage.at<Vec3b>(y_, x_-1) == blackpixel) {
                outImage.at<Vec3b>(y_, x_-1) = image.at<Vec3b>(y,x);
            }
        }
    }

    //draw detected face bounding boxes in original image
    for (int i=0; i<faces.size(); i++) {
        rectangle(image, faces[i], {255, 0, 0}, 10);
    }

    //display original and warped image
    String windowName = "Corrected Image";
    namedWindow(windowName, WINDOW_NORMAL);
    resizeWindow(windowName, 400, 300);
    imshow(windowName, outImage);

    windowName = "Original Image";
    namedWindow(windowName, WINDOW_NORMAL);
    resizeWindow(windowName, 400, 300);
    imshow(windowName, image);

    waitKey(0); // Wait for any keystroke in the window
    destroyWindow(windowName);

    return 0;
}

