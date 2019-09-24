#include <iostream>
#include <opencv2/videoio.hpp>
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

void forwarp_map(Mat *outImage, Mat image, Mat warpmesh) {
    Vec3b blackpixel = Vec3b({0, 0, 0});

    //poorly implemented forward warping -- no interpolation used
    for(int y=0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            int x_ = max(int(std::round(warpmesh.at<Vec2d>(y,x)[0])), 0);
            int y_ = max(int(std::round(warpmesh.at<Vec2d>(y,x)[1])), 0);
            x_ = min(x_, image.cols - 1);
            y_ = min(y_, image.rows - 1);

            outImage->at<Vec3b>(y_, x_) = image.at<Vec3b>(y,x);

            // if any 4-way neighbouring pixels are undefined copy pixel value over (splatting)

            if (outImage->at<Vec3b>(y_+1, x_) == blackpixel) {
                outImage->at<Vec3b>(y_+1, x_) = image.at<Vec3b>(y,x);
            }

            if (outImage->at<Vec3b>(y_-1, x_) == blackpixel) {
                outImage->at<Vec3b>(y_-1, x_) = image.at<Vec3b>(y,x);
            }

            if (outImage->at<Vec3b>(y_, x_+1) == blackpixel) {
                outImage->at<Vec3b>(y_, x_+1) = image.at<Vec3b>(y,x);
            }

            if (outImage->at<Vec3b>(y_, x_-1) == blackpixel) {
                outImage->at<Vec3b>(y_, x_-1) = image.at<Vec3b>(y,x);
            }

            if (outImage->at<Vec3b>(y_+1, x_+1) == blackpixel) {
                outImage->at<Vec3b>(y_+1, x_+1) = image.at<Vec3b>(y,x);
            }

            if (outImage->at<Vec3b>(y_-1, x_-1) == blackpixel) {
                outImage->at<Vec3b>(y_-1, x_-1) = image.at<Vec3b>(y,x);
            }

            if (outImage->at<Vec3b>(y_-1, x_+1) == blackpixel) {
                outImage->at<Vec3b>(y_-1, x_+1) = image.at<Vec3b>(y,x);
            }

            if (outImage->at<Vec3b>(y_+1, x_-1) == blackpixel) {
                outImage->at<Vec3b>(y_+1, x_-1) = image.at<Vec3b>(y,x);
            }
        }
    }
}

int main(int argc, char* argv[]) {

    CascadeClassifier clf;
    clf.load("../cascade-classifiers/haarcascade_frontalface_alt.xml");

    //load video
    VideoCapture cap("../photos/MAH00217.mp4");

    Mat image;
    // Capture frame-by-frame
    cap >> image;

    VideoWriter video("../photos/outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'),10,
            Size(image.cols, image.rows));

    //set fov of image
    double FOV = 100.0;
    double d = double(min(image.rows, image.cols));
    double f = getfocallength(image.cols, FOV);

    double r120 = f * std::tan(double(120.0 / 2));

    double rb = r120 / (std::log((1.0 / 0.01)-1.0)-std::log((1.0 / 0.99)-1.0));
    double ra = rb * std::log((1.0 / 0.01)-1.0);

    //fill in empty meshes
    Mat stereo_yxmap = Mat(103 + 8, 78 + 8, CV_64FC2);
    Mat quad_yxmap = Mat(103 + 8, 78 + 8, CV_64FC2);

    stereographic(stereo_yxmap, f, d, image.cols, image.rows);

    //re-use quadmesh from previous frame for efficiency
    quadmesh(quad_yxmap, image.cols, image.rows);

    while(1){

        // Capture frame-by-frame
        if (image.empty())
            break;

        Mat stereo_yxmap = Mat(103 + 8, 78 + 8, CV_64FC2);
        Mat quad_yxmap = Mat(103 + 8, 78 + 8, CV_64FC2);

        //detect faces
        Mat gray_img;
        cvtColor(image, gray_img, COLOR_BGR2GRAY );
        std::vector<Rect> faces;
        clf.detectMultiScale(gray_img, faces);

        //create and solve optimization problem
        Problem problem;
        DefineProblem(&problem, &quad_yxmap, &image, &stereo_yxmap, faces, ra, rb);

        //create empty array for output image
        Mat outImage = Mat(image.rows, image.cols, image.type());
        Mat warpmesh = Mat(image.rows, image.cols, CV_64FC2);

        //up scale the optimized mesh to full size
        resize(quad_yxmap, warpmesh, warpmesh.size());
        forwarp_map(&outImage, image, warpmesh);
        video.write(outImage);

        // If the frame is empty, break immediately
        cap >> image;

    }

    cap.release();
    video.release();

    return 0;
}

