//
// Created by kieran on 14/9/19.
//
#include <math.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include "EnergyFunctions.h"

using namespace cv;
using namespace std;
using namespace ceres;


struct TimeEnergy {
    TimeEnergy(double last_x, double last_y)
            : last_x(last_x), last_y(last_y) {}

    template <typename T>
    bool operator()(const T* const x, const T* const y, T* residuals) const {
        residuals[0] = T(2) * (last_x  - T(x[0]));
        residuals[1] = T(2) * (last_y - T(y[0]));
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double last_x,
                                       const double last_y) {
        return (new ceres::AutoDiffCostFunction<TimeEnergy, 2, 1, 1>(
                new TimeEnergy(last_x, last_y)));
    }

    double last_x;
    double last_y;
};


struct FaceEnergy {
    FaceEnergy(double stereo_x, double stereo_y, double m)
            : stereo_x(stereo_x), stereo_y(stereo_y), m(m) {}

    template <typename T>
    bool operator()(const T* const x, const T* const y, const T* const a, const T* const b,
                    const T* const tx, const T* const ty, T* residuals) const {
        residuals[0] = m * T(2) * ((a[0] * stereo_x + b[0] * stereo_y + tx[0]) - T(x[0]));
        residuals[1] = m * T(2) * ((a[0] * stereo_y - b[0] * stereo_x + ty[0]) - T(y[0]));
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double stereo_x,
                                       const double stereo_y,
                                       const double m) {
        return (new ceres::AutoDiffCostFunction<FaceEnergy, 2, 1, 1, 1, 1, 1, 1>(
                new FaceEnergy(stereo_x, stereo_y, m)));
    }

    double stereo_x;
    double stereo_y;
    double m;
};


struct BoundaryEnergy {
    BoundaryEnergy(double boundaryValue, double boundaryFlag)
            :  boundaryValue(boundaryValue), boundaryFlag(boundaryFlag) {}

    template <typename T>
    bool operator()(const T* const x, T* residuals) const {
        if (boundaryFlag == 0) {
            //lower boundary
            residuals[0] = T(2) * T(x[0] > boundaryValue) * (x[0] - boundaryValue);
        } else {
            //upper boundary
            residuals[0] = T(2) * T(x[0] < boundaryValue) * (x[0] - boundaryValue);
        }
        return true;
    }
    static ceres::CostFunction* Create(const double boundaryValue,
                                       const double boundaryFlag) {
        return (new ceres::AutoDiffCostFunction<BoundaryEnergy, 1, 1>(
                new BoundaryEnergy(boundaryValue, boundaryFlag)));
    }

    double boundaryValue;
    double boundaryFlag;

};


struct FaceRegularizer {
    template <typename T>
    //prevent scaling faces too much
    bool operator()(const T* const a, T* residuals) const {
        residuals[0] = T(2) * T(45) * (T(1.0) - a[0]);
        return true;
    }
};


struct Regularizer {
    template <typename T>
    //prevent vertex from moving too far from neighbours
    //maybe replace original mesh with prev time step for video?
    bool operator()(const T* const x1, const T* const y1, const T* const x2, const T* const y2, T* residual) const {
        residual[0] = T(0.7) * (x1[0] - x2[0]);
        residual[1] = T(0.7) * (y1[0] - y2[0]);
        return true;
    }
};


struct LineBending {
    template <typename T>
    //prevent local twiting of grid --> prevents rotation, try prevent shear instead?
    bool operator()(const T* const v1, const T* const v2, T* residual) const {
        residual[0] = T(1.4) * (v1[0] - v2[0]);
        return true;
    }
};


void DefineProblem(Problem* problem, Mat* quad_yxmap, Mat *image, Mat *stereo_yxmap, Mat *last_mesh,
                   std::vector<Rect> faces, double ra, double rb, int max_iter){

    double a_s[faces.size()];
    double b_s[faces.size()];
    double tx_s[faces.size()];
    double ty_s[faces.size()];

    for (int i = 0; i < faces.size(); i++ ) {
        a_s[i] = 1.0;
        b_s[i] = 0.0;
        tx_s[i] = 0.0;
        ty_s[i] = 0.0;
        int y0 = int(double(faces[i].y) * double(quad_yxmap->rows - 8) / double(image->rows)) + 4;
        int x0 = int(double(faces[i].x) * double(quad_yxmap->cols - 8) / double(image->cols)) + 4;

        int width = int(double(faces[i].width) * double(quad_yxmap->cols - 8) / double(image->cols));
        int height = int(double(faces[i].height) * double(quad_yxmap->rows - 8) / double(image->rows));

        for(int y=y0; y < (y0 + height); y++) {
            for (int x=x0; x < (x0 + width); x++) {
                double x_ = double(image->cols) * (double(x-4) / double(quad_yxmap->cols-8)) - (double(image->cols) / 2);
                double y_ = double(image->rows) * (double(y-4) / double(quad_yxmap->rows-8)) - (double(image->rows) / 2);
                double rp = std::sqrt((x_ * x_) + (y_ * y_));
                double m =  1.0; //std::sqrt(1.0 / (1.0 + std::exp(-(rp-ra) / rb))); //sqrt to account for squaring in residual

                ceres::CostFunction * cost_function =
                        FaceEnergy::Create(stereo_yxmap->at<Vec2d>(y,x)[0], stereo_yxmap->at<Vec2d>(y,x)[1], m);
                problem->AddResidualBlock(cost_function, nullptr, &quad_yxmap->at<Vec2d>(y,x)[0],
                                          &quad_yxmap->at<Vec2d>(y,x)[1], &a_s[i], &b_s[i], &tx_s[i], &ty_s[i]);
            }
        }

        CostFunction* cost_function =
                new AutoDiffCostFunction<FaceRegularizer, 1, 1>(new FaceRegularizer);
        problem->AddResidualBlock(cost_function, NULL, &a_s[i]);
    }

    for(int y=0; y < quad_yxmap->rows; y++) {
        for (int x = 0; x < quad_yxmap->cols; x++) {
            ceres::CostFunction *cost_function = TimeEnergy::Create(last_mesh->at<Vec2d>(y,x)[0], last_mesh->at<Vec2d>(y,x)[1]);
            problem->AddResidualBlock(cost_function, nullptr, &quad_yxmap->at<Vec2d>(y,x)[0], &quad_yxmap->at<Vec2d>(y,x)[1]);
        }
    }

    int deltas[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    for(int y=0; y < quad_yxmap->rows; y++) {
        for (int x=0; x < quad_yxmap->cols; x++) {
            for (int i=0; i<4; i++) {

                int x_ = x + deltas[i][0];
                int y_ = y + deltas[i][1];
                int k = int((deltas[i][0] != 0));

                if ((x_ >= 0) & (x_ < quad_yxmap->cols) & (y_ >= 0) & (y_ < quad_yxmap->rows)) {
                    CostFunction* cost_function1 =
                            new AutoDiffCostFunction<Regularizer, 2, 1, 1, 1, 1>(new Regularizer);
                    problem->AddResidualBlock(cost_function1, NULL, &quad_yxmap->at<Vec2d>(y,x)[0], &quad_yxmap->at<Vec2d>(y,x)[1],
                                             &quad_yxmap->at<Vec2d>(y_,x_)[0], &quad_yxmap->at<Vec2d>(y_,x_)[1]);

                    CostFunction *cost_function =
                            new AutoDiffCostFunction<LineBending, 1, 1, 1>(new LineBending);
                    problem->AddResidualBlock(cost_function, NULL, &quad_yxmap->at<Vec2d>(y, x)[k],
                                             &quad_yxmap->at<Vec2d>(y_, x_)[k]);
                }
            }
        }
    }

    for (int y=0; y < quad_yxmap->rows; y++) {

        ceres::CostFunction * cost_function1 =
                BoundaryEnergy::Create(0, 0);
        problem->AddResidualBlock(cost_function1, NULL, &quad_yxmap->at<Vec2d>(y, 0)[0]);

        ceres::CostFunction * cost_function2 =
                BoundaryEnergy::Create(double(image->cols), 1);

        problem->AddResidualBlock(cost_function2, NULL, &quad_yxmap->at<Vec2d>(y, 85)[0]);

        problem->SetParameterBlockConstant(&quad_yxmap->at<Vec2d>(y,0)[0]);
        problem->SetParameterBlockConstant(&quad_yxmap->at<Vec2d>(y,(quad_yxmap->cols-1))[0]);
    }


    for (int x=0; x < quad_yxmap->cols; x++) {

        ceres::CostFunction * cost_function1 =
                BoundaryEnergy::Create(0, 0);
        problem->AddResidualBlock(cost_function1, NULL, &quad_yxmap->at<Vec2d>(0, x)[1]);

        ceres::CostFunction * cost_function2 =
                BoundaryEnergy::Create(double(image->rows), 1);
        problem->AddResidualBlock(cost_function2, NULL, &quad_yxmap->at<Vec2d>(110, x)[1]);

        problem->SetParameterBlockConstant(&quad_yxmap->at<Vec2d>(0,x)[1]);
        problem->SetParameterBlockConstant(&quad_yxmap->at<Vec2d>((quad_yxmap->rows-1), x)[1]);
    }

    Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = max_iter;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;
    Solver::Summary summary;
    Solve(options, problem, &summary);

    std::cout << summary.BriefReport() << "\n\n\n";
}

