// Import the aruco module in OpenCV
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp> 
#include "opencv2/calib3d/calib3d.hpp"
#include "iostream"
#include "string"
#include "stdlib.h"
#include "stdio.h"

using namespace cv;
using namespace std;


static bool readCameraParameters(std::string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}


static bool readDetectorParameters(std::string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}


cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
Mat camMatrix, distCoeffs;

// draw 2d cube projection
void drawCubeWireframe(
    cv::InputOutputArray image, cv::InputArray cameraMatrix,
    cv::InputArray distCoeffs, cv::InputArray rvec, cv::InputArray tvec,
    float l )
{

    CV_Assert(
        image.getMat().total() != 0 &&
        (image.getMat().channels() == 1 || image.getMat().channels() == 3)
    );
    CV_Assert(l > 0);
    float half_l = l / 2.0;

    // 3d cube points
    std::vector<cv::Point3f> axisPoints;
    axisPoints.push_back(cv::Point3f( half_l,  half_l, l));
    axisPoints.push_back(cv::Point3f( half_l, -half_l, l));
    axisPoints.push_back(cv::Point3f(-half_l, -half_l, l));
    axisPoints.push_back(cv::Point3f(-half_l,  half_l, l));
    axisPoints.push_back(cv::Point3f( half_l,  half_l, 0));
    axisPoints.push_back(cv::Point3f( half_l, -half_l, 0));
    axisPoints.push_back(cv::Point3f(-half_l, -half_l, 0));
    axisPoints.push_back(cv::Point3f(-half_l,  half_l, 0));

    // project cube points
    std::vector<cv::Point2f> imagePoints;
    projectPoints(
        axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints
    );

    // draw cube edges lines
    cv::line(image, imagePoints[0], imagePoints[1], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[0], imagePoints[4], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[1], imagePoints[2], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[1], imagePoints[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[2], imagePoints[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[2], imagePoints[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[3], imagePoints[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[4], imagePoints[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[4], imagePoints[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[5], imagePoints[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, imagePoints[6], imagePoints[7], cv::Scalar(255, 0, 0), 3);
    //

    //vector<Point2f> right_side    = {imagePoints[0], imagePoints[1], imagePoints[5], imagePoints[4]};
    //vector<Point2f> left_side     = {imagePoints[2], imagePoints[3], imagePoints[7], imagePoints[6]};
    //vector<Point2f> front_side    = {imagePoints[1], imagePoints[2], imagePoints[6], imagePoints[5]};
    //vector<Point2f> back_side     = {imagePoints[0], imagePoints[3], imagePoints[7], imagePoints[4]};
    //vector<vector<Point2f>> sides = {right_side, left_side, front_side, back_side };
    //cv::drawContours(image, sides, -1, cv::Scalar(150, 0, 0));

}

void ProcessShot(cv::Mat& input, cv::Mat& output)
{
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    // detect markers in the shot
    cv::aruco::detectMarkers(input, dictionary, corners, ids);
    int num = ids.size(); 
    if (num == 0) return;
    // if at least one marker detected...

    // draw the basic frame
    cv::aruco::drawDetectedMarkers(output, corners, ids);
    // get the marker positions
    std::vector<cv::Vec3d> rvecs, tvecs;
    aruco::estimatePoseSingleMarkers(corners, 0.02, camMatrix, distCoeffs, rvecs, tvecs);
    for (int m = 0; m < num; m++)
    {
        // move verticies to the marker location and draw the cube
        drawCubeWireframe(output, camMatrix, distCoeffs, rvecs[m], tvecs[m], 0.02);
    }   
}

int main()
{
    std::string distortion_params = "../lab5/calibration.xml";
    std::string detection_params = "../lab5/detector_params.yml";

    bool readOk = readCameraParameters(distortion_params, camMatrix, distCoeffs);
    if(!readOk) {
        cerr << "Invalid camera file" << endl;
        return 0;
    }

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    readOk = readDetectorParameters(detection_params, detectorParams);
    if(!readOk) {
        cerr << "Invalid detector parameters file" << endl;
        return 0;
    }

    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; // do corner refinement in markers

    cv::VideoCapture inputVideo;
    
    if (!inputVideo.open(0)) 
    {
        cerr << "Camera unavailable" << endl;
        return 1;
    }
    while (inputVideo.grab()) {
        cv::Mat image;
        cv::Mat output;
        // get image
        inputVideo.retrieve(image);
        image.copyTo(output);
        // do the work
        ProcessShot(image, output);
        cv::imshow("out", output);

        char key = (char) cv::waitKey(13);
        if (key == 27) // 
            break;
    }
}
