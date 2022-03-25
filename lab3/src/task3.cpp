#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

const int max_value_H = 360/2;
const int max_value = 255;

int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value, minArea=0;
Mat dilated_all;

void find_robots(Mat &src, vector<vector<Point>> &contours, Scalar &range_min, Scalar &range_max, int minArea=0){
    Mat thres_image;
    inRange(src, range_min, range_max, thres_image);
    Mat opened, closed, dilated;
    morphologyEx(thres_image, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(15, 15)));
    morphologyEx(closed, opened, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(10, 10)));
    add(dilated_all,opened, dilated_all);
    imshow("dilated", dilated_all);
    vector<vector<Point>> cnts;
    findContours(opened, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i]) < minArea)
            continue;
        contours.push_back(cnts[i]);
    }
}

int main(int argc, char** argv){
	namedWindow("thres_image");
	string image_file;
	Mat image, thres_image;
	if (argc>1) image_file= argv[1];
	else image_file = "../images/lab3/roboti/roi_robotov.jpg";
	image = imread(image_file);
	imshow("image", image);
    Mat bgrImg, hsvImg, gray, imgLamp;
    imgLamp = image.clone();
    gray = imread("image", 0);
    
    threshold(gray, thres_image, 247, 255, THRESH_BINARY);
    imshow("thres_image", thres_image);

    vector<vector<Point>> cnts;
    findContours(thres_image, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    double maxArea = 0;
    int x = 0;
    int y = 0;
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i]) > maxArea){
            Moments mnts = moments(cnts[i]);
            x = mnts.m10 / mnts.m00;
            y = mnts.m01 / mnts.m00;
        }
    }
    // закрыл красную лампу 
    circle(imgLamp, Point(x, y-10), 25, Scalar(0, 0, 0), 45);

    int low_H = 43;
	int high_H = 93;
    int low_S = 35;
	int high_S = 255;
    int low_V = 151;
	int high_V = 255;
    cv::createTrackbar("low_H", "thres_image", &low_H, 180);
	cv::createTrackbar("high_H", "thres_image", &high_H, 180);
    cv::createTrackbar("low_S", "thres_image", &low_S, 255);
	cv::createTrackbar("high_S", "thres_image", &high_S, 255);
    cv::createTrackbar("low_V", "thres_image", &low_V, 255);
	cv::createTrackbar("high_V", "thres_image", &high_V, 255);
    Mat draw = image.clone();
    while (waitKey(5) != 'w')
    {
        Scalar red_low_0(0, 150, 80); 
        Scalar red_high_0(40, 255, 255);
        Scalar red_low_1(150, 150, 80);
        Scalar red_high_1(180, 224, 255);

        Scalar green_low(88, 133, 25);
        Scalar green_high(130, 255, 255);

        Scalar blue_low(low_H, low_S, low_V);
        Scalar blue_high(high_H, high_S, high_V);

        Scalar blue_clr(255, 0, 0);
        Scalar green_clr(0, 255, 0);
        Scalar red_clr(0, 0, 255);
        cvtColor(imgLamp, bgrImg, COLOR_YUV2BGR);
        cvtColor(bgrImg, hsvImg, COLOR_BGR2HSV);
        
        dilated_all = Mat::zeros(hsvImg.size(), CV_8UC1);
        vector<vector<Point>> contours_green;
        find_robots(hsvImg, contours_green, green_low, green_high);
        vector<vector<Point>> contours_blue;
        find_robots(hsvImg, contours_blue, blue_low, blue_high);
        vector<vector<Point>> contours_red;
        find_robots(hsvImg, contours_red, red_low_0, red_high_0);
        find_robots(hsvImg, contours_red, red_low_1, red_high_1);

        draw = image.clone();
        vector<Point> greenRobotsCenters;
        for (size_t i = 0; i < contours_green.size(); i++)
        {
            Moments mnts = moments(contours_green[i]);
            int centerX = mnts.m10 / mnts.m00;
            int centerY = mnts.m01 / mnts.m00;
            greenRobotsCenters.push_back(Point(centerX, centerY));
            drawContours(draw, contours_green, i, green_clr, 3);
        }
        vector<Point> blueRobotsCenters;
        for (size_t i = 0; i < contours_blue.size(); i++)
        {
            Moments mnts = moments(contours_blue[i]);
            int centerX = mnts.m10 / mnts.m00;
            int centerY = mnts.m01 / mnts.m00;
            blueRobotsCenters.push_back(Point(centerX, centerY));
            drawContours(draw, contours_blue, i, blue_clr, 3);
        }
        vector<Point> redRobotsCenters;
        for (size_t i = 0; i < contours_red.size(); i++)
        {
            Moments mnts = moments(contours_red[i]);
            int centerX = mnts.m10 / mnts.m00;
            int centerY = mnts.m01 / mnts.m00;
            redRobotsCenters.push_back(Point(centerX, centerY));
            drawContours(draw, contours_red, i, red_clr, 3);
        }

        imshow("draw", draw);



        circle(draw, Point(x, y), 3, Scalar(0, 0, 0), 3);
    
        float minDistance = 0;

        Point closestGreen;
        for (size_t i = 0; i < greenRobotsCenters.size(); i++)
        {
            float distance = sqrt((greenRobotsCenters[i].x - x) * (greenRobotsCenters[i].x - x) + 
                                (greenRobotsCenters[i].y - y) * (greenRobotsCenters[i].y - y));
            if ((i == 0) || (distance < minDistance)){
                minDistance = distance;
                closestGreen.x = greenRobotsCenters[i].x;
                closestGreen.y = greenRobotsCenters[i].y;
            }
        }
        circle(draw, closestGreen, 3, Scalar(255, 255, 255), 3);

        minDistance = 0;
        
        Point closestBlue;
        for (size_t i = 0; i < blueRobotsCenters.size(); i++)
        {
            float distance = sqrt((blueRobotsCenters[i].x - x) * (blueRobotsCenters[i].x - x) + 
                                (blueRobotsCenters[i].y - y) * (blueRobotsCenters[i].y - y));
            if ((i == 0) || (distance < minDistance)){
                minDistance = distance;
                closestBlue.x = blueRobotsCenters[i].x;
                closestBlue.y = blueRobotsCenters[i].y;
            }
        }
        circle(draw, closestBlue, 3, Scalar(255, 255, 255), 3);

        minDistance = 0;
        Point closestRed;
        for (size_t i = 0; i < redRobotsCenters.size(); i++)
        {
            float distance = sqrt((redRobotsCenters[i].x - x) * (redRobotsCenters[i].x - x) + 
                                (redRobotsCenters[i].y - y) * (redRobotsCenters[i].y - y));
            if ((i == 0) || (distance < minDistance)){
                minDistance = distance;
                closestRed.x = redRobotsCenters[i].x;
                closestRed.y = redRobotsCenters[i].y;
            }
        }
        circle(draw, closestRed, 3, Scalar(255, 255, 255), 3);
        
        imshow("closest", draw);
    }
    
    
}
