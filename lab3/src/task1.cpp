#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat image, threshold_image;

void detect_goal(int threshold_level){
	threshold(image, threshold_image, threshold_level, 255, cv::THRESH_BINARY);
	imshow("threshold_image", threshold_image);
	std::vector<std::vector<cv::Point>> conturs;
	findContours(threshold_image, conturs, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
	cv::Mat goal_image;
	cv::cvtColor(image,goal_image, cv::COLOR_GRAY2RGB);
	int maxAreaInd = 0;
	double maxArea = 0;
	for (size_t i = 0; i < conturs.size(); i++)
	{
		double area = contourArea(conturs[i]);
		if (area > maxArea){
			maxArea = area;
			maxAreaInd = i;
		}
	}
	if (conturs.size() == 0) return;
	cv::Moments mnts = moments(conturs[maxAreaInd]);
	int x = mnts.m10 / mnts.m00;
	int y = mnts.m01 / mnts.m00;
	circle(goal_image, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), 3);
	imshow("goal_image", goal_image);
}

int main(int argc, char** argv){
	std::string image_file;
	if (argc>1) image_file= argv[1];
	else image_file = "../images/lab3/allababah/ig_2.jpg";
	image = cv::imread(image_file,0);
	imshow("origin", image);
	int treshold_value = 220;
	while (cv::waitKey(5) != 'q'){
		detect_goal(treshold_value);
		cv::createTrackbar("threshold", "threshold_image", &treshold_value, 255);
	}

}