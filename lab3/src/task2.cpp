#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char** argv){
	cv::namedWindow("treshold_image");
	std::string image_file;
	cv::Mat image, treshold_image;
	if (argc>1) image_file= argv[1];
	else image_file = "../images/lab3/teplovizor/ntcs_quest_measurement.png";
	image = cv::imread(image_file);
	

	int low_H = 0;
	int high_H = 30;
	cv::createTrackbar("Low H", "treshold_image", &low_H, 180);
	cv::createTrackbar("High H", "treshold_image", &high_H, 180);

	while(cv::waitKey(5) != 'q'){
		cv::Mat bgr_img, hsv_img;
		cvtColor(image, bgr_img, cv::COLOR_YUV2BGR);
		cvtColor(bgr_img, hsv_img, cv::COLOR_BGR2HSV);
		inRange(hsv_img, cv::Scalar(low_H, 0, 0), cv::Scalar(high_H, 255, 255), treshold_image);
		cv::Mat opened, closed, dilated;
    morphologyEx(treshold_image, opened, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
    morphologyEx(opened, closed, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)));
    morphologyEx(closed, dilated, cv::MORPH_DILATE, getStructuringElement(cv::MORPH_RECT, cv::Size(15,15)));
		imshow("treshold_image", dilated);
		std::vector<std::vector<cv::Point>> cnts;
		findContours(dilated, cnts, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
		cv::Mat goal_image = image.clone();
		for (size_t i = 0; i < cnts.size(); i++)
		{
			cv::Moments mnts = moments(cnts[i]);
			int x = mnts.m10 / mnts.m00;
			int y = mnts.m01 / mnts.m00;
			circle(goal_image, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), 3);
			
		}
		imshow("goal_image", goal_image);
	}
	
}