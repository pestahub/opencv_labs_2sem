#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <vector>
#include <string>


#define _USE_MATH_DEFINES
#include <cmath>


using namespace cv;

void getColorHist(Mat& src, Mat& hist_image, std::vector<Mat>& hist_channels)
{
	std::vector<Mat> bgr_planes;
  split(src, bgr_planes);
  int histSize = 256;
  float range[] = { 0, 256 };
  const float* histRange[] = { range };
  bool uniform = true, accumulate = false;
  Mat b_hist, g_hist, r_hist;
  calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
  calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
  calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);
  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);
  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
  for (int i = 1; i < histSize; i++)
  {
      line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
          Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
          Scalar(255, 0, 0), 2, 8, 0);
      line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
          Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
          Scalar(0, 255, 0), 2, 8, 0);
      line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
          Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
          Scalar(0, 0, 255), 2, 8, 0);
  }
  hist_channels = { b_hist, g_hist, r_hist };

  hist_image = histImage;
}


int numOfTransitions(cv::MatIterator_<uint8_t>* it_block)
{
	int transNum = 0;
	for (int index = 1; index < 9; index++)
	{
		if (*it_block[index] > 0 && *it_block[index - 1] == 0) transNum++;
	}
	return transNum;
}


int numOfWhiteNeighbours(cv::MatIterator_<uint8_t>* it_block)	
{
	int whites = 0;
	for (int index = 0; index < 8; index++)
	{
		if (*it_block[index] > 0) whites++;
	}
	return whites;
}

bool steps(cv::Mat& img, int step)
{
	int cols = img.cols;
	Mat delmap = Mat::zeros(img.rows, img.cols, img.type());
	Mat buffer = Mat::ones(img.rows, img.cols, img.type());
	cv::MatIterator_<uint8_t> del_it = delmap.begin<uint8_t>() + cols + 1;
	cv::MatIterator_<uint8_t> it, end;
	bool imageChanged = false;

	for (it = img.begin<uint8_t>() + cols + 1, end = img.end<uint8_t>() - cols - 1; it != end; ++it, ++del_it)
	{
		cv::MatIterator_<uint8_t> pixels[] = { it - cols, it - cols + 1, it + 1, it + cols + 1, it + cols, it + cols - 1, it - 1, it - cols - 1, it - cols };
		if (*it == 0) continue; 
		int whites = numOfWhiteNeighbours(pixels);
		if (whites < 2 || whites > 6) continue; 

		int trans = numOfTransitions(pixels);
		if (trans != 1)  continue;
		
		if (step == 1)
		{
			if (*pixels[0] * *pixels[2] * *pixels[4]) continue;	
			if (*pixels[2] * *pixels[4] * *pixels[6]) continue;
		}
		else
		{
			if (*pixels[0] * *pixels[2] * *pixels[6]) continue;	
			if (*pixels[0] * *pixels[4] * *pixels[6]) continue;	
		}
		
		*del_it = (uint8_t)255;
		imageChanged = true;
	}
	img.setTo(0, delmap);

	return imageChanged;
}

void skeletize(cv::Mat& input, cv::Mat& output) 
{
	output = input.clone();
	bool changes = true;
	while (changes)
	{
		steps(output, 1); 
		changes = steps(output, 2);
	}

 }


void sortPolyline(std::vector<Vec4i> unsorted, std::vector<Point>& polyline, int threshold)
{
	int s = 2;
	polyline.push_back( Point(unsorted[s][2], unsorted[s][3]) );
	polyline.push_back( Point(unsorted[s][0], unsorted[s][1]) );
	Vec4i cur_line = unsorted[s];
	unsorted.erase(unsorted.begin()+s);
	while(unsorted.size() != 0)
	{
		Point cur_end = Point(cur_line[0], cur_line[1]);
		int minDist = INT_MAX;
		int minIndex = -1;
		for (int j = 0; j < unsorted.size(); j++)
		{
			Point vert_one = Point(unsorted[j][0], unsorted[j][1]);
			Point delta_one = vert_one - cur_end;
			Point vert_two = Point(unsorted[j][2], unsorted[j][3]);
			Point delta_two = vert_two - cur_end;
			int distance = min( (int)sqrt(delta_one.x* delta_one.x + delta_one.y* delta_one.y),
								(int)sqrt(delta_two.x* delta_two.x + delta_two.y* delta_two.y) );
			
			if (distance < minDist)
			{
				minDist = distance;
				minIndex = j;
			}
		}
		if (minIndex == -1) break;
		cur_line = unsorted[minIndex];
		polyline.push_back(Point(cur_line[0], cur_line[1]));
		polyline.push_back(Point(cur_line[2], cur_line[3]));
		unsorted.erase(unsorted.begin()+minIndex);
	}
}


void findAndDrawLines(cv::Mat& input, cv::Mat& output)
{
	std::vector<Vec4i> lines;

	HoughLinesP(input, lines, 1, CV_PI / 180, 45, 75, 45);
	std::vector<Point> uberline;
	std::vector<Vec4i> unsorted;
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(output, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 3);
		unsorted.push_back(l);
	}
	sortPolyline(unsorted, uberline, 100);
	polylines(output, uberline, false, Scalar(255, 0, 0), 2);

}


void findCoins(cv::Mat& input, cv::Mat& output, cv::Mat& nikMean, cv::Mat& copMean)
{
	medianBlur(input, input, 7);
	Mat greyOriginal = input.clone();
	cv::cvtColor(greyOriginal, greyOriginal, cv::COLOR_BGR2GRAY);
	Mat nikhist_img;
	Mat cophist_img;
	std::vector<Mat> hists_cop;
	std::vector<Mat> hists_nik;
	getColorHist(nikMean, nikhist_img, hists_nik);
	getColorHist(copMean, cophist_img, hists_cop);
	std::vector<Vec3f> detCircles;
	HoughCircles(greyOriginal, detCircles, HOUGH_GRADIENT, 1, 15, 150, 45);
	for (size_t i = 0; i < detCircles.size(); i++)
	{
		Vec3f circData = detCircles[i];
		circle(output, Point(circData[0], circData[1]), circData[2], Scalar(0, 0, 0), 3);
		cv::Mat roi = input(cv::Range(circData[1] - circData[2]*2/3, circData[1] + circData[2]*2/3 + 1), cv::Range(circData[0] - circData[2]*2/3, circData[0] + circData[2]*2/3 + 1));
		cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);
		std::vector < Mat > coinhist;
		Mat dbg_hist_image;
		getColorHist(roi, dbg_hist_image, coinhist);		
		double nik_mean = 0;
		double cop_mean = 0;
		for (int i = 0; i < 2; i++) 
		{
			cop_mean += compareHist(coinhist[i], hists_cop[i], HistCompMethods::HISTCMP_CORREL);
			nik_mean += compareHist(coinhist[i], hists_nik[i], HistCompMethods::HISTCMP_CORREL);
		}
		std::string caption;
		caption = nik_mean * (-1) < cop_mean ? "Nikkel" : "Copper";
		putText(output, caption, Point(circData[0]- circData[2] + 3, circData[1] + 6), FONT_HERSHEY_PLAIN, 1.1, Scalar(0, 0, 0), 2);
	}

}


void task1(){
	cv::Mat original = cv::imread("../images/lab6/1.jpg", ImreadModes::IMREAD_GRAYSCALE);
	Mat binary = original.clone();
	threshold(original, binary, 200, 255, THRESH_BINARY);	
	cv::Mat result = original.clone();
	skeletize(binary,result);
	imshow("Skeleton", result);
  cv::waitKey(-1);
}


void task2(){
  cv::Mat original = cv::imread("../images/lab6/2.jpg", ImreadModes::IMREAD_GRAYSCALE);
	Mat binary = original.clone();
	threshold(original, binary, 77, 255, THRESH_BINARY);
  
  morphologyEx( binary, binary, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)) );
  // imshow("binary", binary);
	cv::Mat skeleton = original.clone();
	skeletize(binary, skeleton);
	cv::Mat lines = original.clone();
	cv::cvtColor(lines, lines, cv::COLOR_GRAY2BGR);
  // imshow("skeleton", skeleton);
	findAndDrawLines(skeleton, lines);
	imshow("Coins", lines);
  cv::waitKey(-1);
}


void task3(){
  cv::Mat original = cv::imread("../images/lab6/monetki_0.jpg");
	Mat nikel = cv::imread("../images/lab6/nikel.jpg");
	Mat copper = cv::imread("../images/lab6/copper.jpg");
	cv::cvtColor(nikel, nikel, cv::COLOR_BGR2HSV);
	cv::cvtColor(copper, copper, cv::COLOR_BGR2HSV);
	cv::Mat roi = nikel(Rect(10, 10, 60, 60));
	cv::Mat copRoi = copper(Rect(25, 25, 60, 60));
	Mat coins = original.clone();
	medianBlur(nikel, nikel, 5);
	medianBlur(copper, copper, 5);
	findCoins(original, coins, nikel, copper);
	cv::imshow("Coins", coins);

  cv::waitKey(-1);
}


int main() {
    
  // task1();
  // task2();
  task3();
  return 0;
}