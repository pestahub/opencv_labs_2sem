#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

int main(int argc, char** argv){
    std::string image_file;
	if (argc>1) image_file= argv[1];
	else image_file = "../images/lab3/gk/gk.jpg";
	Mat image = cv::imread(image_file,IMREAD_COLOR);
    Mat gray;
    cvtColor(image, gray,CV_BGR2GRAY);
    Mat templ = imread("../images/lab3/gk/gk_tmplt.jpg", IMREAD_GRAYSCALE);
    Mat templ_tresh;
    Mat finded_temlp = image.clone();
    Mat draw = Mat::zeros(templ.size(), CV_8UC3);
    Mat gray_tresh, opened, closed, dilated;
    
    //ищу контур образца
    threshold(templ, templ_tresh, 200, 255, THRESH_BINARY);
    vector<vector<Point>> cnts_template;    
    findContours(templ_tresh, cnts_template, RETR_LIST, CHAIN_APPROX_NONE);
    drawContours(draw, cnts_template, 0, Scalar(255, 0, 0));

    // ищу все контуры 
    threshold(gray, gray_tresh, 230, 255, THRESH_BINARY);
    morphologyEx(gray_tresh, opened, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    morphologyEx(opened, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
    vector<vector<Point>> cnts;
    findContours(closed, cnts, RETR_LIST, CHAIN_APPROX_NONE);

    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i])>99999)
            continue;
        double diff = matchShapes(cnts[i], cnts_template[0], CV_CONTOURS_MATCH_I2, 0);
        if (diff < 1){
            drawContours(finded_temlp, cnts, i, Scalar(0, 255, 0), 5);  
        }
        else
            drawContours(finded_temlp, cnts, i, Scalar(0, 0, 255), 5);  
    }
    imshow("finded_temlp", finded_temlp);
    waitKey();
}