#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "sstream"
#include "iostream"
#include "string"
#include "stdio.h"
#include "stdlib.h"
#include "fstream"
#include "unistd.h"

using namespace cv;
using namespace std;

double lightThresh=20.0;

int main(int argh, char* argv[])
{
    cv::VideoCapture cap(0);
    cv::Mat frame, fgMaskMOG2;
    cap >> frame;
    int height = frame.size().height;
    int width  = frame.size().width;
    Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(20, 16,true);
    VideoWriter writer("OutVideo.avi", VideoWriter::fourcc('X','V','I','D'), 15.0, Size(width,height));
    if( !cap.isOpened() ){ printf("capture not opened\n"); return -1; };

    while(true)
    {
        cap >> frame; // get a new frame from camera
        pMOG2 -> apply(frame, fgMaskMOG2, 0.001);
        cv::Scalar pixcelSum = sum(fgMaskMOG2);

        if( pixcelSum.val[0] / fgMaskMOG2.total() > lightThresh ){
            cout << "pixel right = " << pixcelSum.val[0] / fgMaskMOG2.total() << endl;
            writer << frame;
        }

        //key event  waitKey is necessary
        int key = cv::waitKey(10);
        if(key == 113)//q key
        {
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}
