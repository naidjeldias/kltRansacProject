#include <iostream>
#include <opencv2/opencv.hpp>
#include "eightpoint.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

void bucketFeatureExtraction(Mat &image, Size block, std::vector<KeyPoint> &kpts) {

    //Ptr<FeatureDetector> detector = ORB::create();
    Ptr<FeatureDetector> detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);

    int maxFeature =50;

    int W = image.size().width;
    int H = image.size().height;

    //std::cout << "Image width : " << W << std::endl;
    //std::cout << "Image height : " << H << std::endl;

    int w = block.width;
    int h = block.height;

    //std::cout << "Patch width : " << w << std::endl;
    //std::cout << "Patch height : " << h << std::endl;

    Size subPixWinSize(3,3);
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);

    int cont = 0;

    for (int y = 0; y <= H - h; y += h){
        for(int x = 0; x <= W - w; x += w){
//            cont ++;
            //std::cout << "Patch: " << cont << std::endl;
            Mat imPatch = image(Rect (x, y, w, h)).clone();

            std::vector<KeyPoint> aux;
            detector -> detect(imPatch, aux);

//            std::cout << "Num features: " << aux.size()<< std::endl;

            //sort keypoints by response
            std::sort(aux.begin(), aux.end(), []( const KeyPoint &p1, const KeyPoint &p2){
                return p1.response > p2.response;
            });

            std::vector<Point2f> vec_;

            if(aux.size() >= maxFeature ){
                for (int i = 0; i < maxFeature; i++){
                    vec_.clear();
                    Point2f kpt_ =  aux.at(i).pt;

                    kpt_.x = kpt_.x + x;
                    kpt_.y = kpt_.y + y;

//                    std::cout << "Point 1: " << kpt_ << std::endl;

                    vec_.push_back(kpt_);

                    cornerSubPix(image, vec_, subPixWinSize, Size (-1,-1), termcrit);

                    KeyPoint kpt ;
                    kpt.pt = vec_.at(0);

//                    std::cout << "Point: " << kpt.pt << std::endl;

                    kpts.push_back(kpt);
                }
            }else if (aux.size() > 0 && aux.size() < maxFeature ){
                for (int i = 0; i < aux.size(); i++){
                    vec_.clear();
                    Point2f kpt_ =  aux.at(i).pt;

                    kpt_.x = kpt_.x + x;
                    kpt_.y = kpt_.y + y;

//                    std::cout << "Point 1: " << kpt_ << std::endl;

                    vec_.push_back(kpt_);

                    cornerSubPix(image, vec_, subPixWinSize, Size (-1,-1), termcrit);

                    KeyPoint kpt ;
                    kpt.pt = vec_.at(0);

//                    std::cout << "Point: " << kpt.pt << std::endl;

                    kpts.push_back(kpt);
                }
            }


//            rectangle(image,Rect (x, y, w, h),(0,0,255),1);
        }
    }
//    Mat rgb;
//    cvtColor(image, rgb, COLOR_GRAY2RGB);
//
//    drawKeypoints(rgb, kpts, rgb, (0,0,255), 1);
//
//    //std::cout << "Num features: " << kpts.size() << std::endl;
//
//    imshow("Image", rgb); // visualization
//    waitKey(0); // visualization

}

int main() {

    EightPoint eightPoint;

    std::vector<KeyPoint> kpts;
    std::vector<Point2f> kpts_l, kpts_r;
    Mat descpt_l, descpt_r, image_mathes;
    std::vector< DMatch > matches;

    //loading images
    std::string path = "/media/nigel/Dados/Documents/Projetos/CLionProjects/RANSAC/images/";
    Mat left_frame  = imread(path + "left_0.png");
    Mat right_frame = imread(path + "left_1.png");

    if(left_frame.channels() == 3)
        cvtColor(left_frame, left_frame, cv::COLOR_RGB2GRAY);
    if(right_frame.channels() == 3)
        cvtColor(right_frame, right_frame, cv::COLOR_RGB2GRAY);

    Ptr<FeatureDetector> detector = ORB::create(1000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);

    detector -> detect(left_frame, kpts);

    //convert vector of keypoints to vector of Point2f
    std::vector<Point2f> prevPoints, nextPoints;
    for (auto& kpt:kpts){
        prevPoints.push_back(kpt.pt);
    }

//    bucketFeatureExtraction(left_frame, Size (203,74), kpts_l);

    std::vector <Mat> left_pyr, right_pyr;

    Size win (21,21);
    int maxLevel = 4;

    buildOpticalFlowPyramid(left_frame, left_pyr, win, maxLevel, true);
    buildOpticalFlowPyramid(right_frame, right_pyr, win, maxLevel, true);

    Mat status;
    calcOpticalFlowPyrLK(left_pyr, right_pyr, prevPoints, nextPoints, status, noArray(), win, maxLevel);

    //get good features tracked
    for (int i = 0; i < status.rows; i++){
//        std::cout << "Status " << i << " : " << (int) status.at<uchar>(i) << std::endl;
        if ((int) status.at<uchar>(i) == 1){
            double dst = norm(Mat(prevPoints.at(i)), Mat(nextPoints.at(i)));
            DMatch match (i,i, dst);
            matches.push_back(match);
            Point2f kpt1, kpt2;
            kpt1 = prevPoints.at(i);
            kpt2 = nextPoints.at(i);

            kpts_l.push_back(kpt1);
            kpts_r.push_back(kpt2);
        }
    }

    std::cout << "Num points tracked left: " << kpts_l.size() << std::endl;
    std::cout << "Num points tracked right: " << kpts_r.size() << std::endl;

    std::vector<DMatch>     finalMatches;
    std::vector<bool>       inliers;
    cv::Mat                 fmat;
    cv::Mat                 epLinesLeft;
    cv::Mat                 epLinesRight;
    cv::Mat                 finalMatchImage;

    eightPoint.setRansacParameters(0.99, 8, 10, 0.9);
    fmat = eightPoint.ransacEightPointAlgorithm(kpts_l, kpts_r, finalMatches, inliers, true, 0);

    epLinesLeft     = eightPoint.drawEpLines(kpts_l, kpts_r, fmat, inliers, 0, left_frame);
    epLinesRight    = eightPoint.drawEpLines(kpts_l, kpts_r, fmat, inliers, 1, right_frame);

    image_mathes    = eightPoint.drawMatches_(left_frame, right_frame, kpts_l, kpts_r, matches);
    finalMatchImage = eightPoint.drawMatches_(left_frame, right_frame, kpts_l, kpts_r, finalMatches);

    imshow("matches", image_mathes);
    imshow("Final matches", finalMatchImage);
    imshow("Epipole lines left", epLinesLeft);
    imshow("Epipole lines Right", epLinesRight);

    waitKey(0);

    return 0;
}