#include "common.h"

void feature_extraction_and_match(const Mat& img1, 
                                  const Mat& img2, 
                                  vector<cv::KeyPoint>& keyPoints1, 
                                  vector<cv::KeyPoint>& keyPoints2, 
                                  vector<cv::DMatch>& good_matches)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detect(img1, keyPoints1);
    orb->detect(img2, keyPoints2);
    
    Mat descriptor1, descriptor2;
    orb->compute(img1, keyPoints1, descriptor1);
    orb->compute(img2, keyPoints2, descriptor2);
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);

    // find the min distance of in all matches
    double min_distance = 10000;
    for(int i=0; i<descriptor1.rows; i++)
    {
        if ( min_distance > matches[i].distance )
            min_distance = matches[i].distance;
    }
    for(int i=0; i<descriptor1.rows; i++)
    {
        if ( matches[i].distance < max(min_distance*2, 30.0) )
            good_matches.push_back(matches[i]);
    }

    Mat matchImg;
    cv::drawMatches(img1, keyPoints1, img2, keyPoints2, good_matches, matchImg);
    cv::imshow("good_matches", matchImg);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    if(argc!=3)
    {
        cout << "Use:./pose2d_2d image1 image2" << endl;
        return 1;
    }
    Mat img1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    vector<cv::KeyPoint> keyPoints1, keyPoints2;
    vector<cv::DMatch> good_matches;
    feature_extraction_and_match(img1, img2, keyPoints1, keyPoints2, good_matches);
    return 0;
}