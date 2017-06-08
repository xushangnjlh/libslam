#include "common.h"

void feature_extraction_and_match(const Mat& img1, 
                                  const Mat& img2, 
                                  vector<cv::KeyPoint>& keyPoints1, 
                                  vector<cv::KeyPoint>& keyPoints2, 
                                  vector<cv::DMatch>& good_matches
                                 )
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
    
    cout << "good_matches = " << good_matches.size() << endl;
    for(cv::DMatch m:good_matches)
    {
        cout << "distance = " << (float)m.distance << endl;
    }

//     Mat matchImg;
//     cv::drawMatches(img1, keyPoints1, img2, keyPoints2, good_matches, matchImg);
//     cv::imshow("good_matches", matchImg);
//     cv::waitKey(0);
}

void pose_estimation_by_epipolar_constraint(vector<cv::KeyPoint>& keyPoints1, 
                                            vector<cv::KeyPoint>& keyPoints2, 
                                            vector<cv::DMatch>& good_matches,
                                            Mat& R, Mat& t
                                           )
{
    Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point2f> points1;
    vector<cv::Point2f> points2;
    
    for(int i=0; i<good_matches.size(); ++i)
    {
        points1.push_back(keyPoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keyPoints2[good_matches[i].trainIdx].pt);
    }
    
    // known pixel coordinates
    Mat F_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
    cout << "FundamentalMatrix = " << endl << F_matrix << endl;
    // known normalized coordinates or the cameraMatrix
    Mat E_matrix = cv::findEssentialMat(points1, points2, K, cv::LMEDS);
    cout << "EssentialMatrix = " << endl << E_matrix << endl;
    
    cv::recoverPose(E_matrix, points1, points2, K, R, t);
}

cv::Point2d pixel2cam(cv::Point2f& pt, Mat& K)
{
    return cv::Point2d(
        ( pt.x - K.at<double>(0,2) ) / K.at<double>(0,0),
        ( pt.y- K.at<double>(1,2) ) / K.at<double>(1,1)
    );
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
    
    Mat R, t;
    pose_estimation_by_epipolar_constraint(keyPoints1, keyPoints2, good_matches, R, t);
    cout << "Recovered pose from EssentialMatrix = " << endl
         << "R = " << endl
         << R << endl
         << "t = " << endl
         << t << endl;
         
    Mat t_x = ( cv::Mat_<double> ( 3,3 ) <<
                0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
                t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
                -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );

    cout<<"t^R="<<endl<<t_x*R<<endl;
    Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);


    //-- 验证对极约束
    for ( cv::DMatch m: good_matches )
    {
        cv::Point2d pt1 = pixel2cam ( keyPoints1[ m.queryIdx ].pt, K );
        Mat y1 = ( cv::Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
        cv::Point2d pt2 = pixel2cam ( keyPoints2[ m.trainIdx ].pt, K );
        Mat y2 = ( cv::Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
    return 0;
}