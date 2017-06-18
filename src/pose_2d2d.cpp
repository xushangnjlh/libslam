#include "common.h"
// #include "opencv2/xfeatures2d/nonfree.hpp"
#include "tic_toc.h"

void feature_extraction_and_match(const Mat& img1, 
                                  const Mat& img2, 
                                  vector<cv::KeyPoint>& keyPoints1, 
                                  vector<cv::KeyPoint>& keyPoints2, 
                                  vector<cv::DMatch>& good_matches
                                 );

void pose_estimation_by_epipolar_constraint(vector<cv::KeyPoint>& keyPoints1, 
                                            vector<cv::KeyPoint>& keyPoints2, 
                                            vector<cv::DMatch>& good_matches,
                                            Mat& R, Mat& t
                                           );

void pose_estimation_by_PnP(const vector<cv::Point3f>& points_3d, 
                            const vector<cv::Point2f>& points_2d,
                            const Mat& K,
                            Mat& R, Mat& t 
                           );

void triangulation(const vector<cv::KeyPoint>& keyPoints1, 
                   const vector<cv::KeyPoint>& keyPoints2,
                   const vector<cv::DMatch>& matches,
                   const Mat& R, const Mat& t,
                   vector<cv::Point3f>& points
                  );

inline cv::Point2f pixel2cam(const cv::Point2d& pt, const Mat& K)
{
    return cv::Point2f(
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
    
    Mat img1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
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
         
    vector<cv::Point3f> points;
    triangulation(keyPoints1, keyPoints2, good_matches, R, t, points);
    
    Mat K = ( cv::Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
//     for ( int i=0; i<good_matches.size(); i++ )
//     {
//         cv::Point2d pt1_cam = pixel2cam( keyPoints1[ good_matches[i].queryIdx ].pt, K );
//         cv::Point2d pt1_cam_3d(
//             points[i].x/points[i].z, 
//             points[i].y/points[i].z 
//         );
//         
//         cout<<"point in the first camera frame: "<<pt1_cam<<endl;
//         cout<<"point projected from 3D "<<pt1_cam_3d<<", d="<<points[i].z<<endl;
//         
//         // 第二个图
//         cv::Point2f pt2_cam = pixel2cam( keyPoints2[ good_matches[i].trainIdx ].pt, K );
//         Mat pt2_trans = R*( cv::Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z ) + t;
//         pt2_trans /= pt2_trans.at<double>(2,0);
//         cout<<"point in the second camera frame: "<<pt2_cam<<endl;
//         cout<<"point reprojected from second frame: "<<pt2_trans.t()<<endl;
//         cout<<endl;
//     }
    
    vector<cv::Point2f> points_2d;
    for(int i=0; i<good_matches.size(); ++i)
    {
        points_2d.push_back(keyPoints2[good_matches[i].trainIdx].pt);
    }
    
    TicToc tic;
    Mat R_pnp, r_pnp, t_pnp;
    // r_pnp is the axis-angle vector; while R_pnp is the rotation matrix
    pose_estimation_by_PnP(points, points_2d, K, r_pnp, t_pnp);
    cout << "Time cost = " << tic.toc() << " mseconds." << endl;
    cv::Rodrigues(r_pnp, R_pnp);
    cout << "R_pnp = " << endl
         << R_pnp << endl;
    cout << "t_pnp = " << endl
         << t_pnp << endl;
    
         
//     Mat t_x = ( cv::Mat_<double> ( 3,3 ) <<
//                 0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
//                 t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
//                 -t.at<double> ( 1.0 ),     t.at<double> ( 0,0 ),      0 );
// 
//     cout<<"t^R="<<endl<<t_x*R<<endl;
//     Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

//     for ( cv::DMatch m: good_matches )
//     {
//         cv::Point2d pt1 = pixel2cam ( keyPoints1[ m.queryIdx ].pt, K );
//         Mat y1 = ( cv::Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
//         cv::Point2d pt2 = pixel2cam ( keyPoints2[ m.trainIdx ].pt, K );
//         Mat y2 = ( cv::Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
//         Mat d = y2.t() * t_x * R * y1;
//         cout << "epipolar constraint = " << d << endl;
//     }
    return 0;
}

void feature_extraction_and_match(const Mat& img1, 
                                  const Mat& img2, 
                                  vector<cv::KeyPoint>& keyPoints1, 
                                  vector<cv::KeyPoint>& keyPoints2, 
                                  vector<cv::DMatch>& good_matches
                                 )
{
//     cv::Ptr<cv::xfeatures2d::SURF> orb = cv::xfeatures2d::SURF::create();
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detect(img1, keyPoints1);
    orb->detect(img2, keyPoints2);
    
    Mat descriptor1, descriptor2;
    orb->compute(img1, keyPoints1, descriptor1);
    orb->compute(img2, keyPoints2, descriptor2);
    
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<cv::DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    
//     cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create();
//     cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
//     cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
//     
//     orb->detect(img1, keyPoints1);
//     orb->detect(img2, keyPoints2);
//     
//     Mat descriptor1, descriptor2;
//     descriptor->compute(img1, keyPoints1, descriptor1);
//     descriptor->compute(img2, keyPoints2, descriptor2);
//     vector<cv::DMatch> matches;
//     matcher->match(descriptor1, descriptor2, matches);

    // find the min distance of in all matches
    double min_distance = 10000.0/*, max_distance = 0.0*/;
    for(int i=0; i<descriptor1.rows; i++)
    {
        if ( min_distance > matches[i].distance )
            min_distance = matches[i].distance;
    }
    
//     cout << "min_distance = " << min_distance << endl;
//     cout << "max_distance = " << max_distance << endl;
    for(int i=0; i<descriptor1.rows; i++)
    {
        if ( matches[i].distance <= max(min_distance*2 , 30.0) )
            good_matches.push_back(matches[i]);
    }
    
    cout << "good_matches = " << good_matches.size() << endl;
//     for(cv::DMatch m:good_matches)
//     {
//         cout << "distance = " << (float)m.distance << endl;
//     }

    Mat matchImg;
    cv::drawMatches(img1, keyPoints1, img2, keyPoints2, good_matches, matchImg);
    cv::imshow("good_matches", matchImg);
    cv::waitKey(0);
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
    
    // known pixel coordinates6t 
    Mat F_matrix = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC);
//     cout << "FundamentalMatrix = " << endl << F_matrix << endl;
    // known normalized coordinates or the cameraMatrix
    
    Mat E_matrix = cv::findEssentialMat(points1, points2, K, cv::RANSAC);
//     cout << "EssentialMatrix = " << endl << E_matrix << endl;
    
    cv::recoverPose(E_matrix, points1, points2, K, R, t);
    
}

void triangulation(const vector< cv::KeyPoint >& keyPoints1, 
                   const vector< cv::KeyPoint >& keyPoints2, 
                   const vector< cv::DMatch >& good_matches, 
                   const Mat& R, const Mat& t, 
                   vector<cv::Point3f>& points)
{
    Mat T1 = (cv::Mat_<double>(3,4) << 
        1,0,0,0,
        0,1,0,0,
        0,0,1,0
    );
    Mat T2 = (cv::Mat_<double>(3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    
    Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<cv::Point2f> points1_cam, points2_cam;
    for(cv::DMatch m:good_matches)
    {
        points1_cam.push_back( pixel2cam(keyPoints1[m.queryIdx].pt, K) );
        points2_cam.push_back( pixel2cam(keyPoints2[m.trainIdx].pt, K) );
    }
    
    
    // the output is homogeneous coordinates of 3D points
    Mat points4D;
    cv::triangulatePoints(T1, T2, points1_cam, points2_cam, points4D);
    
    for(int i=0; i<points4D.cols; ++i)
    {
        Mat x = points4D.col(i);
        x /= x.at<float>(3,0);
        cv::Point3d p(
            x.at<float>(0,0),
            x.at<float>(1,0),
            x.at<float>(2,0)
        );
        points.push_back(p);
//         cout << points[i] << endl;
    }
//     cout << "debug" << endl;
}

void pose_estimation_by_PnP(const vector<cv::Point3f>& points_3d, 
                            const vector<cv::Point2f>& points_2d, 
                            const Mat& K, 
                            Mat& R, Mat& t
                           )
{
    // EPnP is the fast compared with UPnP and ITERATIVE method
    cv::solvePnP(points_3d, points_2d, K, Mat(), R, t, false, cv::SOLVEPNP_EPNP);
}

