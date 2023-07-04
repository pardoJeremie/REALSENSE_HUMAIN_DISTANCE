#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>
//* start new code
#include <librealsense2/rs.hpp>
#include <thread>
#include <math.h>

#define color_w_const 424
#define color_h_const 240
#define depth_w_const 480

#define deth_h_const 270
//* end new code

using namespace std;
//-----------------------------------------------------------------------------------------------------------------------
struct KeyPoint
{
    cv::Point2f p;
    float prob;
};
//-----------------------------------------------------------------------------------------------------------------------
static float draw_pose_and_return_distance(const cv::Mat& image, const std::vector<KeyPoint>& keypoints,rs2::depth_frame &depthFrame) //* changed the fonction name and added the depth frame and a return value
{
    int w = image.cols;
    int h = image.rows;
    // start new code
    cv::Mat bgr = image.clone();//* clone the image to have the correct pixel color
    std::vector<float> distanceVec; //* vector of infered point/joints distance
    // end new code
    // draw bone
    static const int joint_pairs[16][2] = {
        {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
    };
    for (int i = 0; i < 16; i++)
    {
        const KeyPoint& p1 = keypoints[joint_pairs[i][0]];
        const KeyPoint& p2 = keypoints[joint_pairs[i][1]];
        if (p1.prob < 0.2f || p2.prob < 0.2f)
            continue;
        cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
    }
    // draw joint
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const KeyPoint& keypoint = keypoints[i];

        if (keypoint.prob < 0.2f)
            continue;

        //* start new code
        if(0 <= (int)keypoint.p.x && (int)keypoint.p.x < w && 0 <= (int)keypoint.p.y && (int)keypoint.p.y < h) // remove out of pound point to not have bad memory access error
        {
            cv::circle(image, keypoint.p, 5, cv::Scalar(0, 255, 0), 1);
            cv::circle(image, keypoint.p, 4, bgr.at<cv::Vec3b >(keypoint.p) , -1);

            distanceVec.push_back(depthFrame.get_distance(keypoint.p.x,keypoint.p.y));//* add the distance of the point/joint in meter to distanceVec

            std::cout << depthFrame.get_distance(keypoint.p.x,keypoint.p.y)<< "; ";
        }
        //* end new code
    }
    std::cout << std::endl;

    //* start new code
    if (distanceVec.empty())
        return -1; //* return -1 when all infered points/joints have a proba < 0.2

    std::sort(distanceVec.begin(),distanceVec.end());// sort the vector

    return distanceVec.at(round(distanceVec.size()/2)); //* return the mean distance
    //* end new code
}
//-----------------------------------------------------------------------------------------------------------------------
int runpose(cv::Mat& roi, ncnn::Net &posenet, int pose_size_width, int pose_size_height, std::vector<KeyPoint>& keypoints,float x1, float y1)
{
    int w = roi.cols;
    int h = roi.rows;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(roi.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 roi.cols, roi.rows, pose_size_width, pose_size_height);
    //数据预处理
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = posenet.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("hybridsequential0_conv7_fwd", out);
    keypoints.clear();
    for (int p = 0; p < out.c; p++)
    {
        const ncnn::Mat m = out.channel(p);

        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < out.h; y++)
        {
            const float* ptr = m.row(y);
            for (int x = 0; x < out.w; x++)
            {
                float prob = ptr[x];
                if (prob > max_prob)
                {
                    max_prob = prob;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        KeyPoint keypoint;
        keypoint.p = cv::Point2f(max_x * w / (float)out.w+x1, max_y * h / (float)out.h+y1);
        keypoint.prob = max_prob;
        keypoints.push_back(keypoint);
    }
    return 0;
}
//-----------------------------------------------------------------------------------------------------------------------
int demo(cv::Mat& image, ncnn::Net &detectornet, int detector_size_width, int detector_size_height,
         ncnn::Net &posenet, int pose_size_width, int pose_size_height, std::thread & thread_depth,cv::Mat matdepth,rs2::frameset &frameset) //* this fonction was modified to accept depth data
{
    cv::Mat bgr = image.clone();
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    float returnedDistance;//* will contain the distance of the infered humains

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                 bgr.cols, bgr.rows, detector_size_width, detector_size_height);

    //数据预处理
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = detectornet.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);

    thread_depth.join(); //* depth and color frames alignement thread
    rs2::depth_frame depthFrame = frameset.get_depth_frame(); //* get the depth aligned frameset to infere the distance frome the camera of the detected humain (!! must be done after alignement of the color and depth frames)

    for (int i = 0; i < out.h; i++)
    {
        float x1, y1, x2, y2;
        float pw,ph,cx,cy;
        const float* values = out.row(i);

        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        pw = x2-x1;
        ph = y2-y1;
        cx = x1+0.5*pw;
        cy = y1+0.5*ph;

        x1 = cx - 0.7*pw;
        y1 = cy - 0.6*ph;
        x2 = cx + 0.7*pw;
        y2 = cy + 0.6*ph;

        //处理坐标越界问题
        if(x1<0) x1=0;
        if(y1<0) y1=0;
        if(x2<0) x2=0;
        if(y2<0) y2=0;

        if(x1>img_w) x1=img_w;
        if(y1>img_h) y1=img_h;
        if(x2>img_w) x2=img_w;
        if(y2>img_h) y2=img_h;
        //截取人体ROI
        //printf("x1:%f y1:%f x2:%f y2:%f\n",x1,y1,x2,y2);
        cv::Mat roi;
        roi = bgr(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
        std::vector<KeyPoint> keypoints;
        runpose(roi, posenet, pose_size_width, pose_size_height,keypoints, x1, y1);

        returnedDistance = draw_pose_and_return_distance(matdepth, keypoints, depthFrame); //* previously named draw_pose, the fonction now take for input the depth cv:Mat ann the depth frame. This fonction now draw the  skeleton on the depth cv::Mat and return the

        cv::rectangle (image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255), 2, 8, 0);

        //* start new code
        //* add a text giving the distance to the inference box if the returned distance is not -1
        if( returnedDistance != -1)
        {
            char text[256];
            sprintf(text, "Distance = %.1f m",  returnedDistance);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            cv::putText(image, text, cv::Point(x1, y1 + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }
        //* end new code
    }


    return 0;
}

//* start new code

void align_get_mat_depth(rs2::frameset * pframeset, cv::Mat* pmatdepth, rs2::colorizer * pcolor_map, rs2::align * palign_to_color){//* using pointer because thread don't work with reference
    //* align depth frame to color frame
    *pframeset = palign_to_color->process(*pframeset);

    //* get detph frame
    rs2::frame frame_depth = pframeset->get_depth_frame().apply_filter(*pcolor_map);

    //* Create OpenCV matrix of size (w,h) from the colorized depth data after the depth mat is align with the color frame
    *pmatdepth = cv::Mat(cv::Size(color_w_const, color_h_const), CV_8UC3, (void*)frame_depth.get_data(), cv::Mat::AUTO_STEP); //* old data is automatiquely release in the operator=()
}
//* end new code

//-----------------------------------------------------------------------------------------------------------------------
int main(int argc,char ** argv)
{
    float f;
    float FPS[16];
    int i;
    int Fcnt=0;
    chrono::steady_clock::time_point Tbegin, Tend;

    for(i=0;i<16;i++) FPS[i]=0.0;

    //定义检测器
    ncnn::Net detectornet;
    detectornet.load_param("./person_detector.param");
    detectornet.load_model("./person_detector.bin");
    int detector_size_width  =  320;
    int detector_size_height = 320;

    //定义人体姿态关键点预测器
    ncnn::Net posenet;
    posenet.load_param("./Ultralight-Nano-SimplePose.param");
    posenet.load_model("./Ultralight-Nano-SimplePose.bin");
    int pose_size_width  =  192;
    int pose_size_height =  256;


    //* start new code
    //* Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;

    //* Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    //* Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_COLOR, color_w_const, color_h_const, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, depth_w_const, deth_h_const, RS2_FORMAT_Z16, 30);


    //* Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

    //* align objects to aligne depth frame to color frame.
    rs2::align align_to_color(RS2_STREAM_COLOR);
    //* declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    rs2::frameset frames;
    rs2::frame frame_color;

    //* init mat for concate image;
    cv::Mat hconcate;
    cv::Mat matdepth;

    //* Camera warmup - dropping several first frames to let auto-exposure stabilize
    for(int i = 0; i < 30; i++)
    {
        //* Wait for all configured streams to produce a frame
        frames = pipe.wait_for_frames();
    }
    //* end new code

    Tbegin = chrono::steady_clock::now();

    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
    while(1){

        //* start new code
        //* get the realsense new frames (depth + color)
        frames = pipe.wait_for_frames();

        //* extract only the color frame
        frame_color = frames.get_color_frame();

        //* thread that translate an realsense depth frame into an opencv mat and aligne it to a color frame
        std::thread thread_depth(align_get_mat_depth, &frames /*when the thread is .joint, frames will be an aligned frameset*/, &matdepth, &color_map, &align_to_color);

        //* translate an realsense color frame into an opencv mat
        cv::Mat color(cv::Size(color_w_const, color_h_const), CV_8UC3, (void*)frame_color.get_data(), cv::Mat::AUTO_STEP);
        //* end new code

        if(demo(color, detectornet, detector_size_width, detector_size_height, posenet, pose_size_width,pose_size_height, thread_depth, matdepth, frames)) //* this fonction was modified to accept depth data , this is where the inference is down
            return 1;

        Tend = chrono::steady_clock::now();
        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();

        Tbegin = chrono::steady_clock::now();

        FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        putText(color, cv::format("FPS %0.2f",f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));

        //* start new code
        //* concatenate depth and color cv:Mat
        std::vector<cv::Mat> framesvec = {matdepth,color};
        cv::hconcat(framesvec, hconcate);

        //* show result
        cv::imshow("humaind distance detection",hconcate);
        //* end new code

        char esc = cv::waitKey(5);
        if(esc == 27) break;

    }

    return 0;
}
