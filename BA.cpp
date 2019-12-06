#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <string>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
namespace py = boost::python;
namespace np = boost::python::numpy;
using namespace std;

class BA{
    public:
        string word;
        BA(py::list intrinsic, py::list extrinsic, py::list correspond_struct_idx, py::list key_points_for_all, py::list structure);
        
        void save(py::object path, py::list rotation, py::list motions, py::list colors);
    private:
        Mat c_intrinsic;
        vector<Mat> c_extrinsics;
        vector<vector<int> > c_correspond_struct_idx;
        vector<vector<KeyPoint> > c_key_points_for_all;
        vector<Point3d> c_structure;
        void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3d>& structure, vector<Vec3b>& colors);
    
        Mat ConvertNDArrayToMat(const np::ndarray& ndarr);
//        Mat ConvertNDArrayToVec3b(const np::ndarray& ndarr);
        void bundle_adjustment(
                Mat& intrinsic,
                vector<Mat>& extrinsics,
                vector<vector<int> >& correspond_struct_idx,
                vector<vector<KeyPoint> >& key_points_for_all,
                vector<Point3d>& structure);
        struct ReprojectCost
        {
            cv::Point2d observation;

            ReprojectCost(cv::Point2d& observation)
                : observation(observation)
            {
            }

            template <typename T>
            bool operator()(const T* const intrinsic, const T* const extrinsic, const T* const pos3d, T* residuals) const
            {
                const T* r = extrinsic;
                const T* t = &extrinsic[3];

                T pos_proj[3];
                ceres::AngleAxisRotatePoint(r, pos3d, pos_proj);

                // Apply the camera translation
                pos_proj[0] += t[0];
                pos_proj[1] += t[1];
                pos_proj[2] += t[2];

                const T x = pos_proj[0] / pos_proj[2];
                const T y = pos_proj[1] / pos_proj[2];

                const T fx = intrinsic[0];
                const T fy = intrinsic[1];
                const T cx = intrinsic[2];
                const T cy = intrinsic[3];

                // Apply intrinsic
                const T u = fx * x + cx;
                const T v = fy * y + cy;

                residuals[0] = u - T(observation.x);
                residuals[1] = v - T(observation.y);

                return true;
            }
};

};

// example of passing python list as argument (to constructor)
BA::BA(py::list intrinsic, py::list extrinsic, py::list correspond_struct_idx, py::list key_points_for_all, py::list structure)
{
    Py_Initialize();
    np::initialize();
//    Mat token;
    if (len(intrinsic) != 1)
        cout<<"Error, more than one intrinsic."<<endl;
    for (int i = 0; i < len(intrinsic) ; i++){
        np::ndarray token = py::extract<np::ndarray>(intrinsic[i]);
        c_intrinsic = ConvertNDArrayToMat(token);
//        cout<<"c_intrinsic"<<c_intrinsic<<endl;
    }
    for (int i = 0; i < len(extrinsic) ; i++){
        np::ndarray token = py::extract<np::ndarray>(extrinsic[i]);
        c_extrinsics.push_back(ConvertNDArrayToMat(token));
//        cout<<"c_extrinsics"<<c_extrinsics.back()<<endl;
    }
    for (int i = 0; i < len(correspond_struct_idx) ; i++){
        np::ndarray token1 = py::extract<np::ndarray>(correspond_struct_idx[i]);
        vector<int> token2;
        for(int j = 0; j<len(token1); j++)
        {
            token2.push_back(py::extract<int>(token1[j].attr("__int__")()));
//            cout<<"correspond_struct_idx"<<token2.back()<<endl;
        }
        
        c_correspond_struct_idx.push_back(token2);
    }
    for (int i = 0; i < len(key_points_for_all) ; i++){
        np::ndarray token1 = py::extract<np::ndarray>(key_points_for_all[i]);
        vector<KeyPoint> token2;
        const Py_intptr_t* shape = token1.get_shape();
        for(int j = 0; j<shape[0]; j++)
//            for(int k = 0; k<shape[1]; k++)
            {
                float x = py::extract<float>(token1[j][0]);
                float y = py::extract<float>(token1[j][1]);
                float size = py::extract<float>(token1[j][2]);
                token2.push_back(KeyPoint(x,y,size));
//                cout<<"key_points_for_all"<<token2.back().pt<<endl;
            }
        
        c_key_points_for_all.push_back(token2);
    }
    for (int i = 0; i < len(structure) ; i++){
        np::ndarray token1 = py::extract<np::ndarray>(structure[i]);
        
        double x = py::extract<double>(token1[0]);
        double y = py::extract<double>(token1[1]);
        double z = py::extract<double>(token1[2]);
        c_structure.push_back(Point3d(x,y,z));
//        cout<<"correspond_struct_idx"<<c_structure.back()<<endl;
    }
    cout<<"old structure:"<<c_structure<<endl;
    bundle_adjustment(c_intrinsic, c_extrinsics, c_correspond_struct_idx, c_key_points_for_all, c_structure);
    
    cout<<"new structure:"<<c_structure<<endl;
//    this -> word = w;
}


Mat BA::ConvertNDArrayToMat(const np::ndarray& ndarr) {

    const Py_intptr_t* shape = ndarr.get_shape();

    // variables for creating Mat object
    int rows = shape[0];
    int cols = shape[1];
//    int channel = shape[2];
//    int depth;


//    if (!strcmp(dtype_str, "uint8")) {
//        depth = CV_8U;
//    }
//    else {
//        std::cout << "wrong dtype error" << std::endl;
//        return cv::Mat();
//    }

//    int type = CV_MAKETYPE(depth, channel); // CV_8UC3

//    cv::Mat mat = cv::Mat(rows, cols, type);
    cv::Mat mat = cv::Mat(rows, cols, CV_64FC1);
//    memcpy(mat.data, ndarr.get_data(), sizeof(uchar) * rows * cols * channel);
    memcpy(mat.data, ndarr.get_data(), 2*sizeof(CV_64FC1) * rows * cols);


    return mat;
}


void BA::bundle_adjustment(
    Mat& intrinsic,
    vector<Mat>& extrinsics,
    vector<vector<int> >& correspond_struct_idx,
    vector<vector<KeyPoint> >& key_points_for_all,
    vector<Point3d>& structure
)
{
    ceres::Problem problem;

    // load extrinsics (rotations and motions)
    for (size_t i = 0; i < extrinsics.size(); ++i)
    {
        problem.AddParameterBlock(extrinsics[i].ptr<double>(), 6);
    }
    // fix the first camera.
    problem.SetParameterBlockConstant(extrinsics[0].ptr<double>());

    // load intrinsic
    problem.AddParameterBlock(intrinsic.ptr<double>(), 4); // fx, fy, cx, cy

    // load points
    ceres::LossFunction* loss_function = new ceres::HuberLoss(4);   // loss function make bundle adjustment robuster.
    for (size_t img_idx = 0; img_idx < correspond_struct_idx.size(); ++img_idx)
    {
        vector<int>& point3d_ids = correspond_struct_idx[img_idx];
        vector<KeyPoint>& key_points = key_points_for_all[img_idx];
        for (size_t point_idx = 0; point_idx < point3d_ids.size(); ++point_idx)
        {
            int point3d_id = point3d_ids[point_idx];
            if (point3d_id < 0)
                continue;

            Point2d observed = key_points[point_idx].pt;

            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<ReprojectCost, 2, 4, 6, 3>(new ReprojectCost(observed));

            problem.AddResidualBlock(
                cost_function,
                loss_function,
                intrinsic.ptr<double>(),            // Intrinsic
                extrinsics[img_idx].ptr<double>(),  // View Rotation and Translation
                &(structure[point3d_id].x)          // Point in 3D space
            );
        }
    }

    // Solve BA
    ceres::Solver::Options ceres_config_options;
    ceres_config_options.minimizer_progress_to_stdout = false;
    ceres_config_options.logging_type = ceres::SILENT;
    ceres_config_options.num_threads = 1;
    ceres_config_options.preconditioner_type = ceres::JACOBI;
    ceres_config_options.linear_solver_type = ceres::SPARSE_SCHUR;
    ceres_config_options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    ceres::Solver::Summary summary;
    ceres::Solve(ceres_config_options, &problem, &summary);

    if (!summary.IsSolutionUsable())
    {
        std::cout << "Bundle Adjustment failed." << std::endl;
    }
    else
    {
        // Display statistics about the minimization
        std::cout << std::endl
            << "Bundle Adjustment statistics (approximated RMSE):\n"
            << " #views: " << extrinsics.size() << "\n"
            << " #residuals: " << summary.num_residuals << "\n"
            << " Initial RMSE: " << std::sqrt(summary.initial_cost / summary.num_residuals) << "\n"
            << " Final RMSE: " << std::sqrt(summary.final_cost / summary.num_residuals) << "\n"
            << " Time (s): " << summary.total_time_in_seconds << "\n"
            << std::endl;
    }
}
void BA::save(py::object path, py::list rotation, py::list motions, py::list colors)
{
    string file_path = py::extract<string>(path);
    cout<<file_path<<endl;
    vector<Mat> c_rotation;
    vector<Mat> c_motions;
    vector<Vec3b> c_colors;

    for (int i = 0; i < len(rotation) ; i++){
        np::ndarray token = py::extract<np::ndarray>(rotation[i]);
        c_rotation.push_back(ConvertNDArrayToMat(token));
//        cout<<"c_rotation"<<c_rotation.back()<<endl;
    }
    
    for (int i = 0; i < len(motions) ; i++){
        np::ndarray token = py::extract<np::ndarray>(motions[i]);
        c_motions.push_back(ConvertNDArrayToMat(token));
//        cout<<"c_motions"<<c_motions.back()<<endl;
    }
    
    for (int i = 0; i < len(colors) ; i++){
        np::ndarray token = py::extract<np::ndarray>(colors[i]);
        double g = py::extract<uchar>(token[0]);
        double b = py::extract<uchar>(token[1]);
        double r = py::extract<uchar>(token[2]);
        c_colors.push_back(Vec3b(g,b,r));
//        cout<<"c_colors"<<c_colors.back()<<endl;
    }
    
    save_structure(file_path, c_rotation, c_motions, c_structure, c_colors);
    
}


void BA::save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3d>& structure, vector<Vec3b>& colors)
{
    int n = (int)rotations.size();

    FileStorage fs(file_name, FileStorage::WRITE);
    fs << "Camera Count" << n;
    fs << "Point Count" << (int)structure.size();
    
    fs << "Rotations" << "[";
    for (size_t i = 0; i < n; ++i)
    {
        fs << rotations[i];
    }
    fs << "]";

    fs << "Motions" << "[";
    for (size_t i = 0; i < n; ++i)
    {
        fs << motions[i];
    }
    fs << "]";

    fs << "Points" << "[";
    for (size_t i = 0; i < structure.size(); ++i)
    {
        fs << structure[i];
    }
    fs << "]";

    fs << "Colors" << "[";
    for (size_t i = 0; i < colors.size(); ++i)
    {
        fs << colors[i];
    }
    fs << "]";

    fs.release();
}

class EF{
public:
    EF(py::list l);
    py::list return_kp_list_to_py();
    py::list return_color_list_to_py();
    py::list return_des_list_to_py();
private:
    void ConvertNDArrayToVectorP(const np::ndarray& ndarr, vector<Point2f>& p);
    void extract_features(
                            vector<string>& image_names,
                            vector<vector<KeyPoint> >& key_points_for_all,
                            vector<Mat>& descriptor_for_all,
                            vector<vector<Vec3b> >& colors_for_all
                          );
    vector<string> c_image_names;
    
    vector<vector<KeyPoint> > c_key_points_for_all;
    vector<Mat> c_descriptor_for_all;
    vector<vector<Vec3b> > c_colors_for_all;
//    return_E();
//    return_mask();
};

EF::EF(py::list l)
{
    Py_Initialize();
    np::initialize();

    
    for(int i = 0;i < len(l); i++)
    {
        c_image_names.push_back(py::extract<string>(l[i]));
    }
    
    extract_features(c_image_names, c_key_points_for_all, c_descriptor_for_all, c_colors_for_all);
    
    
}

// example of returning a python list
py::list EF::return_kp_list_to_py(){
    py::list l1;
    for (int i=0; i<c_key_points_for_all.size(); i++){
        py::list l2;
        for (int j=0; j<c_key_points_for_all[i].size(); j++)
        {
            py::tuple tu = py::make_tuple(c_key_points_for_all[i][j].pt.x, c_key_points_for_all[i][j].pt.y, c_key_points_for_all[i][j].size);
            l2.append(tu);
        }
        l1.append(l2);
    }
    return l1;
}

py::list EF::return_des_list_to_py(){
    py::list l1;
    for (int i=0; i<c_descriptor_for_all.size(); i++){
        py::list l2;
        for (int j=0; j<c_descriptor_for_all[i].rows; j++)
        {
            py::list l3;
            for (int k=0; k<c_descriptor_for_all[i].cols; k++)
            {
                l3.append(c_descriptor_for_all[i].at<float>(j,k));
            }
            
            l2.append(l3);
        }
        l1.append(l2);
    }
    return l1;
}

py::list EF::return_color_list_to_py(){
    py::list l1;
    for (int i=0; i<c_colors_for_all.size(); i++){
        py::list l2;
        for (int j=0; j<c_colors_for_all[i].size(); j++)
        {
            py::tuple tu = py::make_tuple((unsigned char)c_colors_for_all[i][j][0], (unsigned char)c_colors_for_all[i][j][1], (unsigned char)c_colors_for_all[i][j][2]);
            l2.append(tu);
        }
        l1.append(l2);
    }
    return l1;
}

void EF::ConvertNDArrayToVectorP(const np::ndarray& ndarr, vector<Point2f>& p)
{
    const Py_intptr_t* shape = ndarr.get_shape();
    int rows = shape[0];
    int cols = shape[1];
    cv::Mat mat = cv::Mat(rows, cols, CV_32FC1);
    memcpy(mat.data, ndarr.get_data(), sizeof(CV_32FC1) * rows * cols);
//    cout<<"Mat:"<<mat<<endl;
    for (int i = 0; i < mat.rows; i++)
        p.push_back(cv::Point2f(mat.at<float>(i,0), mat.at<float>(i,1)));
    cout<<"!!!"<<rows<<" "<<cols<<endl;
}

void EF::extract_features(
    vector<string>& image_names,
    vector<vector<KeyPoint> >& key_points_for_all,
    vector<Mat>& descriptor_for_all,
    vector<vector<Vec3b> >& colors_for_all
    )
{
    key_points_for_all.clear();
    descriptor_for_all.clear();
    Mat image;

    //读取图像，获取图像特征点，并保存
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.001, 10);
    for (auto it = image_names.begin(); it != image_names.end(); ++it)
    {
        image = imread(*it);
        if (image.empty()) continue;

        cout << "Extracing features: " << *it << endl;

        vector<KeyPoint> key_points;
        Mat descriptor;
        //偶尔出现内存分配失败的错误
        sift->detectAndCompute(image, noArray(), key_points, descriptor);

        //特征点过少，则排除该图像
        if (key_points.size() <= 10) continue;

        key_points_for_all.push_back(key_points);
        descriptor_for_all.push_back(descriptor);

        vector<Vec3b> colors(key_points.size());
        for (int i = 0; i < key_points.size(); ++i)
        {
            Point2f& p = key_points[i].pt;
            colors[i] = image.at<Vec3b>(p.y, p.x);
        }
        colors_for_all.push_back(colors);
    }
}






 class FE{
 public:
     FE(py::list l);
 private:
     void ConvertNDArrayToVectorP(const np::ndarray& ndarr, vector<Point2f>& p);
     Mat mask, E;
 //    return_E();
 //    return_mask();
 };

 FE::FE(py::list l)
 {
     Py_Initialize();
     np::initialize();
     vector<Point2f> c_p1, c_p2;
     Mat c_k = cv::Mat(3, 3, CV_32FC1);
     
     
     np::ndarray p1 = py::extract<np::ndarray>(l[0]);
     np::ndarray p2 = py::extract<np::ndarray>(l[1]);
     np::ndarray k = py::extract<np::ndarray>(l[2]);
     
     memcpy(c_k.data, k.get_data(), sizeof(CV_32FC1) * 3 * 3);
     
     cout<<"c_k"<<c_k<<endl;

     
     ConvertNDArrayToVectorP(p1, c_p1);
     ConvertNDArrayToVectorP(p2, c_p2);

     double focal_length = 0.5*(c_k.at<float>(0,0) + c_k.at<float>(1,1));
     Point2d principle_point(c_k.at<float>(0,2), c_k.at<float>(1,2));
     
     cout<<"focal_length"<<focal_length<<endl;
     
     E = findEssentialMat(c_p1, c_p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);

     
     for(int i = 0; i< c_p1.size(); i++)
         cout<<c_p1[i].x<<","<<c_p1[i].y<<endl;
     for(int i = 0; i< c_p2.size(); i++)
         cout<<c_p2[i].x<<","<<c_p2[i].y<<endl;
     cout<<"E"<<E<<endl;
     
 }

 void FE::ConvertNDArrayToVectorP(const np::ndarray& ndarr, vector<Point2f>& p)
 {
     const Py_intptr_t* shape = ndarr.get_shape();
     int rows = shape[0];
     int cols = shape[1];
     cv::Mat mat = cv::Mat(rows, cols, CV_32FC1);
     memcpy(mat.data, ndarr.get_data(), sizeof(CV_32FC1) * rows * cols);
 //    cout<<"Mat:"<<mat<<endl;
     for (int i = 0; i < mat.rows; i++)
         p.push_back(cv::Point2f(mat.at<float>(i,0), mat.at<float>(i,1)));
     cout<<"!!!"<<rows<<" "<<cols<<endl;
 }


// // binding with python
// BOOST_PYTHON_MODULE(BundleAdjustment){
//     py::class_<BA>("BA", py::init<py::list, py::list, py::list, py::list, py::list>())
//         .dFE("return_list_to_py", &BA::return_list_to_py)
//         .dFE("save", &BA::save)
//     ;
//
//     py::class_<FE>("FE", py::init<py::list>())
//     ;
//
// }



// binding with python
BOOST_PYTHON_MODULE(BundleAdjustment){
    py::class_<BA>("BA", py::init<py::list, py::list, py::list, py::list, py::list>())
        .def("save", &BA::save)
    ;
    
    py::class_<EF>("EF", py::init<py::list>())
        .def("return_kp_list_to_py", &EF::return_kp_list_to_py)
        .def("return_color_list_to_py", &EF::return_color_list_to_py)
        .def("return_des_list_to_py", &EF::return_des_list_to_py)
    ;
    py::class_<FE>("FE", py::init<py::list>())
    ;
}

