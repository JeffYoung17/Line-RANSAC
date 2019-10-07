#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{
    std::vector<double> linspaced;
    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);
    for(int i = 0; i < num - 1; ++i) {
        linspaced.push_back(start + delta * i);
    }

    linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
    return linspaced;
}

struct Point {
public:
    Point(double x, double y) : x_(x), y_(y) {}
    double x_;
    double y_;
};

/**
 * y = a * x + b
 * a = 3, b = 10
 */
int main(int argc, char** argv)
{
    // 1. 生成数据
    // 真实数据
    double PARAM_A = 3, PARAM_B = 10;
    const size_t SIZE = 50;
    vector<double> xx_coor = linspace(0, 10, SIZE);
    vector<double> yy_coor;
    for (const auto &e : xx_coor) {
        yy_coor.push_back(PARAM_A * e + PARAM_B);
    }
    vector<Point> observed_pts;
    for (size_t i = 0; i < xx_coor.size(); ++i) {
        Point pt(xx_coor[i], yy_coor[i]);
        observed_pts.push_back(pt);
    }
    // 带噪声的inliers
    default_random_engine eng;
    normal_distribution<double> gauss_x(0, 0.25);
    normal_distribution<double> gauss_y(0, 0.25);
    for (auto &e : observed_pts) {
        e.x_ += gauss_x(eng);
        e.y_ += gauss_y(eng);
    }
    // 添加outliers
    uniform_real_distribution<> uni_x(0, 10);
    uniform_real_distribution<> uni_y(PARAM_B, PARAM_A * 10 + PARAM_B);
    for (size_t i = 0; i < SIZE; ++i) {
        Point pt(uni_x(eng), uni_y(eng));
        observed_pts.push_back(pt);
    }

    // 2. RANSAC, 估计模型
    double est_a = 0, est_b = 0;
    long iters = 100000;
    double thres = 0.25;
    int preInliers = 0;
    double expProb = 0.99;
    uniform_int_distribution<> uni_sample(0, observed_pts.size() - 1);
    for (long idx = 0; idx < iters; ++idx) { // 对于每一轮迭代
        size_t sampleIdx_a = uni_sample(eng), sampleIdx_b = uni_sample(eng);
        Point pt1 = observed_pts[sampleIdx_a], pt2 = observed_pts[sampleIdx_b];
        double a = (pt2.y_ - pt1.y_) / (pt2.x_ - pt1.x_);
        double b = pt1.y_ - a * pt1.x_;
        // 计算当前模型参数的内点总数量
        int inliersNum = 0;
        for (int i = 0; i < observed_pts.size(); ++i) {
            double tmp_y = a * observed_pts[i].x_ + b;
            if (fabs(tmp_y - observed_pts[i].y_) < thres) {
                ++inliersNum;
            }
        }
        // 如果本次迭代内点数量比以前的都多, 则认为本次迭代效果更好
        if (inliersNum > preInliers) {
            double t = (double)inliersNum / observed_pts.size();
            iters = log(1 - expProb) / log(1 - pow(t, 2));
            preInliers = inliersNum;
            est_a = a;
            est_b = b;
        }
        // 如果本次迭代内点数量超过观测量的一半, 则认为足够好,退出迭代
        if (inliersNum > observed_pts.size() / 2) {
            break;
        }
    }

    // 3. 输出结果
    cout << "groundtruth: " << PARAM_A << " " << PARAM_B << endl;
    cout << "estimate: " << est_a << " " << est_b << endl;

    return 0;
}