#include <iostream>
#include <Eigen/Dense>


int main() {
    // 创建2x3随机矩阵（类似np.random.rand(2,3)）
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(2, 3);
    std::cout << "随机矩阵:\n" << mat << "\n\n";

    // 创建3维向量并初始化（类似np.array([1,2,3])）
    Eigen::Vector3d vec(1.0, 2.0, 3.0);
    std::cout << "向量:\n" << vec << std::endl;
    return 0;
}