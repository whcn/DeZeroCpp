#include "function.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>


TEST(FUNCTION, SQUARE) {
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3, 4, 5, 6;
    Variable x(data);
    Square f;
    Variable y = f(x);
    std::cout << "Square:\n" << y << std::endl;
}
