#include "variable.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>


TEST(VARIABLE, Init) {
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3, 4, 5, 6;

    Variable var(data);
    std::cout << "Variable data:\n" << var << std::endl;
}
