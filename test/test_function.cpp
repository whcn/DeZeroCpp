#include "function.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>


TEST(FUNCTION, SQUARE) {
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3, 4, 5, 6;
    Variable x(data);
    Square f;
    Variable &y = f(x);
    std::cout << "Square:\n" << y << std::endl;
}

TEST(FUNCTION, EXPONENTIAL) {
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3, 4, 5, 6;

    Variable x(data);
    Exponential f;
    Variable &y = f(x);

    std::cout << "Exponential:\n" << y << std::endl;
}

TEST(FUNCTION, BACKWARD) {
    Eigen::MatrixXd data(1, 1);
    data << 0.5;

    Square f1;
    Exponential f2;
    Square f3;
    Variable x(data);
    Variable &y1 = f1(x);
    Variable &y2 = f2(y1);
    Variable &y3 = f3(y2);

    y3.grad_ = Eigen::MatrixXd::Ones(1, 1);
    y2.grad_ = f3.Backward(y3.grad_);
    y1.grad_ = f2.Backward(y2.grad_);
    x.grad_ = f1.Backward(y1.grad_);

    std::cout << "Backward:\n" << x.grad_ << std::endl;
}

TEST(FUNCTION, BACKWARD_WITH_CREATOR) {
    Eigen::MatrixXd data(1, 1);
    data << 0.5;

    Square f1;
    Exponential f2;
    Square f3;
    Variable x(data);
    Variable &y1 = f1(x);
    Variable &y2 = f2(y1);
    Variable &y3 = f3(y2);

    y3.grad_ = Eigen::MatrixXd::Ones(1, 1);
    y3.Backward();

    std::cout << "Backward:\n" << x.grad_ << std::endl;
}