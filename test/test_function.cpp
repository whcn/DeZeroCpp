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

    for (int i = 0; i < x.data_.rows(); ++i) {
        for (int j = 0; j < x.data_.cols(); ++j) {
            int expect = x.data_(i, j) * x.data_(i, j);
            int value = y.data_(i, j);
            EXPECT_EQ(expect, value);
        }
    }
}

TEST(FUNCTION, EXPONENTIAL) {
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3, 4, 5, 6;

    Variable x(data);
    Exponential f;
    Variable &y = f(x);

    for (int i = 0; i < x.data_.rows(); ++i) {
        for (int j = 0; j < x.data_.cols(); ++j) {
            float expect = std::exp(x.data_(i, j));
            float value = y.data_(i, j);
            EXPECT_EQ(expect, value);
        }
    }
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

    float dy3 = 1;
    float dy2 = NumericalDiff<decltype(f3)>(f3, y2) * dy3;
    float dy1 = NumericalDiff<decltype(f2)>(f2, y1) * dy2;
    float dx = NumericalDiff<decltype(f1)>(f1, x) * dy1;

    EXPECT_NEAR(x.grad_(0, 0), dx, 1e-4);
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

    float dy3 = 1;
    float dy2 = NumericalDiff<decltype(f3)>(f3, y2) * dy3;
    float dy1 = NumericalDiff<decltype(f2)>(f2, y1) * dy2;
    float dx = NumericalDiff<decltype(f1)>(f1, x) * dy1;

    EXPECT_NEAR(x.grad_(0, 0), dx, 1e-4);
}

TEST(FUNCTION, BACKWARD_WITH_HELPER_FUNC) {
    Eigen::MatrixXd data(1, 1);
    data << 0.5;

    Variable x(data);
    Variable &y = square(exp(square(x)));
    y.grad_ = Eigen::MatrixXd::Ones(1, 1);
    y.Backward();

    EXPECT_NEAR(x.grad_(0, 0), 3.29744, 1e-4);
}
