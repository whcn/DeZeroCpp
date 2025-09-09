#include "function.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>


TEST(FUNCTION, SQUARE) {
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3, 4, 5, 6;

    std::shared_ptr<Variable> x = std::make_shared<Variable>(data);
    std::shared_ptr<Square> f = std::make_shared<Square>();
    std::vector<std::shared_ptr<Variable>> xs = {x};
    std::vector<std::shared_ptr<Variable>> ys = (*f)(xs);

    for (int i = 0; i < x->data_.rows(); ++i) {
        for (int j = 0; j < x->data_.cols(); ++j) {
            int expect = x->data_(i, j) * x->data_(i, j);
            int value = ys[0]->data_(i, j);
            EXPECT_EQ(expect, value);
        }
    }
}

TEST(FUNCTION, EXPONENTIAL) {
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3, 4, 5, 6;

    std::shared_ptr<Variable> x = std::make_shared<Variable>(data);
    std::shared_ptr<Exp> f = std::make_shared<Exp>();
    std::vector<std::shared_ptr<Variable>> xs = {x};
    std::vector<std::shared_ptr<Variable>> ys = (*f)(xs);

    for (int i = 0; i < x->data_.rows(); ++i) {
        for (int j = 0; j < x->data_.cols(); ++j) {
            float expect = std::exp(x->data_(i, j));
            float value = ys[0]->data_(i, j);
            EXPECT_EQ(expect, value);
        }
    }
}

//TEST(FUNCTION, BACKWARD) {
//    Eigen::MatrixXd data(1, 1);
//    data << 0.5;
//
//    std::shared_ptr<Square> f1 = std::make_shared<Square>();
//    std::shared_ptr<Exp> f2 = std::make_shared<Exp>();
//    std::shared_ptr<Square> f3 = std::make_shared<Square>();
//    std::shared_ptr<Variable> x = std::make_shared<Variable>(data);
//
//    std::vector<std::shared_ptr<Variable>> xs = {x};
//    std::vector<std::shared_ptr<Variable>> y1 = (*f1)(xs);
//    std::vector<std::shared_ptr<Variable>> y2 = (*f2)(y1);
//    std::vector<std::shared_ptr<Variable>> y3 = (*f3)(y2);
//
//    y3.grad_ = Eigen::MatrixXd::Ones(1, 1);
//    y2.grad_ = f3.Backward(y3.grad_);
//    y1.grad_ = f2.Backward(y2.grad_);
//    x.grad_ = f1.Backward(y1.grad_);
//
//    float dy3 = 1;
//    float dy2 = NumericalDiff<decltype(f3)>(f3, y2) * dy3;
//    float dy1 = NumericalDiff<decltype(f2)>(f2, y1) * dy2;
//    float dx = NumericalDiff<decltype(f1)>(f1, x) * dy1;
//
//    EXPECT_NEAR(x.grad_(0, 0), dx, 1e-4);
//}

//TEST(FUNCTION, BACKWARD_WITH_CREATOR) {
//    Eigen::MatrixXd data(1, 1);
//    data << 0.5;
//
//    Square f1;
//    Exponential f2;
//    Square f3;
//    Variable x(data);
//    Variable &y1 = f1(x);
//    Variable &y2 = f2(y1);
//    Variable &y3 = f3(y2);
//
//    y3.Backward();
//
//    float dy3 = 1;
//    float dy2 = NumericalDiff<decltype(f3)>(f3, y2) * dy3;
//    float dy1 = NumericalDiff<decltype(f2)>(f2, y1) * dy2;
//    float dx = NumericalDiff<decltype(f1)>(f1, x) * dy1;
//
//    EXPECT_NEAR(x.grad_(0, 0), dx, 1e-4);
//}
//
//TEST(FUNCTION, BACKWARD_WITH_HELPER_FUNC) {
//    Eigen::MatrixXd data(1, 1);
//    data << 0.5;
//
//    Variable x(data);
//    Variable &y = square(exp(square(x)));
//    y.Backward();
//
//    EXPECT_NEAR(x.grad_(0, 0), 3.29744, 1e-4);
//}

TEST(FUNCTION, ADD) {
    auto x0 = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 2));
    auto x1 = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 3));
    std::vector<std::shared_ptr<Variable>> xs = {x0, x1};

    std::shared_ptr<Add> f = std::make_shared<Add>();
    std::vector<std::shared_ptr<Variable>> ys = (*f)(xs);
    EXPECT_EQ(ys[0]->data_(0, 0), 5.0);
}

TEST(FUNCTION, ADD_WITH_HELPER_FUNC) {
    auto x0 = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 2));
    auto x1 = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 3));
    auto y = add(x0, x1);
    EXPECT_EQ(y->data_(0, 0), 5.0);
}

TEST(FUNCTION, ADD_SQUARE_BACKWARD) {
    auto x = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 2));
    auto y = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 3));
    auto z = add(square(x), square(y));
    z->Backward();
    EXPECT_EQ(z->data_(0, 0), 13);
    EXPECT_EQ(x->grad_(0, 0), 4);
    EXPECT_EQ(y->grad_(0, 0), 6);
}

TEST(FUNCTION, ACCUMULATE_GRADIENT_FOR_ADD) {
    auto x0 = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 3));
    auto y0 = add(x0, x0);
    y0->Backward();
    EXPECT_EQ(y0->data_(0, 0), 6);
    EXPECT_EQ(x0->grad_(0, 0), 2);

    auto x1 = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 3));
    auto y1 = add(add(x1, x1), x1);
    y1->Backward();
    EXPECT_EQ(y1->data_(0, 0), 9);
    EXPECT_EQ(x1->grad_(0, 0), 3);
}

TEST(FUNCTION, BACKWARD_AVOID_PROCESS_VISITED_FUNC) {
    auto x = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 2));
    auto a = square(x);
    auto y = add(square(a), square(a));
    y->Backward();
    EXPECT_EQ(y->data_(0, 0), 32);
    EXPECT_EQ(x->grad_(0, 0), 64);
}

TEST(FUNCTION, BREAK_CYCLE_REF_IN_GRAPH) {
    std::shared_ptr<Variable> x0, x1, y0, y1 = nullptr;
    EXPECT_EQ(x1.use_count(), 0);
    EXPECT_EQ(y1.use_count(), 0);
    x1 = x0 = std::make_shared<Variable>(Eigen::MatrixXd::Random(1, 1));
    y1 = y0 = square(x0);
    EXPECT_EQ(x1.use_count(), 3);
    EXPECT_EQ(y1.use_count(), 2);
    x0.reset();
    y0.reset();
    EXPECT_EQ(x1.use_count(), 2);
    EXPECT_EQ(y1.use_count(), 1);
}

TEST(FUNCTION, OVERLOAD_ADD_MUL_OPERATOR) {
    std::shared_ptr<Variable> a = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 3));
    std::shared_ptr<Variable> b = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 2));
    std::shared_ptr<Variable> c = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 1));
    auto y = a * b + c;
    y->Backward();
    EXPECT_EQ(y->data_(0, 0), 7);
    EXPECT_EQ(a->grad_(0, 0), 2);
    EXPECT_EQ(b->grad_(0, 0), 3);
    EXPECT_EQ(c->grad_(0, 0), 1);
}

TEST(FUNCTION, BASIC_OPERATOR_WITH_SCALAR) {
    std::shared_ptr<Variable> x = std::make_shared<Variable>(Eigen::MatrixXd::Constant(1, 1, 2.0));
    auto y1 = -x;
    auto y2 = 2.0 -x;
    auto y3 = x - 1.0;
    auto y4 = 3.0 / x;
    EXPECT_EQ(y1->data_(0, 0), -2.0);
    EXPECT_EQ(y2->data_(0, 0), 0);
    EXPECT_EQ(y3->data_(0, 0), 1.0);
    EXPECT_EQ(y4->data_(0, 0), 1.5);
}
