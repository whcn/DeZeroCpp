#ifndef DEZEROCPP_FUNCTION_H
#define DEZEROCPP_FUNCTION_H

#include "variable.h"

class Function {
public:
    Function() = default;

    virtual ~Function() = default;

    Variable& operator()(Variable &input) {
        Eigen::MatrixXd x = input.data_;
        Eigen::MatrixXd y = Forward(x);
        y_ = Variable(y);
        input_ = &input;
        output_ = &y_;
        output_->SetCreator(this);
        return *output_;
    }

    virtual Eigen::MatrixXd Forward(const Eigen::MatrixXd &x) = 0;

    virtual Eigen::MatrixXd Backward(const Eigen::MatrixXd &gy) = 0;

public:
    Variable *input_ = nullptr;
    Variable *output_ = nullptr;
    Variable y_;
};


class Square : public Function {
public:
    Eigen::MatrixXd Forward(const Eigen::MatrixXd &x) override {
        Eigen::MatrixXd result = x.cwiseProduct(x);
        return result;
    }

    Eigen::MatrixXd Backward(const Eigen::MatrixXd &gy) override {
        Eigen::MatrixXd x = input_->data_;
        Eigen::MatrixXd gx = 2 * x * gy;
        return gx;
    }
};

class Exponential : public Function {
public:
    Eigen::MatrixXd Forward(const Eigen::MatrixXd &x) override {
        Eigen::MatrixXd result = x.array().exp();
        return result;
    }

    Eigen::MatrixXd Backward(const Eigen::MatrixXd &gy) override {
        Eigen::MatrixXd x = input_->data_;
        Eigen::MatrixXd gx = static_cast<Eigen::MatrixXd>(x.array().exp()) * gy;
        return gx;
    }
};

#endif//DEZEROCPP_FUNCTION_H
