#ifndef DEZEROCPP_FUNCTION_H
#define DEZEROCPP_FUNCTION_H

#include "variable.h"

class Function {
public:
    Function() = default;

    virtual ~Function() = default;

    Variable operator()(const Variable& input) {
        x = input.data_;
        y = Forward(x);
        return Variable(y);
    }

    virtual Eigen::MatrixXd Forward(const Eigen::MatrixXd& x) = 0;

    virtual Eigen::MatrixXd Backward(const Eigen::MatrixXd& gy) = 0;

protected:
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
};


class Square : public Function {
public:
    Eigen::MatrixXd Forward(const Eigen::MatrixXd& x) override {
        Eigen::MatrixXd result = x.cwiseProduct(x);
        return result;
    }

    Eigen::MatrixXd Backward(const Eigen::MatrixXd& gy) override {
        Eigen::MatrixXd gx = 2 * x * gy;
        return gx;
    }
};

class Exponential : public Function {
public:
    Eigen::MatrixXd Forward(const Eigen::MatrixXd& x) override {
        Eigen::MatrixXd result = x.array().exp();
        return result;
    }

    Eigen::MatrixXd Backward(const Eigen::MatrixXd& gy) override {
        Eigen::MatrixXd gx = static_cast<Eigen::MatrixXd>(x.array().exp()) * gy;
        return gx;
    }
};

#endif//DEZEROCPP_FUNCTION_H
