#ifndef DEZEROCPP_FUNCTION_H
#define DEZEROCPP_FUNCTION_H

#include "variable.h"

class Function {
public:
    Function() = default;

    virtual ~Function() = default;

    Variable operator()(const Variable& input) {
        x = input.GetData();
        y = forward(x);
        return Variable(y);
    }

protected:
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) = 0;

private:
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
};


class Square : public Function {
protected:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override {
        Eigen::MatrixXd result = x.cwiseProduct(x);
        return result;
    }
};

class Exponential : public Function {
protected:
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) override {
        Eigen::MatrixXd result = x.array().exp();
        return result;
    }
};

#endif//DEZEROCPP_FUNCTION_H
