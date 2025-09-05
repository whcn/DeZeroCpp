#ifndef DEZEROCPP_FUNCTION_H
#define DEZEROCPP_FUNCTION_H

#include "variable.h"
#include <memory>

class Function : public std::enable_shared_from_this<Function> {
public:
    Function() = default;

    virtual ~Function() = default;

    Variable& operator()(Variable &input) {
        Eigen::MatrixXd x = input.data_;
        Eigen::MatrixXd y = Forward(x);
        input_ = std::shared_ptr<Variable>(&input, [](Variable*){});
        output_ = std::make_shared<Variable>(y);

        // 使用shared_from_this前本对象要由shared_ptr管理，否则会抛异常bad_weak_ptr
        this_ = std::shared_ptr<Function>(this, [](Function*){});
        output_->SetCreator(shared_from_this());
        return *output_;
    }

    virtual Eigen::MatrixXd Forward(const Eigen::MatrixXd &x) = 0;

    virtual Eigen::MatrixXd Backward(const Eigen::MatrixXd &gy) = 0;

public:
    std::shared_ptr<Variable> input_ = nullptr;
    std::shared_ptr<Variable> output_ = nullptr;
    std::shared_ptr<Function> this_ = nullptr;
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
