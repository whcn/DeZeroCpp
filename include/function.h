#ifndef DEZEROCPP_FUNCTION_H
#define DEZEROCPP_FUNCTION_H

#include "variable.h"
#include <memory>


class Function : public std::enable_shared_from_this<Function> {
public:
    Function() = default;

    virtual ~Function() = default;

    std::vector<std::shared_ptr<Variable>> operator()(std::vector<std::shared_ptr<Variable>> &inputs) {
        std::vector<Eigen::MatrixXd> xs;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(xs),
                       [](const auto &input) { return input->data_; });

        std::vector<Eigen::MatrixXd> ys = Forward(xs);
        input_ = inputs;
        std::vector<std::shared_ptr<Variable>> outputs;
        std::transform(ys.begin(), ys.end(), std::back_inserter(outputs),
                       [](auto &y) { return std::make_shared<Variable>(y); });

        for (auto &output : outputs) {
            output->SetCreator(shared_from_this());
            output_.push_back(output);
        }
        return outputs;
    }

    virtual std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) = 0;

    virtual std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) = 0;

public:
    std::vector<std::shared_ptr<Variable>> input_;
    std::vector<std::weak_ptr<Variable>> output_;
};


class Square : public Function {
public:
    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        Eigen::MatrixXd y = xs[0].cwiseProduct(xs[0]);
        return {y};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        Eigen::MatrixXd x = input_[0]->data_;
        Eigen::MatrixXd gx = 2 * x * gys[0];
        return {gx};
    }
};


class Exp : public Function {
public:
    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        Eigen::MatrixXd y = xs[0].array().exp();
        return {y};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        Eigen::MatrixXd x = input_[0]->data_;
        Eigen::MatrixXd gx = static_cast<Eigen::MatrixXd>(x.array().exp()) * gys[0];
        return {gx};
    }
};

class Pow : public Function {
public:
    explicit Pow(double index) : index_(index) {}

    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        Eigen::MatrixXd y = xs[0].array().pow(index_).matrix();
        return {y};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        Eigen::MatrixXd x = input_[0]->data_;
        Eigen::MatrixXd gx = index_ * x.array().pow(index_ - 1).matrix() * gys[0];
        return {gx};
    }

public:
    double index_ = 1.0;
};

class Neg : public Function {
public:
    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        return {-xs[0]};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        return {-gys[0]};
    }

public:
    float index_ = 1.0;
};

class Add : public Function {
public:
    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        Eigen::MatrixXd y = xs[0] + xs[1];
        return {y};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        return {gys[0], gys[0]};
    }
};

class Sub : public Function {
public:
    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        Eigen::MatrixXd y = xs[0] - xs[1];
        return {y};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        return {gys[0], -gys[0]};
    }
};

class Mul : public Function {
public:
    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        Eigen::MatrixXd y = xs[0] * xs[1];
        return {y};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        Eigen::MatrixXd x0 = input_[0]->data_;
        Eigen::MatrixXd x1 = input_[1]->data_;
        return {gys[0] * x1, gys[0] * x0};
    }
};

class Div : public Function {
public:
    std::vector<Eigen::MatrixXd> Forward(const std::vector<Eigen::MatrixXd> &xs) override {
        Eigen::MatrixXd y = xs[0].cwiseQuotient(xs[1]);
        return {y};
    }

    std::vector<Eigen::MatrixXd> Backward(const std::vector<Eigen::MatrixXd> &gys) override {
        Eigen::MatrixXd x0 = input_[0]->data_;
        Eigen::MatrixXd x1 = input_[1]->data_;
        return {gys[0].cwiseQuotient(x1), gys[0] * (-x0.cwiseQuotient(x1.cwiseProduct(x1)))};
    }
};


std::shared_ptr<Variable> square(std::shared_ptr<Variable> x);
std::shared_ptr<Variable> exp(std::shared_ptr<Variable> x);
std::shared_ptr<Variable> pow(std::shared_ptr<Variable> x, double index);
std::shared_ptr<Variable> neg(std::shared_ptr<Variable> x);
std::shared_ptr<Variable> add(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1);
std::shared_ptr<Variable> sub(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1);
std::shared_ptr<Variable> mul(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1);
std::shared_ptr<Variable> div(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1);

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs, double val);
std::shared_ptr<Variable> operator+(double val, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs, double val);
std::shared_ptr<Variable> operator-(double val, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs, double val);
std::shared_ptr<Variable> operator*(double val, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> lhs, double val);
std::shared_ptr<Variable> operator/(double val, std::shared_ptr<Variable> rhs);
std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> x);
std::shared_ptr<Variable> operator^(std::shared_ptr<Variable> x, double index);

#endif//DEZEROCPP_FUNCTION_H
