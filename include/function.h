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

template<typename FuncType>
static float NumericalDiff(FuncType f, Variable &x, float eps = 1e-4) {
    Eigen::MatrixXd _x0 = x.data_.array() - eps;
    Eigen::MatrixXd _x1 = x.data_.array() + eps;
    Variable x0 = Variable(_x0);
    Variable x1 = Variable(_x1);
    FuncType f0 = f;
    FuncType f1 = f;
    Variable &y0 = f0(x0);
    Variable &y1 = f1(x1);
    float diff = (y1.data_(0, 0) - y0.data_(0, 0)) / (2 * eps);
    return diff;
}

static std::shared_ptr<Variable> square(std::shared_ptr<Variable> x) {
    std::vector<std::shared_ptr<Variable>> xs = {x};
    std::shared_ptr<Square> f = std::make_shared<Square>();
    return (*f)(xs)[0];
}

//static Variable& exp(Variable &x) {
//    std::shared_ptr<Exponential> f = std::make_shared<Exponential>();
//    return (*f)(x);
//}

static std::shared_ptr<Variable> add(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1) {
    std::vector<std::shared_ptr<Variable>> xs = {x0, x1};
    std::shared_ptr<Add> f = std::make_shared<Add>();
    return (*f)(xs)[0];
}

#endif//DEZEROCPP_FUNCTION_H
