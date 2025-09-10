#ifndef DEZEROCPP_VARIABLE_H
#define DEZEROCPP_VARIABLE_H

#include <Eigen/Dense>
#include <iostream>
#include <utility>

class Function;

class Variable {
public:
    Variable() = default;

    explicit Variable(const Eigen::MatrixXd &data) : data_(std::move(data)) {}

    void SetCreator(std::shared_ptr<Function> creator);

    void Backward();

    friend std::ostream &operator<<(std::ostream &os, const Variable &var) {
        os << var.data_;
        return os;
    }

public:
    Eigen::MatrixXd data_;
    Eigen::MatrixXd grad_;
    std::shared_ptr<Function> creator_ = nullptr;
    std::string name_;
};

#endif//DEZEROCPP_VARIABLE_H
