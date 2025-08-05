#ifndef DEZEROCPP_VARIABLE_H
#define DEZEROCPP_VARIABLE_H

#include <Eigen/Dense>
#include <iostream>
#include <utility>

class Variable {
public:
    Variable() = default;

    explicit Variable(Eigen::MatrixXd &data) : data_(std::move(data)) {}

    friend std::ostream& operator<<(std::ostream& os, const Variable& var) {
        os << var.data_;
        return os;
    }

public:
    Eigen::MatrixXd data_;
    Eigen::MatrixXd grad_;
};

#endif//DEZEROCPP_VARIABLE_H
