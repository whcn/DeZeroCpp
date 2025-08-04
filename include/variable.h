#ifndef DEZEROCPP_VARIABLE_H
#define DEZEROCPP_VARIABLE_H

#include <Eigen/Dense>
#include <iostream>
#include <utility>

class Variable {
public:
    Variable() = default;

    explicit Variable(Eigen::MatrixXd &data) : data_(std::move(data)) {}

   Eigen::MatrixXd GetData() const {
        return data_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Variable& var) {
        os << var.data_;
        return os;
    }

private:
    Eigen::MatrixXd data_;
};

#endif//DEZEROCPP_VARIABLE_H
