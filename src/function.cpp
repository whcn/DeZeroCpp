#include "function.h"


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

std::shared_ptr<Variable> square(std::shared_ptr<Variable> x) {
    std::vector<std::shared_ptr<Variable>> xs = {x};
    std::shared_ptr<Square> f = std::make_shared<Square>();
    return (*f)(xs)[0];
}

std::shared_ptr<Variable> exp(std::shared_ptr<Variable> x) {
    std::vector<std::shared_ptr<Variable>> xs = {x};
    std::shared_ptr<Exp> f = std::make_shared<Exp>();
    return (*f)(xs)[0];
}

std::shared_ptr<Variable> pow(std::shared_ptr<Variable> x, double index) {
    std::vector<std::shared_ptr<Variable>> xs = {x};
    std::shared_ptr<Pow> f = std::make_shared<Pow>(index);
    return (*f)(xs)[0];
}

std::shared_ptr<Variable> neg(std::shared_ptr<Variable> x) {
    std::vector<std::shared_ptr<Variable>> xs = {x};
    std::shared_ptr<Neg> f = std::make_shared<Neg>();
    return (*f)(xs)[0];
}

std::shared_ptr<Variable> add(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1) {
    std::vector<std::shared_ptr<Variable>> xs = {x0, x1};
    std::shared_ptr<Add> f = std::make_shared<Add>();
    return (*f)(xs)[0];
}

std::shared_ptr<Variable> sub(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1) {
    std::vector<std::shared_ptr<Variable>> xs = {x0, x1};
    std::shared_ptr<Sub> f = std::make_shared<Sub>();
    return (*f)(xs)[0];
}

std::shared_ptr<Variable> mul(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1) {
    std::vector<std::shared_ptr<Variable>> xs = {x0, x1};
    std::shared_ptr<Mul> f = std::make_shared<Mul>();
    return (*f)(xs)[0];
}

std::shared_ptr<Variable> div(std::shared_ptr<Variable> x0, std::shared_ptr<Variable> x1) {
    std::vector<std::shared_ptr<Variable>> xs = {x0, x1};
    std::shared_ptr<Div> f = std::make_shared<Div>();
    return (*f)(xs)[0];
}


std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs) {
    return add(lhs, rhs);
}

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> lhs, double val) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(lhs->data_.rows(), lhs->data_.cols(), val);
    std::shared_ptr<Variable> rhs = std::make_shared<Variable>(matrix);
    return add(lhs, rhs);
}

std::shared_ptr<Variable> operator+(double val, std::shared_ptr<Variable> rhs) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(rhs->data_.rows(), rhs->data_.cols(), val);
    std::shared_ptr<Variable> lhs = std::make_shared<Variable>(matrix);
    return add(lhs, rhs);
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs) {
    return sub(lhs, rhs);
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> lhs, double val) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(lhs->data_.rows(), lhs->data_.cols(), val);
    std::shared_ptr<Variable> rhs = std::make_shared<Variable>(matrix);
    return sub(lhs, rhs);
}

std::shared_ptr<Variable> operator-(double val, std::shared_ptr<Variable> rhs) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(rhs->data_.rows(), rhs->data_.cols(), val);
    std::shared_ptr<Variable> lhs = std::make_shared<Variable>(matrix);
    return sub(lhs, rhs);
}

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs) {
    return mul(lhs, rhs);
}

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> lhs, double val) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(lhs->data_.rows(), lhs->data_.cols(), val);
    std::shared_ptr<Variable> rhs = std::make_shared<Variable>(matrix);
    return mul(lhs, rhs);
}

std::shared_ptr<Variable> operator*(double val, std::shared_ptr<Variable> rhs) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(rhs->data_.rows(), rhs->data_.cols(), val);
    std::shared_ptr<Variable> lhs = std::make_shared<Variable>(matrix);
    return mul(lhs, rhs);
}

std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> lhs, std::shared_ptr<Variable> rhs) {
    return div(lhs, rhs);
}

std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> lhs, double val) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(lhs->data_.rows(), lhs->data_.cols(), val);
    std::shared_ptr<Variable> rhs = std::make_shared<Variable>(matrix);
    return div(lhs, rhs);
}

std::shared_ptr<Variable> operator/(double val, std::shared_ptr<Variable> rhs) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(rhs->data_.rows(), rhs->data_.cols(), val);
    std::shared_ptr<Variable> lhs = std::make_shared<Variable>(matrix);
    return div(lhs, rhs);
}

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> x) {
    return neg(x);
}

std::shared_ptr<Variable> operator^(std::shared_ptr<Variable> x, double index) {
    return pow(x, index);
}
