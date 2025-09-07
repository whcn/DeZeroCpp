#include <queue>
#include "variable.h"
#include "function.h"


void Variable::SetCreator(std::shared_ptr<Function> creator) {
    creator_ = std::move(creator);
}

void Variable::Backward() {
    if (grad_.size() == 0) {
        grad_ = Eigen::MatrixXd::Ones(data_.rows(), data_.cols());
    }

    std::queue<std::shared_ptr<Function>> funcs;
    funcs.push(creator_);
    while (!funcs.empty()) {
        std::shared_ptr<Function> f = funcs.front();
        funcs.pop();
        auto &xs = f->input_;
        auto &ys = f->output_;
        std::vector<Eigen::MatrixXd> gys;
        std::transform(ys.begin(), ys.end(), std::back_inserter(gys), [&](auto y) { return y->grad_; });
        auto gxs = f->Backward(gys);
        for (int i = 0; i < gxs.size(); ++i) {
            xs[i]->grad_ = gxs[i];
            if (xs[i]->creator_ != nullptr) {
                funcs.push(xs[i]->creator_);
            }
        }
    }
}