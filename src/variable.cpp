#include <unordered_set>
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
    std::unordered_set<std::shared_ptr<Function>> visited;
    auto AddFunc = [&](std::shared_ptr<Function> func) {
        if (func != nullptr && visited.find(func) == visited.end()) {
            funcs.push(func);
            visited.insert(func);
        }
    };
    AddFunc(creator_);

    while (!funcs.empty()) {
        std::shared_ptr<Function> f = funcs.front();
        funcs.pop();
        auto &xs = f->input_;
        auto &ys = f->output_;
        std::vector<Eigen::MatrixXd> gys;
        std::transform(ys.begin(), ys.end(), std::back_inserter(gys), [&](auto y) { return y.lock()->grad_; });
        auto gxs = f->Backward(gys);
        for (int i = 0; i < gxs.size(); ++i) {
            if (xs[i]->grad_.size() == 0) {
                xs[i]->grad_ = gxs[i];
            } else {
                xs[i]->grad_ += gxs[i];
            }
            AddFunc(xs[i]->creator_);
        }
    }
}