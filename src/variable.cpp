#include <queue>
#include "variable.h"
#include "function.h"


void Variable::SetCreator(std::shared_ptr<Function> creator) {
    creator_ = std::move(creator);
}

void Variable::Backward() {
    std::queue<std::shared_ptr<Function>> funcs;
    funcs.push(creator_);
    while (!funcs.empty()) {
        std::shared_ptr<Function> f = funcs.front();
        funcs.pop();
        std::shared_ptr<Variable> x = f->input_;
        std::shared_ptr<Variable> y = f->output_;
        x->grad_ = f->Backward(y->grad_);
        if (x->creator_ != nullptr) {
            funcs.push(x->creator_);
        }
    }
}