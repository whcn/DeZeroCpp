#include <queue>
#include "variable.h"
#include "function.h"


void Variable::SetCreator(Function *creator) { creator_ = creator; }

void Variable::Backward() {
    std::queue<Function*> funcs;
    funcs.push(creator_);
    while (!funcs.empty()) {
        Function* f = funcs.front();
        funcs.pop();
        Variable *x = f->input_;
        Variable *y = f->output_;
        x->grad_ = f->Backward(y->grad_);
        if (x->creator_ != nullptr) {
            funcs.push(x->creator_);
        }
    }
}