#include "variable.h"
#include "function.h"


void Variable::SetCreator(Function *creator) { creator_ = creator; }

void Variable::Backward() {
    if (creator_ != nullptr) {
        Variable *x = creator_->input_;
        x->grad_ = creator_->Backward(grad_);
        x->Backward();
    }
}