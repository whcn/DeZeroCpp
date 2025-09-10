#include <fstream>
#include <queue>
#include <unordered_set>
#include <typeinfo>
#include "plot_graph.h"


std::string GenVarDotText(std::shared_ptr<Variable> var) {
    char dot_var[256];
    std::snprintf(dot_var, sizeof(dot_var), "\"%p\" [label=\"%s\", color=orange, style=filled]\n", var.get(), var->name_.c_str());
    return dot_var;
}

std::string GenFuncDotText(std::shared_ptr<Function> func) {
    char dot_func[1024];
    int len = 0;
    len = std::snprintf(dot_func, sizeof(dot_func), "\"%p\" [label=\"%s\", color=lightblue, style=filled, shape=box]\n", func.get(), func->name_.c_str());

    for (auto &input: func->input_) {
        len += std::snprintf(dot_func + len, sizeof(dot_func) - len, "\"%p\" -> \"%p\"\n", input.get(), func.get());
    }
    for (auto &output: func->output_) {
        len += std::snprintf(dot_func + len, sizeof(dot_func) - len, "\"%p\" -> \"%p\"\n", func.get(), output.lock().get());
    }

    return dot_func;
}

std::string GenGraphDotText(std::shared_ptr<Variable> output) {
    std::string dot_graph;
    std::queue<std::shared_ptr<Function>> funcs;
    std::unordered_set<std::shared_ptr<Function>> visited;

    auto AddFunc = [&](std::shared_ptr<Function> func) {
      if (func != nullptr && visited.find(func) == visited.end()) {
          funcs.push(func);
          visited.insert(func);
      }
    };
    AddFunc(output->creator_);
    dot_graph += GenVarDotText(output);

    while (!funcs.empty()) {
        std::shared_ptr<Function> f = funcs.front();
        funcs.pop();
        dot_graph += GenFuncDotText(f);
        auto &xs = f->input_;
        for (auto &x : xs) {
            dot_graph += GenVarDotText(x);
            if (x->creator_) {
                AddFunc(x->creator_);
            }
        }
    }
    return "digraph g {\n" + dot_graph + "}\n";
}

void PlotDotGraph(std::shared_ptr<Variable> output, std::string to_file) {
    auto pos = to_file.rfind('.');
    std::string dot_file = to_file.substr(0, pos) + ".dot";
    std::string extension = to_file.substr(pos + 1);

    std::string dot_graph = GenGraphDotText(output);
    std::ofstream ofs(dot_file);
    ofs << dot_graph;
    ofs.close();

    char cmd[256];
    std::snprintf(cmd, sizeof(cmd), "dot %s -T %s -o %s",
                  dot_file.c_str(), extension.c_str(), to_file.c_str());
    std::system(cmd);
}

