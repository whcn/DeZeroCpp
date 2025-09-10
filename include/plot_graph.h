#ifndef DEZEROCPP_PLOT_GRAPH_H
#define DEZEROCPP_PLOT_GRAPH_H

#include "function.h"
#include <string>

std::string GenVarDotText(std::shared_ptr<Variable> var);

std::string GenFuncDotText(std::shared_ptr<Function> func);

std::string GenDotText(std::shared_ptr<Variable> output);

void PlotDotGraph(std::shared_ptr<Variable> output, std::string to_file);

#endif//DEZEROCPP_PLOT_GRAPH_H
