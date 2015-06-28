#ifndef UTIL_H_
#define UTIL_H_

#include "data_type.h"

void Split(const string &input, const char spliter, vector<string> *out);
bool IsDigit(const string &input, const string &POS, bool chinese);
bool IsChinese(const string &input);
bool IsPunc(const string &POS);
void CopyToMatrix(const vector<vector<double> >&input, MatrixXd* output);
int ReadConfig(const string &config_file, Config* config);
int ReadMeta(Config* config);
void GenMask(int len, double rate, MatrixXd* mask);
#endif
