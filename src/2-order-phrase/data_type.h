#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <set>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#define MAX_SEN_LEN 256

using namespace std;
using Eigen::MatrixXd;
using Eigen::ArrayXXd;

// Type for each element in the sentence
struct SeqNode
{
    int word;
    int pos;
    int type;
    int head;
    bool punc;
};

struct State
{
    int pos1;
    int pos2;
    int head;
    int incomplete;
    double score;
    State* left;
    State* right;
    State(int p1, int p2, int h, int c, double s)
    {
        pos1 = p1;
        pos2 = p2;
        head = h;
        incomplete = c;
        score = s;
        left = NULL;
        right = NULL;
    }
    State(int p1, int p2, int h, int c, double s,
          State* l, State* r)
    {
        pos1 = p1;
        pos2 = p2;
        head = h;
        incomplete = c;
        score = s;
        left = l;
        right = r;
    }
};

struct StateCmp
{
    bool operator()(const State* tmp1, const State* tmp2) const
    {
        return tmp1->score > tmp2->score;
    }
};

struct Act
{
    string name;
    int act_num;
    Act(const string &tmp, int num):name(tmp), act_num(num){}
};

struct FeatureId
{
    int head_father_word;
    int head_father_pos;
    vector<int> head_child_word;
    vector<int> head_child_pos;
    vector<int> modifier_child_word;
    vector<int> modifier_child_pos;
    vector<int> head_sibling_word;
    vector<int> head_sibling_pos;
};

struct Config
{
    int n_hidden;
    int window_size;
    int word_embed_size;
    int pos_embed_size;
    int dist_embed_size;
    int max_dist;
    int word_num;
    int pos_num;
    int edge_type_num;
    double alpha;
    int batch_size;
    int epoch_num;
    double l2_reg;
    double margin_reg;
    double dropout_rate;
    string model_dir;
};

class Layer
{
public:
    void Init(int in_size, int out_size, double coff)
    {
        n_in = in_size;
        n_out = out_size;
        W = MatrixXd::Random(n_in, n_out) * coff;
        b = MatrixXd::Zero(1, n_out);
        W_grad = MatrixXd::Zero(n_in, n_out);
        b_grad = MatrixXd::Zero(1, n_out);
        W_grad_sqr_sum = MatrixXd::Ones(n_in, n_out);
        b_grad_sqr_sum = MatrixXd::Ones(1, n_out);
    }
    void UpdateParam(double alpha, double l2_reg, int batch_size)
    {
        W_grad /= batch_size;
        b_grad /= batch_size;
        
        // only W is regularized
        W_grad += l2_reg * W;
        // Update gradient square sum
        W_grad_sqr_sum = W_grad_sqr_sum.array() + W_grad.array().square();
        b_grad_sqr_sum = b_grad_sqr_sum.array() + b_grad.array().square();
        // calculate the new learning rate
        ArrayXXd lr_W = alpha/W_grad_sqr_sum.array().sqrt();
        ArrayXXd lr_b = alpha/b_grad_sqr_sum.array().sqrt();
        // update the parameters
        W -= (lr_W * W_grad.array()).matrix();
        b -= (lr_b * b_grad.array()).matrix();
        // set the batch grad to Zero
        W_grad.setZero();
        b_grad.setZero();
    }
    void SaveParam(FILE* fp) const
    {
        for (int i = 0;i < n_in;i++)
        {
            for(int j = 0;j < n_out - 1;j++)
            {
                fprintf(fp, "%lf ", W(i,j));
            }
            fprintf(fp, "%lf\n", W(i, n_out - 1));
        }
        for (int i = 0;i < n_out - 1;i++)
            fprintf(fp, "%lf ", b(0,i));
        fprintf(fp, "%lf\n", b(0,n_out - 1));
    }
    void LoadParam(FILE* fp)
    {
        for (int i = 0;i < n_in;i++)
        {
            for(int j = 0;j < n_out;j++)
            {
                fscanf(fp, "%lf", &W(i,j));
            }
        }
        for (int i = 0;i < n_out;i++)
            fscanf(fp, "%lf", &b(0,i));
    }
    int n_in;
    int n_out;
    MatrixXd W;
    MatrixXd b;
    MatrixXd W_grad;
    MatrixXd b_grad;
    MatrixXd W_grad_sqr_sum;
    MatrixXd b_grad_sqr_sum;
};

class Embedding
{
public:
    void Init(int row, int col)
    {
        row_num = row;
        col_num = col;
        lt = MatrixXd::Random(row_num, col_num) * 0.01;
        lt_grad = MatrixXd::Zero(row_num, col_num);
        lt_grad_sqr_sum = MatrixXd::Ones(row_num, col_num);
    }
    void UpdateParam(double alpha, double l2_reg, int batch_size)
    {
        lt_grad /= batch_size;
        
        // l2_reg
        lt_grad += l2_reg * lt;
        // Update gradient square sum
        lt_grad_sqr_sum = lt_grad_sqr_sum.array() + lt_grad.array().square();
        // calculate the new learning rate
        ArrayXXd lr = alpha/lt_grad_sqr_sum.array().sqrt();
        // update the parameters
        lt -= (lr * lt_grad.array()).matrix();
        // set the batch grad to Zero
        lt_grad.setZero();
    }
    void SaveParam(FILE* fp) const
    {
        for (int i = 0;i < row_num;i++)
        {
            for(int j = 0;j < col_num - 1;j++)
            {
                fprintf(fp, "%lf ", lt(i,j));
            }
            fprintf(fp, "%lf\n", lt(i, col_num - 1));
        }
    }
    void SaveParam(const string &file_name, const vector<string> &id_to_str) const
    {
        FILE* fp = fopen(file_name.c_str(), "w");

        for (int i = 0;i < row_num;i++)
        {
            fprintf(fp, "%s ", id_to_str[i].c_str());
            for(int j = 0;j < col_num - 1;j++)
            {
                fprintf(fp, "%lf ", lt(i,j));
            }
            fprintf(fp, "%lf\n", lt(i, col_num - 1));
        }
        fclose(fp);
    }
    void LoadParam(FILE* fp)
    {
        for (int i = 0;i < row_num;i++)
        {
            for(int j = 0;j < col_num;j++)
            {
                fscanf(fp, "%lf", &lt(i,j));
            }
        }
    }
    void LoadParam(const string &file_name, map<string, int>* dict, vector<string>* id_to_str)
    {
        FILE* fp = fopen(file_name.c_str(), "r");
        char buf[1024];

        for (int i = 0;i < row_num;i++)
        {
            fscanf(fp, "%s", buf);
            (*dict)[buf] = i;
            id_to_str->push_back(buf);
            for(int j = 0;j < col_num;j++)
            {
                fscanf(fp, "%lf", &lt(i,j));
            }
        }
        fclose(fp);
    }
    int row_num;
    int col_num;
    MatrixXd lt;
    MatrixXd lt_grad;
    MatrixXd lt_grad_sqr_sum;
};

class Cache
{
public:
    Cache(int s)
    {
        position_left[0] = vector<MatrixXd>(s, MatrixXd());
        position_left[1] = vector<MatrixXd>(s, MatrixXd());
        position_left_word[0] = vector<MatrixXd>(s, MatrixXd());
        position_left_word[1] = vector<MatrixXd>(s, MatrixXd());
        position_left_pos[0] = vector<MatrixXd>(s, MatrixXd());
        position_left_pos[1] = vector<MatrixXd>(s, MatrixXd());
        position_right[0] = vector<MatrixXd>(s, MatrixXd());
        position_right[1] = vector<MatrixXd>(s, MatrixXd());
        position_right_word[0] = vector<MatrixXd>(s, MatrixXd());
        position_right_word[1] = vector<MatrixXd>(s, MatrixXd());
        position_right_pos[0] = vector<MatrixXd>(s, MatrixXd());
        position_right_pos[1] = vector<MatrixXd>(s, MatrixXd());
        position_input_word = vector<MatrixXd>(s, MatrixXd());
        position_input_pos = vector<MatrixXd>(s, MatrixXd());
        sib[0] = vector<MatrixXd>(s, MatrixXd());
        sib[1] = vector<MatrixXd>(s, MatrixXd());
        sib_middle[0] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        sib_middle[1] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle[0] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle[1] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle_word[0] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle_word[1] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle_pos[0] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle_pos[1] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle_input_word = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        middle_input_pos = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        prefix[0] = vector<MatrixXd>(s, MatrixXd());
        prefix[1] = vector<MatrixXd>(s, MatrixXd());
        prefix_word[0] = vector<MatrixXd>(s, MatrixXd());
        prefix_word[1] = vector<MatrixXd>(s, MatrixXd());
        prefix_pos[0] = vector<MatrixXd>(s, MatrixXd());
        prefix_pos[1] = vector<MatrixXd>(s, MatrixXd());
        prefix_input_word = vector<MatrixXd>(s, MatrixXd());
        prefix_input_pos = vector<MatrixXd>(s, MatrixXd());
        suffix[0] = vector<MatrixXd>(s, MatrixXd());
        suffix[1] = vector<MatrixXd>(s, MatrixXd());
        suffix_word[0] = vector<MatrixXd>(s, MatrixXd());
        suffix_word[1] = vector<MatrixXd>(s, MatrixXd());
        suffix_pos[0] = vector<MatrixXd>(s, MatrixXd());
        suffix_pos[1] = vector<MatrixXd>(s, MatrixXd());
        suffix_input_word = vector<MatrixXd>(s, MatrixXd());
        suffix_input_pos = vector<MatrixXd>(s, MatrixXd());
        dist[0] = vector<MatrixXd>(s + 1, MatrixXd());
        dist[1] = vector<MatrixXd>(s + 1, MatrixXd());
        sib_dist[0] = vector<MatrixXd>(s + 1, MatrixXd());
        sib_dist[1] = vector<MatrixXd>(s + 1, MatrixXd());
        for (int i = 0;i < s;i++)
        {
            hidden_input[0][i] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
            hidden_input[1][i] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
            hidden_activate[0][i] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
            hidden_activate[1][i] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
            output_score[0][i] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
            output_score[1][i] = vector<vector<MatrixXd> >(s, vector<MatrixXd>(s, MatrixXd()));
        }
    }
    vector<MatrixXd> position_left[2];
    vector<MatrixXd> position_left_word[2];
    vector<MatrixXd> position_left_pos[2];
    vector<MatrixXd> position_right[2];
    vector<MatrixXd> position_right_word[2];
    vector<MatrixXd> position_right_pos[2];
    vector<MatrixXd> position_input_word;
    vector<MatrixXd> position_input_pos;
    vector<MatrixXd> sib[2];
    vector<vector<MatrixXd> > middle_input_word;
    vector<vector<MatrixXd> > middle_input_pos;
    vector<vector<MatrixXd> > middle_word[2];
    vector<vector<MatrixXd> > middle_pos[2];
    vector<vector<MatrixXd> > middle[2];
    vector<vector<MatrixXd> > sib_middle[2];
    vector<MatrixXd> prefix_input_word;
    vector<MatrixXd> prefix_input_pos;
    vector<MatrixXd> prefix_word[2];
    vector<MatrixXd> prefix_pos[2];
    vector<MatrixXd> prefix[2];
    vector<MatrixXd> suffix_input_word;
    vector<MatrixXd> suffix_input_pos;
    vector<MatrixXd> suffix_word[2];
    vector<MatrixXd> suffix_pos[2];
    vector<MatrixXd> suffix[2];
    vector<MatrixXd> dist[2];
    vector<MatrixXd> sib_dist[2];
    vector<vector<MatrixXd> > hidden_input[2][MAX_SEN_LEN];
    vector<vector<MatrixXd> > hidden_activate[2][MAX_SEN_LEN];
    vector<vector<MatrixXd> > output_score[2][MAX_SEN_LEN];
    MatrixXd mask;
};
#endif
