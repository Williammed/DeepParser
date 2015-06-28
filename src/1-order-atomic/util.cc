#include "util.h"

// Split the string seperated by spliter, e.g. '\t'
void Split(const string &input, const char spliter, vector<string> *out)
{
    int len = input.size();

    if (len == 0)
        return;
    if (out == NULL)
        return;
    
    string tmp;
    for (int i = 0;i < len; ++i)
    {
        if (input[i] != spliter)
        {
            // Append one char to the end of the string
            tmp.append(1, input[i]);
        }
        else
        {
            out->push_back(tmp);
            tmp.clear();
        }
    }

    if (tmp.size() != 0)
        out->push_back(tmp);
}

bool IsChinese(const string &input)
{
    int len = input.size();

    for (int i = 0;i < len;i++)
    {
        if (input[i] >= 'a' && input[i] <= 'z')
            return false;
        if (input[i] >= 'A' && input[i] <= 'Z')
            return false;
    }

    return true;
}
// Judge whether a string is a number
// number examples: 100; 100,000; 1/2; 1.25

bool IsDigit(const string &input, const string &POS, bool chinese)
{
    int len = input.size();
    bool digit_appear = false;

    if (len == 0)
        return false;

    if (chinese && POS == "CD")
        return true;

    for (int i = 0;i < len;i++)
    {
        if (input[i] >= '0' && input[i] <= '9')
            digit_appear = true;
        else if (input[i] == '.' || input[i] == ',' || input[i] == '-' || input[i] == '/')
            continue;
        else
            return false;
    }

    return digit_appear;
}

bool IsPunc(const string &POS)
{
   if (POS == "." || POS == "," || POS == ":" || POS == "``" || POS == "''" || POS == "PU")
       return true;
   return false;
}

void CopyToMatrix(const vector<vector<double> >& input, MatrixXd* output)
{
    int n = input.size();

    if (n == 0)
        return;
    int m = input[0].size();

    for (int i = 0;i < n;i++)
        for (int j = 0;j < m;j++)
            (*output)(i,j) = input[i][j];
}

int ReadConfig(const string &config_file, Config* config)
{
    ifstream fin(config_file.c_str());
    map<string, string> dict;

    if (fin.is_open())
    {
        string buf;

        while(getline(fin, buf))
        {
            //buf = buf.substr(0, buf.size() - 1);
            vector<string> field;
            
            Split(buf, '=', &field);
            dict[field[0]] = field[1];
        }
        fin.close();
        config->n_hidden = atoi(dict["n_hidden"].c_str());
        config->window_size = atoi(dict["window_size"].c_str());
        config->word_embed_size = atoi(dict["word_embed_size"].c_str());
        config->pos_embed_size = atoi(dict["pos_embed_size"].c_str());
        config->dist_embed_size = atoi(dict["dist_embed_size"].c_str());
        config->max_dist = atoi(dict["max_dist"].c_str());
        config->alpha = atof(dict["alpha"].c_str());
        config->batch_size = atoi(dict["batch_size"].c_str());
        config->epoch_num = atoi(dict["epoch_num"].c_str());
        config->l2_reg = atof(dict["l2_reg"].c_str());
        config->margin_reg = atof(dict["margin_reg"].c_str());
        config->dropout_rate = atof(dict["dropout_rate"].c_str());
        config->model_dir = dict["model_dir"];
    }
    else
    {
        return -1;
    }

    return 0;

}

int ReadMeta(Config* config)
{
    string meta_file = config->model_dir + "meta.txt";
    FILE* fp = NULL;

    fp = fopen(meta_file.c_str(), "r");
    if (fp == NULL)
    {
        return -1;
    }
    fscanf(fp, "%d", &(config->word_num));
    fscanf(fp, "%d", &(config->pos_num));
    fscanf(fp, "%d", &(config->edge_type_num));

    fclose(fp);
    return 0;
}

void GenMask(int len, double rate, MatrixXd* mask)
{
    MatrixXd p_vec = (MatrixXd::Random(1, len).array() + 1.0)/2.0;
    (*mask) = MatrixXd::Ones(1, len);
    for (int i = 0;i < len;i++)
    {
        if (p_vec(0,i) <= rate)
            (*mask)(0, i) = 0.0;
    }
}
