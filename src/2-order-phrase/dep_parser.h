#ifndef DEP_PARSER_H_
#define DEP_PARSER_H_

#include "dep_tree.h"
#include "util.h"
typedef priority_queue<State*, vector<State*>, StateCmp> PQ;
typedef set<State*, StateCmp> SET;

class DepParser
{
public:
    DepParser(struct Config &config):conf(config)
    {
        double coff = 0.0;
        int input_size;
        
        // Initialize word embedding
        word_embedding.Init(conf.word_num, conf.word_embed_size);
        embed_pointers.push_back(&word_embedding);
        // Initialize pos embedding
        pos_embedding.Init(conf.pos_num, conf.pos_embed_size);
        embed_pointers.push_back(&pos_embedding);
        // Initialize dist embedding
        dist_embedding.Init(conf.max_dist, conf.dist_embed_size);
        embed_pointers.push_back(&dist_embedding);
        // Initialize input layer matices
        input_size = 2 * (2*conf.window_size + 1) * conf.word_embed_size + conf.word_embed_size + 
                     2 * (2*conf.window_size + 1) * conf.pos_embed_size + conf.pos_embed_size +
                     conf.dist_embed_size;
        coff = 0.01;//sqrt(6.0/(input_size + conf.n_hidden));
        InitWordInputLayer(coff);
        InitPOSInputLayer(coff);
        // Initialize dist layer
        dist_layer[0].Init(conf.dist_embed_size, conf.n_hidden, coff);
        layer_pointers.push_back(&dist_layer[0]);
        dist_layer[1].Init(conf.dist_embed_size, conf.n_hidden, coff);
        layer_pointers.push_back(&dist_layer[1]);
        // Initialize sib dist layer
        sib_dist_layer[0].Init(conf.dist_embed_size, conf.n_hidden, coff);
        layer_pointers.push_back(&sib_dist_layer[0]);
        sib_dist_layer[1].Init(conf.dist_embed_size, conf.n_hidden, coff);
        layer_pointers.push_back(&sib_dist_layer[1]);
        // Initialize output layer
        //coff = sqrt(6.0/(conf.n_hidden + conf.edge_type_num));
        output_layer[0].Init(conf.n_hidden, conf.edge_type_num, coff);
        layer_pointers.push_back(&output_layer[0]);
        output_layer[1].Init(conf.n_hidden, conf.edge_type_num, coff);
        layer_pointers.push_back(&output_layer[1]);
    }
    void Fit(const vector<vector<SeqNode> > &train_data,
             const vector<vector<SeqNode> > &dev_data,
             const vector<vector<double> > &word_lt,
             const double alpha,
             const int batch_size,
             const int epoch_num,
             const double l2_reg,
             const double margin_reg);
    void Test(const vector<vector<SeqNode> > &test_data,
                const vector<DepTree> &test_gold_trees,
                bool dump,
                double* uas,
                double* las);
    void LoadParam();
    void LoadParam(map<string, int> *word_dict,
                   map<string, int> *pos_dict,
                   map<string, int> *edge_type_dict,
                   vector<string> *id_to_word,
                   vector<string> *id_to_pos,
                   vector<string> *id_to_edge_type);
    void SaveParam();
    void SaveParam(const vector<string> &id_to_word,
                   const vector<string> &id_to_pos,
                   const vector<string> &id_to_edge_type);
    void Analysis();
private:
    void InitWordInputLayer(double coff);
    void InitPOSInputLayer(double coff);
    void Decode(int sen_len, 
                const MatrixXd &sen_word_vec,
                const MatrixXd &sen_pos_vec,
                const DepTree &gold_tree, 
                double penalty,
                bool train, 
                Cache* cache, 
                DepTree* tree);
    double GetArcScore(int pos1, int sib, int pos2, int arc_dir, 
                       const DepTree &gold_tree, 
                       double penalty, 
                       bool train,
                       Cache* cache, 
                       vector<vector<int> >* edge_type);
    void GetSentenceVec(const vector<SeqNode> &seq, MatrixXd* sen_word_vec, MatrixXd* sen_pos_vec);
    void PreCompute(int sen_len, const MatrixXd &sen_word_vec, const MatrixXd &sen_pos_vec, Cache* cache);
    void GetTree(int start, int end, int right, int state, 
                 int rec[][MAX_SEN_LEN][2][3], 
                 DepTree* tree);
    void Activate(const MatrixXd &input_vec, MatrixXd *output_vec);
    double Evaluate(const vector<DepTree> &result, const vector<DepTree> &gold, bool label);
    void BackProp(const DepTree &best_tree,
                    const DepTree &gold_tree,
                    const Cache &cache,
                    const vector<SeqNode> &data);
    void BackPropArc(int start, int sib, int end, int right, int sign,
                     const Cache &cache,
                     const vector<SeqNode> &data);
    // This is only used for gradient check
    double GetTreeScore(const vector<SeqNode> &data, const DepTree &tree, const MatrixXd &mask);
    bool GradientCheck(const DepTree &best_tree,
                       const DepTree &gold_tree,
                       const vector<SeqNode> &data,
                       const MatrixXd &mask,
                       const MatrixXd &my_grad,
                       MatrixXd* mat);
    void UpdateParam(double alpha, double l2_reg, int batch_size);
    int GetActNum(int pos, const Layer& layer);

    struct Config conf;
    Embedding word_embedding;
    Embedding pos_embedding;
    Embedding dist_embedding;
    Layer left_layer_word[2]; // One for left arc and one for right arc,0 is left
    Layer prefix_layer_word[2];
    Layer suffix_layer_word[2];
    Layer middle_layer_word[2];
    Layer right_layer_word[2];
    Layer left_layer_pos[2];
    Layer prefix_layer_pos[2];
    Layer suffix_layer_pos[2];
    Layer middle_layer_pos[2];
    Layer right_layer_pos[2];
    Layer dist_layer[2];
    Layer sib_layer_word[2];
    Layer sib_layer_pos[2];
    Layer sib_middle_layer_word[2];
    Layer sib_middle_layer_pos[2];
    Layer sib_dist_layer[2];
    Layer output_layer[2]; // for now, we use two seperate output_layer for left arc and right arc
    vector<Embedding*> embed_pointers;
    vector<Layer*> layer_pointers;
};

#endif
