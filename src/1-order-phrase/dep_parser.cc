#include "dep_parser.h"
#include <time.h>

void DepParser::InitWordInputLayer(double coff)
{
    // left word layer
    int input_size = (2 * conf.window_size + 1) * conf.word_embed_size;
    left_layer_word[0].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&left_layer_word[0]);
    left_layer_word[1].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&left_layer_word[1]);
    // prefix word layer
    prefix_layer_word[0].Init(conf.word_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&prefix_layer_word[0]);
    prefix_layer_word[1].Init(conf.word_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&prefix_layer_word[1]);
    // suffix word layer
    suffix_layer_word[0].Init(conf.word_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&suffix_layer_word[0]);
    suffix_layer_word[1].Init(conf.word_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&suffix_layer_word[1]);
    // middle word layer
    middle_layer_word[0].Init(conf.word_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&middle_layer_word[0]);
    middle_layer_word[1].Init(conf.word_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&middle_layer_word[1]);
    // right word layer
    right_layer_word[0].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&right_layer_word[0]);
    right_layer_word[1].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&right_layer_word[1]);
}

void DepParser::InitPOSInputLayer(double coff)
{
    int input_size = (2 * conf.window_size + 1) * conf.pos_embed_size;
    // left pos layer
    left_layer_pos[0].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&left_layer_pos[0]);
    left_layer_pos[1].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&left_layer_pos[1]);
    // prefix pos layer
    prefix_layer_pos[0].Init(conf.pos_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&prefix_layer_pos[0]);
    prefix_layer_pos[1].Init(conf.pos_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&prefix_layer_pos[1]);
    // suffix pos layer
    suffix_layer_pos[0].Init(conf.pos_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&suffix_layer_pos[0]);
    suffix_layer_pos[1].Init(conf.pos_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&suffix_layer_pos[1]);
    // middle pos layer
    middle_layer_pos[0].Init(conf.pos_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&middle_layer_pos[0]);
    middle_layer_pos[1].Init(conf.pos_embed_size, conf.n_hidden, coff);
    layer_pointers.push_back(&middle_layer_pos[1]);
    // right pos layer
    right_layer_pos[0].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&right_layer_pos[0]);
    right_layer_pos[1].Init(input_size, conf.n_hidden, coff);
    layer_pointers.push_back(&right_layer_pos[1]);
}

void DepParser::GetSentenceVec(const vector<SeqNode>& seq,
                               MatrixXd* sen_word_vec,
                               MatrixXd* sen_pos_vec)
{
    int sen_len = seq.size();

    // Initialize the sentence vec
    int cur_pos = 0;
    int start = 0;
    // First add <BOS>
    // Note that the word_id and POS_id of <BOS> is always 0
    // and the word_id and POS_id for <EOS> is always 1
    for (int i = 0;i < conf.window_size;i++)
    {
        start = cur_pos * conf.word_embed_size;
        sen_word_vec->row(0).segment(start, conf.word_embed_size) = word_embedding.lt.row(0);
        start = cur_pos * conf.pos_embed_size;
        sen_pos_vec->row(0).segment(start, conf.pos_embed_size) = pos_embedding.lt.row(0);
        cur_pos++;
    }
    // Add sentence word
    for (int i = 0;i < sen_len;i++)
    {
        start = cur_pos * conf.word_embed_size;
        int word_id = seq[i].word;
        sen_word_vec->row(0).segment(start, conf.word_embed_size) = word_embedding.lt.row(word_id);
        start = cur_pos * conf.pos_embed_size;
        int pos_id = seq[i].pos;
        sen_pos_vec->row(0).segment(start, conf.pos_embed_size) = pos_embedding.lt.row(pos_id);
        cur_pos++;
    }
    // Add <EOS>
    for (int i = 0;i < conf.window_size;i++)
    {
        start = cur_pos * conf.word_embed_size;
        sen_word_vec->row(0).segment(start, conf.word_embed_size) = word_embedding.lt.row(1);
        start = cur_pos * conf.pos_embed_size;
        sen_pos_vec->row(0).segment(start, conf.pos_embed_size) = pos_embedding.lt.row(1);
        cur_pos++;
    }
}

void DepParser::PreCompute(int sen_len, const MatrixXd &sen_word_vec, const MatrixXd &sen_pos_vec, Cache* cache)
{
    vector<MatrixXd> sum_word;
    vector<MatrixXd> sum_pos;
    
    //printf("Start postion\n");
    //printf("sen_len: %d\n", sen_len);
    // Get the prefix sum for each position and compute cache for postion
    // Note that the sen_word_vec and sen_pos_vec has <BOS> and <EOS>
    for (int i = 0;i < sen_len;i++)
    {
        // to skip <BOS> and <EOS>
        int start_word = (i + conf.window_size) * conf.word_embed_size;
        int start_pos = (i + conf.window_size) * conf.pos_embed_size;
        if ( i == 0)
        {
            sum_word.push_back(sen_word_vec.row(0).segment(start_word, conf.word_embed_size));
            sum_pos.push_back(sen_pos_vec.row(0).segment(start_pos, conf.pos_embed_size));
        }
        else
        {
            MatrixXd tmp_word = sum_word[i - 1] + sen_word_vec.row(0).segment(start_word, conf.word_embed_size);
            MatrixXd tmp_pos = sum_pos[i - 1] + sen_pos_vec.row(0).segment(start_pos, conf.pos_embed_size);
            sum_word.push_back(tmp_word);
            sum_pos.push_back(tmp_pos);
        }
        start_word = i * conf.word_embed_size;
        start_pos = i * conf.pos_embed_size;
        MatrixXd input_position_word = sen_word_vec.row(0).segment(start_word, (2*conf.window_size + 1)*conf.word_embed_size);
        MatrixXd input_position_pos = sen_pos_vec.row(0).segment(start_pos, (2*conf.window_size+1)*conf.pos_embed_size);
        cache->position_input_word[i] = input_position_word;
        cache->position_input_pos[i] = input_position_pos;
        // left arc for left position
        cache->position_left_word[0][i] = input_position_word * left_layer_word[0].W + left_layer_word[0].b;
        cache->position_left_pos[0][i] = input_position_pos * left_layer_pos[0].W + left_layer_pos[0].b;
        cache->position_left[0][i] = cache->position_left_word[0][i] + cache->position_left_pos[0][i];
        // right arc for left position
        cache->position_left_word[1][i] = input_position_word * left_layer_word[1].W + left_layer_word[1].b;
        cache->position_left_pos[1][i] = input_position_pos * left_layer_pos[1].W + left_layer_pos[1].b;
        cache->position_left[1][i] = cache->position_left_word[1][i] + cache->position_left_pos[1][i];
        // left arc for right position
        cache->position_right_word[0][i] = input_position_word * right_layer_word[0].W + right_layer_word[0].b;
        cache->position_right_pos[0][i] = input_position_pos * right_layer_pos[0].W + right_layer_pos[0].b;
        cache->position_right[0][i] = cache->position_right_word[0][i] + cache->position_right_pos[0][i];
        // right arc for right postion
        cache->position_right_word[1][i] = input_position_word * right_layer_word[1].W + right_layer_word[1].b;
        cache->position_right_pos[1][i] = input_position_pos * right_layer_pos[1].W + right_layer_pos[1].b;
        cache->position_right[1][i] = cache->position_right_word[1][i] + cache->position_right_pos[1][i];
        // left arc for dist
        cache->dist[0][i + 1] = dist_embedding.lt.row(i + 1) * dist_layer[0].W + dist_layer[0].b;
        // right arc for dist
        cache->dist[1][i + 1] = dist_embedding.lt.row(i + 1) * dist_layer[1].W + dist_layer[1].b;
    }
    //printf("Start middle\n");
    MatrixXd aver_word;
    MatrixXd aver_pos;
    //cache for prefix
    for (int j = 0;j < sen_len;j++)
    {
        if (j == 0)
        {
            cache->prefix_input_word[j] = MatrixXd::Zero(1, conf.word_embed_size);
            cache->prefix_input_pos[j] = MatrixXd::Zero(1, conf.pos_embed_size);
        }
        else
        {
            cache->prefix_input_word[j] = sum_word[j - 1] / j;
            cache->prefix_input_pos[j] = sum_pos[j - 1]/j;
        }
        // left arc
        cache->prefix_word[0][j] = cache->prefix_input_word[j] * prefix_layer_word[0].W + 
                                   prefix_layer_word[0].b;
        cache->prefix_pos[0][j] = cache->prefix_input_pos[j] * prefix_layer_pos[0].W + 
                                  prefix_layer_pos[0].b;
        cache->prefix[0][j] = cache->prefix_word[0][j] + cache->prefix_pos[0][j];
        // right arc
        cache->prefix_word[1][j] = cache->prefix_input_word[j] * prefix_layer_word[1].W + 
                                   prefix_layer_word[1].b;
        cache->prefix_pos[1][j] = cache->prefix_input_pos[j] * prefix_layer_pos[1].W + 
                                  prefix_layer_pos[1].b;
        cache->prefix[1][j] = cache->prefix_word[1][j] + cache->prefix_pos[1][j];
    }
    // cache for suffix
    for (int j = 0;j < sen_len;j++)
    {
        if (j == sen_len - 1)
        {
            cache->suffix_input_word[j] = MatrixXd::Zero(1, conf.word_embed_size);
            cache->suffix_input_pos[j] = MatrixXd::Zero(1, conf.pos_embed_size);
        }
        else
        {
            cache->suffix_input_word[j] = (sum_word[sen_len - 1] - sum_word[j]) / (sen_len - 1 - j);
            cache->suffix_input_pos[j] = (sum_pos[sen_len - 1] - sum_pos[j]) / (sen_len - 1 - j);
        }
        // left arc
        cache->suffix_word[0][j] = cache->suffix_input_word[j] * suffix_layer_word[0].W + 
                                   suffix_layer_word[0].b;
        cache->suffix_pos[0][j] = cache->suffix_input_pos[j] * suffix_layer_pos[0].W + 
                                  suffix_layer_pos[0].b;
        cache->suffix[0][j] = cache->suffix_word[0][j] + cache->suffix_pos[0][j];
        // right arc
        cache->suffix_word[1][j] = cache->suffix_input_word[j] * suffix_layer_word[1].W + 
                                   suffix_layer_word[1].b;
        cache->suffix_pos[1][j] = cache->suffix_input_pos[j] * suffix_layer_pos[1].W + 
                                  suffix_layer_pos[1].b;
        cache->suffix[1][j] = cache->suffix_word[1][j] + cache->suffix_pos[1][j];
    }

    // cache for middle
    for (int j = 1; j < sen_len;j++)
    {
        // left arc for middle
        aver_word = sum_word[j]/(j+1);
        aver_pos = sum_pos[j]/(j+1);
        cache->middle_input_word[0][j] = aver_word;
        cache->middle_input_pos[0][j] = aver_pos;

        cache->middle_word[0][0][j] = aver_word * middle_layer_word[0].W + middle_layer_word[0].b;
        cache->middle_pos[0][0][j] = aver_pos * middle_layer_pos[0].W + middle_layer_pos[0].b;
        cache->middle[0][0][j] = cache->middle_word[0][0][j] + cache->middle_pos[0][0][j];
        // right arc for middle
        cache->middle_word[1][0][j] = aver_word * middle_layer_word[1].W + middle_layer_word[1].b;
        cache->middle_pos[1][0][j] = aver_pos * middle_layer_pos[1].W + middle_layer_pos[1].b;
        cache->middle[1][0][j] = cache->middle_word[1][0][j] + cache->middle_pos[1][0][j];
    }
    //printf("start middle hidden\n");
    for (int i = 1; i < sen_len;i++)
        for (int j = i + 1;j < sen_len;j++)
        {
            // left arc for middel
            aver_word = (sum_word[j] - sum_word[i - 1])/(j - i + 1);
            aver_pos = (sum_pos[j] - sum_pos[i - 1])/(j - i + 1);
            cache->middle_input_word[i][j] = aver_word;
            cache->middle_input_pos[i][j] = aver_pos;
            cache->middle_word[0][i][j] = aver_word * middle_layer_word[0].W + middle_layer_word[0].b;
            cache->middle_pos[0][i][j] = aver_pos * middle_layer_pos[0].W + middle_layer_pos[0].b;
            cache->middle[0][i][j] = cache->middle_word[0][i][j] + cache->middle_pos[0][i][j];
            // right arc for middle
            cache->middle_word[1][i][j] = aver_word * middle_layer_word[1].W + middle_layer_word[1].b;
            cache->middle_pos[1][i][j] = aver_pos * middle_layer_pos[1].W + middle_layer_pos[1].b;
            cache->middle[1][i][j] = cache->middle_word[1][i][j] + cache->middle_pos[1][i][j];
        }
    // each sentence has only one mask
    GenMask(conf.n_hidden, conf.dropout_rate, &(cache->mask));
}
void DepParser::Decode(int sen_len,
                       const MatrixXd &sen_word_vec,
                       const MatrixXd &sen_pos_vec,
                       const DepTree &gold_tree, 
                       double penalty, 
                       bool train,
                       Cache* cache, 
                       DepTree* tree)
{
    // In the future, we should change this to more adaptable code
    const double INF = DBL_MAX - 1.0;
    double dp[MAX_SEN_LEN][MAX_SEN_LEN][2][2];
    int rec[MAX_SEN_LEN][MAX_SEN_LEN][2][2];
    vector<vector<int> > edge_type(sen_len, vector<int>(sen_len, -1));

    // Precomputer all kinds of score, position_score, average_score, dist_score
    // which will be used later for decoding
    //printf("Start precompute\n");
    PreCompute(sen_len, sen_word_vec, sen_pos_vec, cache);

    for (int i = 0;i < sen_len;i++)
        for (int j = i;j < sen_len;j++)
            for(int k = 0;k < 2;k++)
                for(int p = 0;p < 2;p++)
                {
                    if(i == j)
                        dp[i][i][k][p] = 0.0;
                    else
                        dp[i][j][k][p] = -INF;
                    rec[i][j][k][p] = -1;
                }
    //printf("Start dp\n");
    for (int m = 1; m <= sen_len;m++)
        for(int s = 1; s < sen_len;s++)
        {
            int t = s + m;
            if ( t >= sen_len)
                break;
            
            double score_left = GetArcScore(s, t, 0, gold_tree, penalty, train, cache, &edge_type);
            double score_right = GetArcScore(s, t, 1, gold_tree, penalty, train, cache, &edge_type);
            for (int q = s;q < t;q++)
            {
                // Create subgraphs with c = 1 by adding arcs
                if (dp[s][q][1][0] != -INF && dp[q+1][t][0][0] != -INF &&
                    dp[s][q][1][0] + dp[q+1][t][0][0] + score_left > dp[s][t][0][1])
                {
                    dp[s][t][0][1] = dp[s][q][1][0] + dp[q+1][t][0][0] + score_left;
                    rec[s][t][0][1] = q;
                }
            }
            for (int q = s;q < t;q++)
            {
                if (dp[s][q][1][0] != -INF && dp[q+1][t][0][0] != -INF &&
                    dp[s][q][1][0] + dp[q+1][t][0][0] + score_right > dp[s][t][1][1])
                {
                    dp[s][t][1][1] = dp[s][q][1][0] + dp[q+1][t][0][0] + score_right;
                    rec[s][t][1][1] = q;
                }
            }
            for (int q = s;q < t;q++)
            {   
                // Add corresponding left subgraphs, q==t is meaningless
                if (dp[s][q][0][0] != -INF && dp[q][t][0][1] != -INF &&
                    dp[s][q][0][0] + dp[q][t][0][1] > dp[s][t][0][0])
                {
                    dp[s][t][0][0] = dp[s][q][0][0] + dp[q][t][0][1];
                    rec[s][t][0][0] = q;
                }
            }
            for (int q = s + 1; q <= t;q++)
            {
                // Add corresponding right subgraphs, q==t is ok
                if (dp[s][q][1][1] != -INF && dp[q][t][1][0] != -INF &&
                    dp[s][q][1][1] + dp[q][t][1][0] > dp[s][t][1][0])
                {
                    dp[s][t][1][0] = dp[s][q][1][1] + dp[q][t][1][0];
                    rec[s][t][1][0] = q;
                }
            }
        }
    
    //printf("Start root dp\n");
    // Find the root, we assume there would be only one child for root node
    for (int i = 1;i < sen_len; i++)
    {
        double score = GetArcScore(0, i, 1, gold_tree, penalty, train, cache, &edge_type);
        if (dp[1][i][0][0] != -INF && score + dp[1][i][0][0] > dp[0][i][1][1])
        {
            dp[0][i][1][1] = dp[1][i][0][0] + score;
            rec[0][i][1][1] = 0;
        }
    }
    for (int i = 1;i < sen_len;i++)
    {
        // Final graph
        if (dp[0][i][1][1] != -INF && dp[i][sen_len - 1][1][0] != -INF &&
            dp[0][i][1][1] + dp[i][sen_len - 1][1][0] > dp[0][sen_len-1][1][0])
        {
            dp[0][sen_len-1][1][0] = dp[0][i][1][1] + dp[i][sen_len-1][1][0];
            rec[0][sen_len-1][1][0] = i;
        }
    }
    
    // Get the tree
    tree->score = dp[0][sen_len - 1][1][0];
    //printf("Start Get tree\n");
    GetTree(0, sen_len - 1, 1, 0, rec, edge_type, tree);
}

void DepParser::GetTree(State* state, 
                        const vector<vector<int> > &edge_type,
                        DepTree* tree)
{
    if (state == NULL)
        return;
    
    int start = state->pos1;
    int end = state->pos2;
    int right = state->head;
    int incomplete = state->incomplete;

    if (start >= end)
        return;

    if (incomplete == 1) // We add edge in this situation
    {
        if (right == 0)
        {
            tree->head[start] = end;
            tree->children[end].push_back(start);
            // edge_type is uni_directional
            tree->edge_type[start] = edge_type[end][start];
        }
        else
        {
            tree->head[end] = start;
            tree->children[start].push_back(end);
            tree->edge_type[end] = edge_type[start][end];
        }
        GetTree(state->left, edge_type, tree);
        GetTree(state->right, edge_type, tree);
    }
    else // A complete half tree, just split them into two parts
    {
        GetTree(state->left, edge_type, tree);
        GetTree(state->right, edge_type, tree);
    }
}

void DepParser::GetTree(int start, int end, int right, int incomplete,
                        int rec[][MAX_SEN_LEN][2][2],
                        const vector<vector<int> > &edge_type,
                        DepTree* tree)
{
    if (start >= end)
        return;
    
    if (incomplete == 1) // We add edge in this situation
    {
        if (right == 0)
        {
            tree->head[start] = end;
            tree->children[end].push_back(start);
            // edge_type is uni_directional
            tree->edge_type[start] = edge_type[end][start];
        }
        else
        {
            tree->head[end] = start;
            tree->children[start].push_back(end);
            tree->edge_type[end] = edge_type[start][end];
        }
        int q = rec[start][end][right][incomplete];
        GetTree(start, q, 1, 0, rec, edge_type, tree);
        GetTree(q + 1, end, 0, 0, rec, edge_type, tree);
    }
    else // A complete half tree, just split them into two parts
    {
        int q = rec[start][end][right][incomplete];
        if (right == 0)
        {
            GetTree(start, q, 0, 0, rec, edge_type, tree);
            GetTree(q, end, 0, 1, rec, edge_type, tree);
        }
        else
        {
            GetTree(start, q, 1, 1, rec, edge_type, tree);
            GetTree(q, end, 1, 0, rec, edge_type, tree);
        }
    }
}

void DepParser::Activate(const MatrixXd &input_vec, MatrixXd *output_vec)
{
    // We use cubic activation for now
    // Eigen allows to assign a array to a matrix
    ArrayXXd cube = input_vec.array() * input_vec.array() * input_vec.array() + input_vec.array();
    ArrayXXd pos = cube.exp();
    ArrayXXd neg = (-cube).exp();
    (*output_vec) = (pos - neg)/(pos + neg);
}

double DepParser::GetArcScore(int pos1, int pos2, int arc_dir, 
                              const DepTree &gold_tree,
                              double penalty,
                              bool train,
                              Cache *cache, 
                              vector<vector<int> >* edge_type)
{
    MatrixXd position_input = cache->position_left[arc_dir][pos1] + cache->position_right[arc_dir][pos2];
    MatrixXd prefix_input = cache->prefix[arc_dir][pos1];
    MatrixXd suffix_input = cache->suffix[arc_dir][pos2];
    MatrixXd middle_input = cache->middle[arc_dir][pos1][pos2];
    MatrixXd dist_input = cache->dist[arc_dir][pos2 - pos1];
    cache->hidden_input[arc_dir][pos1][pos2] = position_input + prefix_input + suffix_input +
                                               middle_input + dist_input;
    Activate(cache->hidden_input[arc_dir][pos1][pos2], &(cache->hidden_activate[arc_dir][pos1][pos2]));

    if (train)
    {
        // dropout
        cache->hidden_activate[arc_dir][pos1][pos2] = cache->hidden_activate[arc_dir][pos1][pos2].array() *
                                                      cache->mask.array();
    }
    else
    {
        // decay the output of hidden layer to 1.0 - dropout_rate during testing
        cache->hidden_activate[arc_dir][pos1][pos2] *= (1.0 - conf.dropout_rate);
    }
    
    MatrixXd output_score = cache->hidden_activate[arc_dir][pos1][pos2] * output_layer[arc_dir].W + output_layer[arc_dir].b;
    cache->output_score[arc_dir][pos1][pos2] = output_score;
    if (penalty != 0.0)
    {
        MatrixXd penalty_vec = MatrixXd::Ones(1, conf.edge_type_num) * penalty;
    
        // Set penalty vec, if the head is right
        // Set the correct edge type penalty to 0
        if (arc_dir == 0 && gold_tree.head[pos1] == pos2)
            penalty_vec(0, gold_tree.edge_type[pos1]) = 0.0;
        else if(arc_dir == 1 && gold_tree.head[pos2] == pos1)
            penalty_vec(0, gold_tree.edge_type[pos2]) = 0.0;
        output_score += penalty_vec;
    }

    int row_id = 0, col_id = 0; // row_id is useless, since the shape is (1, col_num)
    double score = output_score.maxCoeff(&row_id, &col_id);
    if (arc_dir == 0)
        (*edge_type)[pos2][pos1] = col_id;
    else
        (*edge_type)[pos1][pos2] = col_id;

    return score;
}

double DepParser::Evaluate(const vector<DepTree> &result, 
                           const vector<DepTree> &gold,
                           bool label)
{
    int n = result.size();
    float wrong = 0.0;
    float tot = 0.0;

    for (int i = 0;i < n;i++)
    {
        wrong += TreeDiffNum(result[i], gold[i], label);
        // note that only gold tree has pun_num !!!!
        tot += gold[i].n - 1 - gold[i].punc_num; // without root and punctuation
    }
    return (tot - wrong)/tot;
}

void DepParser::UpdateState(SET &left_set,
                            SET &right_set,
                            int pos1,
                            int pos2,
                            int head,
                            int complete,
                            double score,
                            int K,
                            PQ *que,
                            SET *state_set,
                            vector<State*>* gabbage)
{
    SET::iterator iter, iter1, iter2;
    
    for (iter1 = left_set.begin(); iter1 != left_set.end(); iter1++)
        for (iter2 = right_set.begin(); iter2 != right_set.end(); iter2++)
        {
            State* left = (*iter1);
            State* right = (*iter2);
            double val = left->score + right->score + score;
            if (que->size() < K)
            {
                State* state = new State(pos1,pos2,head,complete,val,left,right);
                gabbage->push_back(state);
                que->push(state);
                state_set->insert(state);
            }
            else
            {
                State* top_state = que->top();
                if (val > top_state->score)
                {
                    State* state = new State(pos1,pos2,head,complete,val,left,right);
                    gabbage->push_back(state);
                    // erase the smallest state
                    iter = state_set->find(top_state);
                    state_set->erase(iter);
                    que->pop();
                    // Push in the new state
                    que->push(state);
                    state_set->insert(state);
                }
            }
        }
}

int DepParser::SaveBestK(const string &result_file, 
                         const vector<vector<SeqNode> > &data,
                         const vector<string> &id_to_word,
                         const vector<string> &id_to_pos,
                         const vector<string> &id_to_edge_type,
                         int K)
{
    FILE* fp = NULL;

    fp = fopen(result_file.c_str(), "w");
    if (fp == NULL)
    {
        return -1;
    }
    
    int data_size = data.size();

    for (int i = 0;i < data_size;i++)
    {
        vector<DepTree> k_best;
        GetBestK(data[i], &k_best, K);
        int list_size = k_best.size();
        fprintf(fp, "%d\n", list_size);
        for (int j = 0;j < list_size;j++)
        {
            SaveTree(fp, k_best[j], data[i], id_to_word, id_to_pos, id_to_edge_type);
        }
    }
    fclose(fp);

    return 0;
}
void DepParser::GetBestK(const vector<SeqNode> &seq, vector<DepTree>* k_best_list, int K)
{
    // In the future, we should change this to more adaptable code
    vector<State*> gabbage;
    const int INF = 1000000000;
    int sen_len = seq.size();
    // we use static variable because of the space
    static PQ dp[MAX_SEN_LEN][MAX_SEN_LEN][2][2];
    static SET state_set[MAX_SEN_LEN][MAX_SEN_LEN][2][2];
    vector<vector<int> > edge_type(sen_len, vector<int>(sen_len, -1));
    
    //printf("Get Sentence vector\n");
    MatrixXd sen_word_vec(1, (sen_len + 2*conf.window_size) * conf.word_embed_size);
    MatrixXd sen_pos_vec(1, (sen_len + 2*conf.window_size) * conf.pos_embed_size);
    GetSentenceVec(seq, &sen_word_vec, &sen_pos_vec);
    
    Cache cache(sen_len);
    PreCompute(sen_len, sen_word_vec, sen_pos_vec, &cache);

    for (int i = 0;i < sen_len;i++)
        for(int k = 0;k < 2;k++)
            for(int p = 0;p < 2;p++)
            {
                State* tmp = new State(i,i,k,p,0.0);
                gabbage.push_back(tmp);
                dp[i][i][k][p].push(tmp);
                state_set[i][i][k][p].insert(tmp);
            }
    
    // This variable is useless, it's only used as fake gold_tree in GetArcScore
    DepTree dummy_tree(sen_len);
    //printf("Start dp\n");
    for (int m = 1; m <= sen_len;m++)
        for(int s = 1; s < sen_len;s++)
        {
            int t = s + m;
            if ( t >= sen_len)
                break;

            double score_left = GetArcScore(s, t, 0, dummy_tree, 0.0, false, &cache, &edge_type);
            double score_right = GetArcScore(s, t, 1, dummy_tree, 0.0, false, &cache, &edge_type);
            for (int q = s;q < t;q++)
            {
                // Create subgraphs with c = 1 by adding arcs
                UpdateState(state_set[s][q][1][0], state_set[q+1][t][0][0], 
                            s, t, 0, 1, score_left, K, &dp[s][t][0][1], &state_set[s][t][0][1], &gabbage);
            }
            for (int q = s;q < t;q++)
            {
                // right arc
                UpdateState(state_set[s][q][1][0], state_set[q+1][t][0][0], 
                            s, t, 1, 1, score_right, K, &dp[s][t][1][1], &state_set[s][t][1][1], &gabbage);
            }
            for (int q = s;q < t;q++)
            {
                // Add corresponding left subgraphs, q==t is meaningless
                UpdateState(state_set[s][q][0][0], state_set[q][t][0][1], 
                            s, t, 0, 0, 0.0, K, &dp[s][t][0][0], &state_set[s][t][0][0], &gabbage);
            }
            for (int q = s + 1; q <= t;q++)
            {
                // Add corresponding right subgraphs, q==t is ok
                UpdateState(state_set[s][q][1][1], state_set[q][t][1][0], 
                            s, t, 1, 0, 0.0, K, &dp[s][t][1][0], &state_set[s][t][1][0], &gabbage);
            }
        }
    
    //printf("Start root dp\n");
    // Find the root, we assume there would be only one child for root node
    for (int i = 1;i < sen_len; i++)
    {
        double score = GetArcScore(0, i, 1, dummy_tree, 0.0, false, &cache, &edge_type);
        UpdateState(state_set[0][0][1][0], state_set[1][i][0][0], 
                    0, i, 1, 1, score, K, &dp[0][i][1][1], &state_set[0][i][1][1], &gabbage);
    }
    
    for (int i = 1;i < sen_len;i++)
    {
        // Final graph
        UpdateState(state_set[0][i][1][1], state_set[i][sen_len - 1][1][0], 
                    0, sen_len - 1, 1, 0, 0.0, K, &dp[0][sen_len - 1][1][0], &state_set[0][sen_len-1][1][0], &gabbage);
    }
    
    SET::iterator iter;
    //printf("Start get tree\n");
    for (iter = state_set[0][sen_len-1][1][0].begin(); iter != state_set[0][sen_len-1][1][0].end();iter++)
    {
        State* state = *iter;
        DepTree tree(sen_len);
        
        GetTree(state, edge_type, &tree);
        tree.score = state->score;
        k_best_list->push_back(tree);
    }
    // release mem
    int mem_size = gabbage.size();

    for (int i = 0;i < mem_size;i++)
    {
        State* state = gabbage[i];
        delete state;
    }
    // clear queue and set
    for (int i = 0;i < sen_len;i++)
        for (int j = 0;j < sen_len;j++)
            for (int k = 0;k < 2;k++)
                for (int p = 0;p < 2;p++)
                {
                    while(!dp[i][j][k][p].empty())
                    {
                        dp[i][j][k][p].pop();
                    }
                    state_set[i][j][k][p].clear();
                }
}

void DepParser::Fit(const vector<vector<SeqNode> > &train_data,
         const vector<vector<SeqNode> > &dev_data,
         const vector<vector<double> > &word_lt,
         const double alpha,
         const int batch_size,
         const int epoch_num,
         const double l2_reg,
         const double margin_reg)
{
    int train_data_size = train_data.size();
    int dev_data_size = dev_data.size();
    vector<DepTree> train_gold_trees;
    vector<DepTree> dev_gold_trees;
    vector<Embedding> best_embeddings;
    vector<Layer> best_layers;
    clock_t start_t, end_t;
    int mins = 0;
    double sec = 0.0;
    
    // Some preparition
    for (int i = 0;i < train_data_size;i++)
    {
        train_gold_trees.push_back(DepTree(train_data[i]));
    }
    for (int i = 0;i < dev_data_size;i++)
    {
        dev_gold_trees.push_back(DepTree(dev_data[i]));
    }
    CopyToMatrix(word_lt, &(word_embedding.lt));

    printf("Start training...\n");
    int batch_num = train_data_size / batch_size + 1;
    double best = 0.0;
    int drop_time = 0;
    int embed_num = embed_pointers.size();
    int layer_num = layer_pointers.size();

    for (int epoch = 0; epoch < epoch_num; epoch++)
    {
        printf("Start epoch %d\n", epoch + 1);
        fflush(stdout);
        start_t = clock();
        for (int i = 0;i < batch_num;i++)
        {
            int start = i * batch_size;
            if (start >= train_data_size)
                break;
            int end = min((i + 1)*batch_size, train_data_size);
            for (int j = start;j < end;j++)
            {
                int sen_len = train_data[j].size();
                DepTree best_tree(sen_len);
                MatrixXd sen_word_vec(1, (sen_len + 2*conf.window_size) * conf.word_embed_size);
                MatrixXd sen_pos_vec(1, (sen_len + 2*conf.window_size) * conf.pos_embed_size);
                
                //printf("Start GetSentenceVec\n");
                GetSentenceVec(train_data[j], &sen_word_vec, &sen_pos_vec);
                Cache cache(sen_len);
                //printf("Start Decode\n");
                Decode(sen_len, sen_word_vec, sen_pos_vec, train_gold_trees[j], margin_reg, true, &cache, &best_tree);
                // backprop
                //printf("Start Backprop\n");
                BackProp(best_tree, train_gold_trees[j], cache, train_data[j]);
                /*bool ok = GradientCheck(train_best_trees[j], train_gold_trees[j], train_data[j],
                                        cache.mask, pos_embedding.lt_grad, &pos_embedding.lt);
                if (!ok)
                {
                    printf("Gradient check fail!\n");
                    return;
                }*/
            }
            // update parameters;
            UpdateParam(alpha, l2_reg, end - start);
        }
        double uas = 0.0;
        double las = 0.0;
        Test(dev_data, dev_gold_trees, false, &uas, &las);
        printf("epoch %d, uas: %lf\tlas: %lf\n", epoch, uas*100, las*100);
        end_t = clock();
        sec = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        mins = (int)(sec/60.0);
        sec = sec - 60*mins;
        printf("Total time: %d mins %.2lfs\n", mins, sec);
        fflush(stdout);
        if (uas > best)
        {
            drop_time = 0;
            best = uas;
            best_embeddings.clear();
            best_layers.clear();
            for (int i = 0;i < embed_num;i++)
                best_embeddings.push_back(*embed_pointers[i]);
            for (int i = 0;i < layer_num;i++)
                best_layers.push_back(*layer_pointers[i]);
        }
        else
        {
            drop_time++;
            if (drop_time >= 3)
                break;
        }
    }
    // Change the weights to best weights
    for (int i = 0;i < embed_num;i++)
        (*embed_pointers[i]) = best_embeddings[i];
    for (int i = 0;i < layer_num;i++)
        (*layer_pointers[i]) = best_layers[i];
}

double DepParser::BackProp(const DepTree &best_tree,
                           const DepTree &gold_tree,
                           const Cache &cache,
                           const vector<SeqNode> &data)
{
    int n = best_tree.n;
    double score = 0.0;
    
    score = 0.0;
    for (int i = 1;i < n;i++) // omit head for root
    {
        int head_best = best_tree.head[i];
        int edge_type_best = best_tree.edge_type[i];
        int head_gold = gold_tree.head[i];
        int edge_type_gold = gold_tree.edge_type[i];
        if (head_best != head_gold || edge_type_best != edge_type_gold)
        {
            // Update only if head or edge_type is different
            BackPropArc(i, head_best, edge_type_best, 1, cache, data);
            BackPropArc(i, head_gold, edge_type_gold, -1, cache, data);
        }
        // Get gold tree score
        int arc_dir = 0;
        int start = 0, end = 0;
        if (head_gold < i)
        {
            arc_dir = 1;
            start = head_gold;
            end = i;
        }
        else
        {
            arc_dir = 0;
            start = i;
            end = head_gold;
        }
        score += cache.output_score[arc_dir][start][end](0, edge_type_gold);
    }

    return best_tree.score - score;
}

void DepParser::BackPropArc(int pos, int head, int edge_type, int sign,
                            const Cache &cache,
                            const vector<SeqNode> &data)
{
    int arc_dir = 0;
    int start = 0, end = 0;
    int dist = 0;

    if (head < pos)
    {
        arc_dir = 1;
        start = head;
        end = pos;
        dist = pos - head;
    }
    else
    {
        arc_dir = 0;
        start = pos;
        end = head;
        dist = head - pos;
    }
    // Backprop at output layer
    output_layer[arc_dir].W_grad.col(edge_type) += sign * cache.hidden_activate[arc_dir][start][end].transpose();
    output_layer[arc_dir].b_grad += sign * MatrixXd::Ones(1, conf.edge_type_num);
    // Backprop to hidden_activate
    MatrixXd hidden_activate_grad = output_layer[arc_dir].W.col(edge_type).transpose();
    hidden_activate_grad = hidden_activate_grad.array() * cache.mask.array();
    // Backprop to hidden input, activation is x^3
    MatrixXd activate_tanh_grad = 1.0 - cache.hidden_activate[arc_dir][start][end].array().square();
    MatrixXd activate_input_grad = 3.0 * (cache.hidden_input[arc_dir][start][end].array() * 
                                          cache.hidden_input[arc_dir][start][end].array()) + ArrayXXd::Ones(1, conf.n_hidden);
    MatrixXd hidden_input_grad = hidden_activate_grad.array() * activate_tanh_grad.array() * activate_input_grad.array();
    // Backprop to left_word layer
    left_layer_word[arc_dir].W_grad += sign * cache.position_input_word[start].transpose() * 
                                       hidden_input_grad;
    left_layer_word[arc_dir].b_grad += sign * hidden_input_grad;
    // Backprop to postion_left_word input
    MatrixXd position_left_word_grad = hidden_input_grad * left_layer_word[arc_dir].W.transpose();
    // Backprop to left_pos layer
    left_layer_pos[arc_dir].W_grad += sign * cache.position_input_pos[start].transpose() *
                                      hidden_input_grad;
    left_layer_pos[arc_dir].b_grad += sign * hidden_input_grad;
    // Backprop to position_left_pos input
    MatrixXd position_left_pos_grad = hidden_input_grad * left_layer_pos[arc_dir].W.transpose();
    // Backprop to right_word layer
    right_layer_word[arc_dir].W_grad += sign * cache.position_input_word[end].transpose() * 
                                        hidden_input_grad;
    right_layer_word[arc_dir].b_grad += sign * hidden_input_grad;
    // Backprop to position_right_word input
    MatrixXd position_right_word_grad = hidden_input_grad * right_layer_word[arc_dir].W.transpose();
    // Backprop to right_pos layer
    right_layer_pos[arc_dir].W_grad += sign * cache.position_input_pos[end].transpose() *
                                       hidden_input_grad;
    right_layer_pos[arc_dir].b_grad += sign * hidden_input_grad;
    // Backprop to position_right pos input
    MatrixXd position_right_pos_grad = hidden_input_grad * right_layer_pos[arc_dir].W.transpose();
    // Backprop to prefix word layer
    prefix_layer_word[arc_dir].W_grad += sign * cache.prefix_input_word[start].transpose() * 
                                         hidden_input_grad;
    prefix_layer_word[arc_dir].b_grad += sign * hidden_input_grad;
    MatrixXd prefix_word_grad = hidden_input_grad * prefix_layer_word[arc_dir].W.transpose();
    // Backprop to prefix pos layer
    prefix_layer_pos[arc_dir].W_grad += sign * cache.prefix_input_pos[start].transpose() * 
                                         hidden_input_grad;
    prefix_layer_pos[arc_dir].b_grad += sign * hidden_input_grad;
    MatrixXd prefix_pos_grad = hidden_input_grad * prefix_layer_pos[arc_dir].W.transpose();
    // Backprop to suffix word layer
    suffix_layer_word[arc_dir].W_grad += sign * cache.suffix_input_word[end].transpose() * 
                                         hidden_input_grad;
    suffix_layer_word[arc_dir].b_grad += sign * hidden_input_grad;
    MatrixXd suffix_word_grad = hidden_input_grad * suffix_layer_word[arc_dir].W.transpose();
    // Backprop to suffix pos layer
    suffix_layer_pos[arc_dir].W_grad += sign * cache.suffix_input_pos[end].transpose() * 
                                         hidden_input_grad;
    suffix_layer_pos[arc_dir].b_grad += sign * hidden_input_grad;
    MatrixXd suffix_pos_grad = hidden_input_grad * suffix_layer_pos[arc_dir].W.transpose();
    // Backprop to middel word layer
    middle_layer_word[arc_dir].W_grad += sign * cache.middle_input_word[start][end].transpose() *
                                         hidden_input_grad;
    middle_layer_word[arc_dir].b_grad += sign * hidden_input_grad;
    // Backprop to middle word input
    MatrixXd middle_word_grad = hidden_input_grad * middle_layer_word[arc_dir].W.transpose();
    // Backprop to middle pos layer
    middle_layer_pos[arc_dir].W_grad += sign * cache.middle_input_pos[start][end].transpose() *
                                        hidden_input_grad;
    middle_layer_pos[arc_dir].b_grad += sign * hidden_input_grad;
    // Backprop to middle pos input
    MatrixXd middle_pos_grad = hidden_input_grad * middle_layer_pos[arc_dir].W.transpose();
    // Backprop to dist layer
    dist_layer[arc_dir].W_grad += sign * dist_embedding.lt.row(dist).transpose() * 
                                  hidden_input_grad;
    dist_layer[arc_dir].b_grad += sign * hidden_input_grad;
    // Backprop to dist input
    MatrixXd dist_grad = hidden_input_grad * dist_layer[arc_dir].W.transpose();
    
    // Update corresponding embeddings
    // left word and pos
    int sen_len = data.size();
    int cnt = 0;
    for (int i = start - conf.window_size;i <= start + conf.window_size;i++)
    {
        int word_id = 0;
        int pos_id = 0;

        if (i < 0)
        {
            word_id = 0; // <BOS>
            pos_id = 0; // <EOS>
        }
        else if (i >= sen_len)
        {
            word_id = 1; // <EOS>
            pos_id = 1; // <EOS>
        }
        else
        {
            word_id = data[i].word;
            pos_id = data[i].pos;
        }
        int start_pos_word = cnt * conf.word_embed_size;
        int start_pos_pos = cnt * conf.pos_embed_size;
        word_embedding.lt_grad.row(word_id) += sign * position_left_word_grad.row(0).segment(start_pos_word, conf.word_embed_size);
        pos_embedding.lt_grad.row(pos_id) += sign * position_left_pos_grad.row(0).segment(start_pos_pos, conf.pos_embed_size);
        cnt++;
    }
    // right word and pos
    cnt = 0;
    for (int i = end - conf.window_size; i <= end + conf.window_size;i++)
    {
        int word_id = 0;
        int pos_id = 0;

        if (i < 0)
        {
            word_id = 0; // <BOS>
            pos_id = 0; // <EOS>
        }
        else if (i >= sen_len)
        {
            word_id = 1; // <EOS>
            pos_id = 1; // <EOS>
        }
        else
        {
            word_id = data[i].word;
            pos_id = data[i].pos;
        }
        int start_pos_word = cnt * conf.word_embed_size;
        int start_pos_pos = cnt * conf.pos_embed_size;
        word_embedding.lt_grad.row(word_id) += sign * position_right_word_grad.row(0).segment(start_pos_word, conf.word_embed_size);
        pos_embedding.lt_grad.row(pos_id) += sign * position_right_pos_grad.row(0).segment(start_pos_pos, conf.pos_embed_size);
        cnt++;
    }
    // update prefix word and pos
    for (int i = 0;i < start;i++)
    {
        int word_id = data[i].word;
        int pos_id = data[i].pos;
        word_embedding.lt_grad.row(word_id) += sign * prefix_word_grad / start;
        pos_embedding.lt_grad.row(pos_id) += sign * prefix_pos_grad / start;
    }
    // update suffix word and pos
    for (int i = end + 1;i < sen_len;i++)
    {
        int word_id = data[i].word;
        int pos_id = data[i].pos;
        word_embedding.lt_grad.row(word_id) += sign * suffix_word_grad / (sen_len - 1.0 - end);
        pos_embedding.lt_grad.row(pos_id) += sign * suffix_pos_grad / (sen_len - 1.0 - end);
    }
    // update middle word and pos
    for (int i = start; i <= end;i++)
    {
        int word_id = data[i].word;
        int pos_id = data[i].pos;
        word_embedding.lt_grad.row(word_id) += sign * middle_word_grad / (end - start + 1.0);
        pos_embedding.lt_grad.row(pos_id) += sign * middle_pos_grad / (end - start + 1.0);
    }
    // update dist
    dist_embedding.lt_grad.row(dist) += sign * dist_grad;
}

double DepParser::GetTreeScore(const vector<SeqNode> &data, const DepTree &tree, const MatrixXd &mask)
{
    int sen_len = data.size();
    MatrixXd sen_word_vec(1, (sen_len + 2*conf.window_size) * conf.word_embed_size);
    MatrixXd sen_pos_vec(1, (sen_len + 2*conf.window_size) * conf.pos_embed_size);
    
    GetSentenceVec(data, &sen_word_vec, &sen_pos_vec);
    double score = 0.0;
    for (int i = 1;i < sen_len;i++) // omit head for ROOT
    {
        int head = tree.head[i];
        int edge_type = tree.edge_type[i];
        int start = 0, end = 0;
        int arc_dir = 0;

        if (head < i)
        {
            arc_dir = 1;
            start = head;
            end = i;
        }
        else
        {
            arc_dir = 0;
            start = i;
            end = head;
        }
        // position left
        int start_word = start * conf.word_embed_size;
        int start_pos = start * conf.pos_embed_size;
        MatrixXd input_position_word = sen_word_vec.row(0).segment(start_word, (2*conf.window_size + 1)*conf.word_embed_size);
        MatrixXd input_position_pos = sen_pos_vec.row(0).segment(start_pos, (2*conf.window_size+1)*conf.pos_embed_size);
        MatrixXd position_left = input_position_word * left_layer_word[arc_dir].W + left_layer_word[arc_dir].b + 
                                 input_position_pos * left_layer_pos[arc_dir].W + left_layer_pos[arc_dir].b;
        // position right
        start_word = end * conf.word_embed_size;
        start_pos = end * conf.pos_embed_size;
        input_position_word = sen_word_vec.row(0).segment(start_word, (2*conf.window_size + 1)*conf.word_embed_size);
        input_position_pos = sen_pos_vec.row(0).segment(start_pos, (2*conf.window_size+1)*conf.pos_embed_size);
        MatrixXd position_right = input_position_word * right_layer_word[arc_dir].W + right_layer_word[arc_dir].b + 
                                 input_position_pos * right_layer_pos[arc_dir].W + right_layer_pos[arc_dir].b;
        // prefix
        MatrixXd prefix_input_word = MatrixXd::Zero(1, conf.word_embed_size);
        MatrixXd prefix_input_pos = MatrixXd::Zero(1, conf.pos_embed_size);
        for (int j = 0;j < start;j++)
        {
            prefix_input_word += word_embedding.lt.row(data[j].word);
            prefix_input_pos += pos_embedding.lt.row(data[j].pos);
        }
        if (start != 0)
        {
            prefix_input_word /= start;
            prefix_input_pos /= start;
        }
        MatrixXd prefix = prefix_input_word * prefix_layer_word[arc_dir].W + prefix_layer_word[arc_dir].b + 
                          prefix_input_pos * prefix_layer_pos[arc_dir].W + prefix_layer_pos[arc_dir].b;
        // suffix
        MatrixXd suffix_input_word = MatrixXd::Zero(1, conf.word_embed_size);
        MatrixXd suffix_input_pos = MatrixXd::Zero(1, conf.pos_embed_size);
        for (int j = end + 1;j < sen_len;j++)
        {
            suffix_input_word += word_embedding.lt.row(data[j].word);
            suffix_input_pos += pos_embedding.lt.row(data[j].pos);
        }
        if (end != sen_len - 1)
        {
            suffix_input_word /= (sen_len - 1 - end);
            suffix_input_pos /= (sen_len - 1 -end);;
        }
        MatrixXd suffix = suffix_input_word * suffix_layer_word[arc_dir].W + suffix_layer_word[arc_dir].b + 
                          suffix_input_pos * suffix_layer_pos[arc_dir].W + suffix_layer_pos[arc_dir].b;

        // middle
        MatrixXd middle_input_word = word_embedding.lt.row(data[start].word);
        MatrixXd middle_input_pos = pos_embedding.lt.row(data[start].pos);
        for (int j = start + 1;j <= end;j++)
        {
            middle_input_word += word_embedding.lt.row(data[j].word);
            middle_input_pos += pos_embedding.lt.row(data[j].pos);
        }
        middle_input_word /= (end - start + 1);
        middle_input_pos /= (end - start + 1);
        MatrixXd middle = middle_input_word * middle_layer_word[arc_dir].W + middle_layer_word[arc_dir].b + 
                          middle_input_pos * middle_layer_pos[arc_dir].W + middle_layer_pos[arc_dir].b;
        // dist
        MatrixXd dist = dist_embedding.lt.row(end - start) * dist_layer[arc_dir].W + dist_layer[arc_dir].b;
        MatrixXd hidden_input = position_left + position_right + prefix + suffix + middle + dist;
        MatrixXd hidden_input_activate(1, conf.n_hidden);
        Activate(hidden_input, &hidden_input_activate);
        hidden_input_activate = hidden_input_activate.array() * mask.array();
        MatrixXd output_score = hidden_input_activate * output_layer[arc_dir].W + output_layer[arc_dir].b;
        score += output_score(0, edge_type);
    }

    return score;
}

bool DepParser::GradientCheck(const DepTree &best_tree,
                              const DepTree &gold_tree,
                              const vector<SeqNode> &data,
                              const MatrixXd &mask,
                              const MatrixXd &my_grad,
                              MatrixXd* mat)
{
    int n = my_grad.rows();
    int m = my_grad.cols();
    const double eps = 1.0e-6;

    for (int i = 0;i < n;i++)
    {
        for (int j = 0;j < m;j++)
        {
            (*mat)(i,j) += eps;
            double score1 = GetTreeScore(data, best_tree, mask) - GetTreeScore(data, gold_tree, mask);
            (*mat)(i,j) -= 2*eps;
            double score2 = GetTreeScore(data, best_tree, mask) - GetTreeScore(data, gold_tree, mask);
            double gradient = (score1 - score2)/(2*eps);
            printf("At %d %d, gradient = %lf, my_gradient = %lf\n", i, j, gradient, my_grad(i,j));
            if (fabs(gradient - my_grad(i,j)) > eps)
                return false;
            (*mat)(i,j) += eps;
        }
    }

    return true;
}

void DepParser::UpdateParam(double alpha, double l2_reg, int batch_size)
{
    int embed_num = embed_pointers.size();
    int layer_num = layer_pointers.size();

    for (int i = 0;i < embed_num;i++)
    {
        embed_pointers[i]->UpdateParam(alpha, l2_reg, batch_size);
    }
    for (int i = 0;i < layer_num;i++)
    {
        layer_pointers[i]->UpdateParam(alpha, l2_reg, batch_size);
    }
}

void DepParser::Test(const vector<vector<SeqNode> > &test_data,
                     const vector<DepTree> &test_gold_trees,
                     bool dump,
                     double* uas,
                     double* las)
{
    int test_data_size = test_data.size();
    vector<DepTree> test_best_trees;
    clock_t start_t, end_t;


    for (int i = 0;i < test_data_size;i++)
        test_best_trees.push_back(DepTree(test_data[i].size()));
    
    start_t = clock();
    for (int i = 0;i < test_data_size;i++)
    {
        int sen_len = test_data[i].size();
        MatrixXd sen_word_vec(1, (sen_len + 2*conf.window_size) * conf.word_embed_size);
        MatrixXd sen_pos_vec(1, (sen_len + 2*conf.window_size) * conf.pos_embed_size);
                
        //printf("Start GetSentenceVec\n");
        GetSentenceVec(test_data[i], &sen_word_vec, &sen_pos_vec);
        Cache cache(sen_len);
        //printf("Start Decode\n");
        Decode(sen_len, sen_word_vec, sen_pos_vec, test_gold_trees[i], 0.0, false, &cache, &test_best_trees[i]);
    }
    end_t = clock();
    double sec = (end_t - start_t)/CLOCKS_PER_SEC;

    *uas = Evaluate(test_best_trees, test_gold_trees, false);
    *las = Evaluate(test_best_trees, test_gold_trees, true);
    printf("Speed: %.2lf sens/sec\n", test_data_size/sec);
}

void DepParser::SaveParam(const vector<string> &id_to_word,
                          const vector<string> &id_to_pos,
                          const vector<string> &id_to_edge_type)
{
    FILE* fp = NULL;
    string meta_file = conf.model_dir + "meta.txt";
    string word_embed_file = conf.model_dir + "word_embedding.txt";
    string pos_embed_file = conf.model_dir + "pos_embedding.txt";
    string edge_type_file = conf.model_dir + "edge_type.txt";
    string layer_weights_file = conf.model_dir + "layer_weights.txt";
    
    // save meta data, including word_dict size, pos_dict size
    // and edge_type_data size
    fp = fopen(meta_file.c_str(), "w");
    fprintf(fp, "%d\n", id_to_word.size());
    fprintf(fp, "%d\n", id_to_pos.size());
    fprintf(fp, "%d\n", id_to_edge_type.size());
    fclose(fp);
    // save word embedding and pos embedding
    word_embedding.SaveParam(word_embed_file, id_to_word);
    pos_embedding.SaveParam(pos_embed_file, id_to_pos);
    // save edge type
    fp = fopen(edge_type_file.c_str(), "w");
    int edge_type_num = id_to_edge_type.size();
    for (int i = 0;i < edge_type_num;i++)
        fprintf(fp, "%s\n", id_to_edge_type[i].c_str());
    fclose(fp);
    // save layer weights, including dist embedding
    fp = fopen(layer_weights_file.c_str(), "w");
    dist_embedding.SaveParam(fp);
    int layer_num = layer_pointers.size();
    for (int i = 0;i < layer_num;i++)
    {
        layer_pointers[i]->SaveParam(fp);
    }
    fclose(fp);
}

void DepParser::SaveParam()
{
    int embed_num = embed_pointers.size();
    int layer_num = layer_pointers.size();
    FILE* fp = NULL;
    string weights_file = conf.model_dir + "weights.txt";
    
    fp = fopen(weights_file.c_str(), "w");
    for (int i = 0;i < embed_num;i++)
    {
        embed_pointers[i]->SaveParam(fp);
    }
    for (int i = 0;i < layer_num;i++)
    {
        layer_pointers[i]->SaveParam(fp);
    }
    fclose(fp);
}

void DepParser::LoadParam(map<string, int> *word_dict,
                          map<string, int> *pos_dict,
                          map<string, int> *edge_type_dict,
                          vector<string> *id_to_word,
                          vector<string> *id_to_pos,
                          vector<string> *id_to_edge_type)
{
    FILE* fp = NULL;
    char buf[1024];
    string word_embed_file = conf.model_dir + "word_embedding.txt";
    string pos_embed_file = conf.model_dir + "pos_embedding.txt";
    string edge_type_file = conf.model_dir + "edge_type.txt";
    string layer_weights_file = conf.model_dir + "layer_weights.txt";
    
    // read word embedding and pos embedding
    word_embedding.LoadParam(word_embed_file, word_dict, id_to_word);
    pos_embedding.LoadParam(pos_embed_file, pos_dict, id_to_pos);
    // read edge type
    fp = fopen(edge_type_file.c_str(), "r");
    int edge_type_num = 0;
    while (fscanf(fp, "%s", buf) != EOF)
    {
        (*edge_type_dict)[buf] = edge_type_num;
        id_to_edge_type->push_back(buf);
        edge_type_num++;
    }
    fclose(fp);
    // read layer weights, including dist embedding
    fp = fopen(layer_weights_file.c_str(), "r");
    dist_embedding.LoadParam(fp);
    int layer_num = layer_pointers.size();
    for (int i = 0;i < layer_num;i++)
    {
        layer_pointers[i]->LoadParam(fp);
    }
    fclose(fp);
}

void DepParser::LoadParam()
{
    int embed_num = embed_pointers.size();
    int layer_num = layer_pointers.size();
    FILE* fp = NULL;
    string weights_file = conf.model_dir + "weights.txt";
    
    fp = fopen(weights_file.c_str(), "r");
    for (int i = 0;i < embed_num;i++)
    {
        embed_pointers[i]->LoadParam(fp);
    }
    for (int i = 0;i < layer_num;i++)
    {
        layer_pointers[i]->LoadParam(fp);
    }
    fclose(fp);
}

