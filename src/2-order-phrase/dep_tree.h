#ifndef DEP_TREE_H_
#define DEP_TREE_H_

#include "data_type.h"

class DepTree
{
public:
    DepTree(int node_num):n(node_num), 
                          head(vector<int>(node_num, -1)),
                          edge_type(vector<int>(node_num, -1)),
                          punc(vector<bool>(node_num, false)),
                          children(vector<vector<int> >(node_num, vector<int>())),
                          sibling(node_num, -1)
    {
        score = 0.0;
        punc_num = 0;
    }
    DepTree(const vector<SeqNode> &seq)
    {
        int len = seq.size();

        n = len;
        score = 0.0;
        punc_num = 0;
        children = vector<vector<int> >(len, vector<int>());
        sibling = vector<int>(len, -1);

        for (int i = 0;i < len;i++)
        {
            int head_id = seq[i].head;
            int edge_type_id = seq[i].type;
            bool is_punc = seq[i].punc;

            head.push_back(head_id);
            if (head_id != -1) // current node is not ROOT
                children[head_id].push_back(i);
            edge_type.push_back(edge_type_id);
            punc.push_back(is_punc);
            if (is_punc)
                punc_num++;
        }
        GetSibling();
    }
    void GetSibling()
    {
        int arc_dir = 0;

        for (int i = 0;i < n;i++)
        {
            sort(children[i].begin(), children[i].end());
            int child_num = children[i].size();
            for (int j = 0;j < child_num;j++)
            {
                int child = children[i][j];
                if (child < i && j + 1 < child_num)
                {
                    // this is a left child, find the right sibling
                    int sib = children[i][j + 1];
                    if (sib < i) // the same side
                    {
                        sibling[child] = sib;
                    }
                }
                else if(child > i && j >= 1)
                {
                    // this is a right child, find the left sibling
                    int sib = children[i][j - 1];
                    if (sib > i) // the same side
                    {
                        sibling[child] = sib;
                    }
                }
            }
        }
    }
    void GetType(const vector<vector<int> > edge_type_best[])
    {
        for (int i = 1;i < n;i++) // omit ROOT
        {
            int head_id = head[i];
            int sib_id = sibling[i];
            if (sib_id == -1)
                sib_id = head_id;
            edge_type[i] = edge_type_best[sib_id][head_id][i];
        }
    }
    int n; // number of node
    int punc_num; // number of punctuation
    vector<int> head; // head for each node
    vector<int> edge_type; // edge type for each edge
    vector<vector<int> > children; // children for each node
    vector<int> sibling; // sibling for each node
    vector<bool> punc;
    double score; // score for the tree
};

// Cmp function for DepTree
bool CmpDepTree(const DepTree& tree1, const DepTree& tree2);

// Get the number of different edge between best_tree and gold_tree
int TreeDiffNum(const DepTree& best_tree, const DepTree& gold_tree, const bool label);

// Print the dep edge of a tree
void PrintTree(const DepTree& tree,
               const vector<SeqNode> &seq,
               const vector<string> &id_to_word,
               const vector<string> &id_to_edge_type);

// Save the dep tree into file
void SaveTree(FILE* fp,
              const DepTree& tree, 
              const vector<SeqNode> &seq, 
              const vector<string> &id_to_word,
              const vector<string> &id_to_pos,
              const vector<string> &id_to_edge_type);

#endif
