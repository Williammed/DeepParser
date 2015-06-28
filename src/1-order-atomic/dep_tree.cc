#include "dep_tree.h"

bool CmpDepTree(const DepTree& tree1, const DepTree& tree2)
{
    return tree1.score > tree2.score;
}

int TreeDiffNum(const DepTree& best_tree, const DepTree& gold_tree, const bool label)
{
    int wrong = 0;
    int n = gold_tree.n;

    for(int i = 1;i < n;i++) // ignore head for ROOT
    {
        int head1 = best_tree.head[i];
        int edge_type1 = best_tree.edge_type[i];
        int head2 = gold_tree.head[i];
        int edge_type2 = gold_tree.edge_type[i];
        
        // Don't count punc in evaluation
        if (gold_tree.punc[i] == true)
            continue;

        if (head1 != head2)
            ++wrong;
        else
        {
            if (label && edge_type1 != edge_type2) // labeled evaluation
            {
                ++wrong;
            }
        }
    }

    return wrong;
}

void PrintTree(const DepTree& tree, 
               const vector<SeqNode> &seq, 
               const vector<string> &id_to_word,
               const vector<string> &id_to_edge_type)
{
    int n = tree.n; // number of node

    for (int i = 1;i < n;i++) // ignore head for ROOT
    {
        int word_id = seq[i].word;
        int head = tree.head[i];
        int edge_type_id = tree.edge_type[i];

        string word = id_to_word[word_id];
        string edge_type = id_to_edge_type[edge_type_id];

        printf("%s\t%d\t%s\n", word.c_str(), head, edge_type.c_str());
    }
    printf("\n");
}

void SaveTree(FILE* fp,
              const DepTree& tree, 
              const vector<SeqNode> &seq, 
              const vector<string> &id_to_word,
              const vector<string> &id_to_pos,
              const vector<string> &id_to_edge_type)
{
    int n = tree.n; // number of node

    for (int i = 1;i < n;i++) // ignore head for ROOT
    {
        int word_id = seq[i].word;
        int pos_id = seq[i].pos;
        int head = tree.head[i];
        int edge_type_id = tree.edge_type[i];
        
        string word = id_to_word[word_id];
        string pos = id_to_pos[pos_id];
        string edge_type = id_to_edge_type[edge_type_id];

        fprintf(fp, "%s\t%s\t%d\t%s\n", word.c_str(), pos.c_str(), head, edge_type.c_str());
    }
    fprintf(fp, "\n");
}
