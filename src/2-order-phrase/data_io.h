#ifndef DATA_IO_H
#define DATA_IO_H

#include "data_type.h"
#include "dep_tree.h"

// read the dependency data from file
int ReadDepData(const string &file_name,
                map<string, int> &word_dict,
                map<string, int> *pos_dict,
                vector<string> *id_to_pos,
                map<string, int> *edge_type_dict,
                vector<string> *id_to_edge_type,
                vector<vector<SeqNode> > *out);

// read the dependency data from file
int ReadDepData(const string &file_name,
                map<string, int> &word_dict,
                map<string, int> &pos_dict,
                map<string, int> &edge_type_dict,
                vector<vector<SeqNode> > *out);

// read topk tree from file
int ReadTopK(const string &file_name, 
             map<string, int> &word_dict,
             map<string, int> &pos_dict,
             map<string, int> &edge_type_dict,
             vector<vector<DepTree> > *out);

// read word embedding from file
int ReadWordEmbedding(const string &file_name,
                      map<string, int>* word_dict,
                      vector<string>* id_to_word,
                      vector<vector<double> >*  word_embedding);
#endif
