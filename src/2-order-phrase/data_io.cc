#include "data_io.h"
#include "util.h"
#include "dep_tree.h"

void AddSpecialPOS(const string &POS,
                  map<string, int>* pos_dict,
                  vector<string>* id_to_pos)
{
    int cur_size = pos_dict->size();
    (*pos_dict)[POS] = cur_size;
    id_to_pos->push_back(POS);
}
// Read the dependency data and encode them
// return the encoded data and updated pos && edge type dict
int ReadDepData(const string &file_name, 
                map<string, int> &word_dict,
                map<string, int> *pos_dict,
                vector<string>* id_to_pos,
                map<string, int> *edge_type_dict,
                vector<string>* id_to_edge_type,
                vector<vector<SeqNode> > *out)
{
    vector<SeqNode> seq;
    SeqNode root;
    bool first = true;
    bool chinese = false;
    map<string, int>::iterator iter;
    ifstream fin(file_name.c_str());
    
    // First Add <BOS> and <EOS> to the POS dict
    // So <BOS> id is always 0 and <EOS> 1
    AddSpecialPOS("<BOS>", pos_dict, id_to_pos);
    AddSpecialPOS("<EOS>", pos_dict, id_to_pos);
    
    // The ROOT node
    AddSpecialPOS("<ROOT>", pos_dict, id_to_pos);
    root.word = word_dict["<ROOT>"];
    root.pos = 2; // The ID for root
    root.punc = false; // it's not punc
    root.head = -1; // The head for root is meaningless
    root.type = -1; // The head for root is meaningless
    //Add ROOT node
    seq.push_back(root);

    if (fin.is_open())
    {
        string buf;

        while (getline(fin, buf))
        {
            if (buf.size() != 0 && (buf[buf.size() - 1] == '\n' || buf[buf.size() - 1] == '\r'))
            {
                buf = buf.substr(0, buf.size() - 1);
            }
            
            if (buf.size() == 0) // end of sen
            {
                out->push_back(seq);
                seq.clear();
                // Add ROOT
                seq.push_back(root);
            }
            else
            {
                vector<string> field;
                SeqNode tmp;

                Split(buf, '\t', &field);

                if (first)
                {
                    chinese = IsChinese(field[0]);
                    first = false;
                }
                // encode word
                string word(field[0]);
                if ( IsDigit(field[0], field[1], chinese) )
                    word = "<NUM>";
                iter = word_dict.find(word);
                if (iter != word_dict.end())
                    tmp.word = iter->second;
                else
                    tmp.word = word_dict["<OOV>"];
                // encode pos
                iter = pos_dict->find(field[1]);
                if (iter != pos_dict->end())
                    tmp.pos = iter->second;
                else
                {
                    tmp.pos = pos_dict->size();
                    (*pos_dict)[field[1]] = tmp.pos;
                    id_to_pos->push_back(field[1]);
                }
                // whether this is punc
                tmp.punc = IsPunc(field[1]);
                // save head
                tmp.head = atoi(field[2].c_str());
                // encode edge type
                iter = edge_type_dict->find(field[3]);
                if (iter != edge_type_dict->end())
                    tmp.type = iter->second;
                else
                {
                    tmp.type = edge_type_dict->size();
                    (*edge_type_dict)[field[3]] = tmp.type;
                    id_to_edge_type->push_back(field[3]);
                }
                seq.push_back(tmp);
            }
        }

        fin.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

// Read the dependency data
int ReadDepData(const string &file_name, 
                map<string, int> &word_dict,
                map<string, int> &pos_dict,
                map<string, int> &edge_type_dict,
                vector<vector<SeqNode> > *out)
{
    vector<SeqNode> seq;
    SeqNode root;
    bool first = true;
    bool chinese = false;
    map<string, int>::iterator iter;
    ifstream fin(file_name.c_str());
    
    root.word = word_dict["<ROOT>"];
    root.pos = 2; // The ID for root
    root.punc = false; // it's not punc
    root.head = -1; // The head for root is meaningless
    root.type = -1; // The head for root is meaningless
    //Add ROOT node
    seq.push_back(root);

    if (fin.is_open())
    {
        string buf;

        while (getline(fin, buf))
        {
            if (buf.size() != 0 && (buf[buf.size() - 1] == '\n' || buf[buf.size() - 1] == '\r'))
            {
                buf = buf.substr(0, buf.size() - 1);
            }

            if (buf.size() == 0) // end of sen
            {
                out->push_back(seq);
                seq.clear();
                // Add ROOT
                seq.push_back(root);
            }
            else
            {
                vector<string> field;
                SeqNode tmp;

                Split(buf, '\t', &field);
                
                if (first)
                {
                    chinese = IsChinese(field[0]);
                    first = false;
                }
                // encode word
                string word(field[0]);
                if ( IsDigit(field[0], field[1], chinese) )
                    word = "<NUM>";
                iter = word_dict.find(word);
                if (iter != word_dict.end())
                    tmp.word = iter->second;
                else
                    tmp.word = word_dict["<OOV>"];
                // encode pos, we assume no oov
                tmp.pos = pos_dict[field[1]];
                // whether this is punc
                tmp.punc = IsPunc(field[1]);
                // save head
                tmp.head = atoi(field[2].c_str());
                // encode edge type
                tmp.type = edge_type_dict[field[3]];
                seq.push_back(tmp);
            }
        }

        fin.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

// Read the TopK data
int ReadTopK(const string &file_name, 
             map<string, int> &word_dict,
             map<string, int> &pos_dict,
             map<string, int> &edge_type_dict,
             vector<vector<DepTree> > *out)
{
    vector<SeqNode> seq;
    SeqNode root;
    bool first = true;
    bool chinese = false;
    map<string, int>::iterator iter;
    ifstream fin(file_name.c_str());
    
    root.word = word_dict["<ROOT>"];
    root.pos = 2; // The ID for root
    root.punc = false; // it's not punc
    root.head = -1; // The head for root is meaningless
    root.type = -1; // The head for root is meaningless
    //Add ROOT node
    seq.push_back(root);

    if (fin.is_open())
    {
        string buf;

        while (getline(fin, buf))
        {
            int list_size = atoi(buf.c_str());
            vector<DepTree> k_best_list;

            while(getline(fin, buf))
            {
                if (buf.size() == 0) // end of sen
                {
                    k_best_list.push_back(DepTree(seq));
                    seq.clear();
                    // Add root
                    seq.push_back(root);
                    list_size--;
                    if (list_size == 0)
                    {
                        out->push_back(k_best_list);
                        break;
                    }
                }
                else
                {
                    vector<string> field;
                    SeqNode tmp;

                    Split(buf, '\t', &field);

                    if (first)
                    {
                        chinese = IsChinese(field[0]);
                        first = false;
                    }
                    // encode word
                    string word(field[0]);
                    if ( IsDigit(field[0], field[1], chinese) )
                        word = "<NUM>";
                    iter = word_dict.find(word);
                    if (iter != word_dict.end())
                        tmp.word = iter->second;
                    else
                        tmp.word = word_dict["<OOV>"];
                    // encode pos, we assume no oov
                    tmp.pos = pos_dict[field[1]];
                    // whether this is punc
                    tmp.punc = IsPunc(field[1]);
                    // save head
                    tmp.head = atoi(field[2].c_str());
                    // encode edge type
                    tmp.type = edge_type_dict[field[3]];
                    seq.push_back(tmp);
                }
            }
        }

        fin.close();
    }
    else
    {
        return -1;
    }

    return 0;
}

// Read word embedding and return the word embeeding 
// and update the word_dict

int ReadWordEmbedding(const string &file_name,
                      map<string, int> *word_dict,
                      vector<string>* id_to_word,
                      vector<vector<double> > *word_embedding)
{
    ifstream fin(file_name.c_str());

    if (fin.is_open())
    {
        string buf;

        while (getline(fin, buf))
        {
            vector<double> vec;
            vector<string> field;

            // get rid of \n
            if (buf.size() != 0 && (buf[buf.size() - 1] == '\n' || buf[buf.size() - 1] == '\r'))
            {
                buf = buf.substr(0, buf.size() - 1);
            }

            Split(buf, ' ', &field);
            if (word_dict->find(field[0]) == word_dict->end())
            {
                int id = word_dict->size();
                (*word_dict)[field[0]] = id;
                id_to_word->push_back(field[0]);
            }
            int embed_size = field.size() - 1;

            for (int i = 0;i < embed_size;i++)
                vec.push_back(atof(field[i + 1].c_str()));
            word_embedding->push_back(vec);
        }
    }
    else
    {
        return -1;
    }

    return 0;
}
