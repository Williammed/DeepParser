#include "data_io.h"
#include "dep_parser.h"

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("Usage: ./dep_parser_test test_data config_file\n");
        return 1;
    }
    
    map<string, int> word_dict;
    vector<string> id_to_word;
    map<string, int> pos_dict;
    vector<string> id_to_pos;
    map<string, int> edge_type_dict;
    vector<string> id_to_edge_type;
    vector<vector<double> > word_embedding;
    vector<vector<SeqNode> > test_data;
    int ret = 0;
    
    printf("Loading model...\n");
    Config config;

    ReadConfig(argv[2], &config);
    ReadMeta(&config);

    srand((unsigned int) time(0));
    DepParser dep_parser(config);
    dep_parser.LoadParam(&word_dict, &pos_dict, &edge_type_dict,
                         &id_to_word, &id_to_pos, &id_to_edge_type);

    printf("Reading test data...\n");
    ret = ReadDepData(argv[1], word_dict, pos_dict, edge_type_dict, &test_data);
    if (ret == -1)
    {
        printf("Reading dev data fail!\n");
        return -1;
    }
    vector<DepTree> test_gold_trees;
    int test_data_size = test_data.size();

    for (int i = 0;i < test_data_size;i++)
    {
        test_gold_trees.push_back(DepTree(test_data[i]));
    }
    printf("Start testing...\n");
    double uas = 0.0;
    double las = 0.0;

    dep_parser.Test(test_data, test_gold_trees, false, &uas, &las);
    printf("UAS: %lf\tLAS: %lf\n", uas * 100, las * 100);
    return 0;
}
