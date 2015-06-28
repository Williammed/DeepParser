#include "data_io.h"
#include "dep_parser.h"

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        printf("Usage: ./dep_parser_train train_data dev_data word_embedding config_file\n");
        return 1;
    }
    
    map<string, int> word_dict;
    vector<string> id_to_word;
    map<string, int> pos_dict;
    vector<string> id_to_pos;
    map<string, int> edge_type_dict;
    vector<string> id_to_edge_type;
    vector<vector<double> > word_embedding;
    vector<vector<SeqNode> > train_data;
    vector<vector<SeqNode> > dev_data;
    int ret = 0;
    
    printf("Reading word embedding...\n");
    ret = ReadWordEmbedding(argv[3], &word_dict, &id_to_word, &word_embedding);
    if (ret == -1)
    {
        printf("Reading word embedding fail!\n");
        return -1;
    }
    printf("Reading training data...\n");
    ret = ReadDepData(argv[1], word_dict, &pos_dict, &id_to_pos, 
                          &edge_type_dict, &id_to_edge_type, &train_data);
    if (ret == -1)
    {
        printf("Reading training data fail!\n");
        return -1;
    }
    printf("Reading dev data...\n");
    ret = ReadDepData(argv[2], word_dict, &pos_dict, &id_to_pos,
                      &edge_type_dict, &id_to_edge_type, &dev_data);
    if (ret == -1)
    {
        printf("Reading dev data fail!\n");
        return -1;
    }
    
    Config config;

    ReadConfig(argv[4], &config);

    config.word_num = word_dict.size();
    config.pos_num = pos_dict.size();
    config.edge_type_num = edge_type_dict.size();
    
    unsigned int t = (unsigned int)time(0);
    printf("random seed = %u\n", t);
    srand(t);
    DepParser dep_parser(config);
    
    dep_parser.Fit(train_data, dev_data, word_embedding, 
               config.alpha, config.batch_size, config.epoch_num, 
               config.l2_reg, config.margin_reg);
    dep_parser.SaveParam(id_to_word, id_to_pos, id_to_edge_type);
    
    return 0;
}
