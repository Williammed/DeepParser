#A Graph-based dependency parser using Deep Learning
DeepParser is a graph-based dependency parser based on deep learning model. This work has been accepted by ACL2015 as oral long paper. <br>

##Project structures
<b>model</b>: Trained model for English parser and Chinese Parser <br>
<b>data</b>: Sample dependency data and trained word embeddings (via word2vec) <br>
<b>src</b>: Codes for our three models <br>

##Training and Testing
For compiling training code, go to <b>src</b> directory and run: <br>
make dep_parser_train <br>
<br>
For compiling testing code, go to <b>src</b> directory and run: <br>
make dep_parser_test <br>

