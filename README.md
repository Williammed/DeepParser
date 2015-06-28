#A Graph-based dependency parser using Deep Learning
DeepParser is a graph-based dependency parser based on deep learning model. This work has been accepted by ACL2015 as oral long paper. The title of the paper is 《An Effecive Neural Network model for Graph-based Dependency Parsing》. The code is a little messy since I was rushing for the paper deadline. But you can use it to get the experiment results in our paper. I'm still refactoring the code and I'll update the project when I'm finished.  

##Project structures
<b>model</b>: Trained model for English parser and Chinese Parser <br>
<b>data</b>: Sample dependency data and trained word embeddings (via word2vec) <br>
<b>src</b>: Codes for our three models <br>

##Training and Testing
For compiling training code, go to <b>src</b> directory and run: <br>
<i>make dep_parser_train</i> <br>
<br>
For compiling testing code, go to <b>src</b> directory and run: <br>
<i>make dep_parser_test</i> <br>

