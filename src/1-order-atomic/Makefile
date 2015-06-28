CC = g++
OBJ = data_io.o util.o dep_tree.o dep_parser.o
CFLAGS = -I ../../eigen/ -O2 -march=native -funroll-loops -Wno-unused-result
TARGET = dep_parser_train
TEST = dep_parser_test

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) train.cc
$(TEST): $(OBJ)
	$(CC) $(CFLAGS) -o $(TEST) $(OBJ) test.cc

$(OBJ): %.o:%.cc
	$(CC) $(CFLAGS) -c $< -o $@

all: $(TARGET)
clean:
	@rm -rf *.o $(TARGET) $(TEST)
