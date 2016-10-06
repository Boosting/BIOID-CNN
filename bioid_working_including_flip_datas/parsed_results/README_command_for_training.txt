1. Edit and include within the train.sh file in caffe: 
" ... 2>&1 | tee <log file path>" ==> to redirect the stderr and stdout and record the outputs to a log file 

2. Parse the informations generated during the training process 
python ~/caffe/tools/extra/parse_log.py <log_file_generated_during_training> <path_to_store_model>

3. gnuplot -persist <gnuplot_commands_file> 
will plot the results (training learning rate and test learning rate, and also the accuracy) 

# Format in train.log
# NumIters,Seconds,LearningRate,loss

# Format in test.log 
# NumIters,Seconds,LearningRate,accuracy,loss

set datafile separator ','
# new plot window 
set term x11 0
# plot between NumIters and train.loss (column 1 and column 4)
plot '/home/wzleong/caffe/examples/mnist/result_full_convolution/lenet_model_train.log.train' using 1:4 title "train loss",\
# plot between NumIters and test.loss (column 1 and column 5)
     '/home/wzleong/caffe/examples/mnist/result_full_convolution/lenet_model_train.log.test' using 1:5 with line
set term x11 1
plot '/home/wzleong/caffe/examples/mnist/result_full_convolution/lenet_model_train1.log.train' using 1:4  with line,\
     '/home/wzleong/caffe/examples/mnist/result_full_convolution/lenet_model_train1.log.test' using 1:4 with line,\
     '/home/wzleong/caffe/examples/mnist/result_full_convolution/lenet_model_train1.log.test' using 1:5 with line
