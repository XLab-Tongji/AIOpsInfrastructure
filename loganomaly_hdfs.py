
import os
from extractfeature import template2Vec_preprocessor
from anomalydetection.loganomaly import LogAnomaly_Train
from anomalydetection.loganomaly import LogAnomaly_Test
from anomalydetection.loganomaly import FindThresholdValue


# parameters for early prepare
logparser_structed_file = './Data/logparser_result/Drain/HDFS.log_structured.csv'
logparser_event_file = './Data/logparser_result/Drain/HDFS.log_templates.csv'
anomaly_label_file = './Data/log/hdfs/anomaly_label.csv'
train_root_path = './Data/DrainResult-HDFS/log_anomaly_new/'

sequential_directory = train_root_path + 'sequential_files/'
train_file_name = 'logano_train_file'
test_file_name = 'logano_test_file'
valid_file_name = 'logano_valid_file'

wordvec_file_path = './Data/pretrainedwordvec/crawl-300d-2M.vec'
pattern_vec_out_path = train_root_path + 'pattern_vec(avg)'
variable_symbol = '<*>'

# log anomaly sequential model parameters some parameter maybe changed to train similar models
hidden_size = 128
num_of_layers = 2
num_of_classes = 48
num_epochs = 50

window_length = 5
input_size_sequential = 300
input_size_quantitive = 48
batch_size = 512
# for log anomaly
model_out_path = train_root_path + 'model_out/'
train_file = sequential_directory + train_file_name
valid_file = sequential_directory + valid_file_name
pattern_vec_file = pattern_vec_out_path

# predict parameters
num_of_candidates = 5
threshold = 0.20812571048736572
# log anomaly sequential model parameters

if not os.path.exists(sequential_directory):
    os.makedirs(sequential_directory)
if not os.path.exists(train_root_path):
    os.makedirs(train_root_path)



# 同时生成train file 和 test file好点
def extract_feature():
    template2Vec_preprocessor.generate_train_and_test_file(logparser_structed_file, logparser_event_file, anomaly_label_file, sequential_directory, train_file_name, valid_file_name, test_file_name, wordvec_file_path, pattern_vec_out_path, variable_symbol)


def train_model():
    LogAnomaly_Train.train_model(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
                num_epochs, batch_size, train_root_path, model_out_path, train_file, pattern_vec_file)

def pattern_to_vec():
    template2Vec_preprocessor.pattern_to_vec_template_from_log(logparser_event_file, wordvec_file_path, pattern_vec_out_path, variable_symbol)

def test_model():
    # do something
    LogAnomaly_Test.do_predict(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers, num_of_classes,
               model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt',
               sequential_directory + valid_file_name, pattern_vec_file, num_of_candidates, threshold, batch_size)

def cal_threshold():
    FindThresholdValue.get_threshold_value(window_length, input_size_sequential, input_size_quantitive, hidden_size, num_of_layers,
                        num_of_classes,
                        model_out_path + 'Adam_batch_size=' + str(batch_size) + ';epoch=' + str(num_epochs) + '.pt',
                        valid_file, pattern_vec_out_path, batch_size)


#extract_feature()
#pattern_to_vec()
#print(cal_threshold())
#train_model()
#cal_threshold()
test_model()

# deep log
# log_preprocessor.execute_process()
# value_extract.get_value()
# value_extract.value_deal()
# value_extract.value_extract()
# train predict

