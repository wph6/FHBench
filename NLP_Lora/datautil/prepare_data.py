from datasets import load_dataset, Dataset, DatasetDict,concatenate_datasets
import os
from sklearn.preprocessing import LabelEncoder
from datautil.datasplit import getdataloader
def get_data(data_name):
    """Return the algorithm class with the given name."""
    datalist = {'med_abs': 'med_abs', 'pubmed': 'pubmed'}
    if datalist[data_name] not in globals():
        raise NotImplementedError("dataset not found: {}".format(data_name))
    return globals()[datalist[data_name]]

def convert_all_clients(partitions):
    datasets = [partition_to_dataset(part) for part in partitions]
    return datasets
def partition_to_dataset(partition):
    data_list = []
    label_list = []
    
    for i in range(len(partition)):
        text, label = partition[i]
        data_list.append(text)
        label_list.append(label)
    
    return Dataset.from_dict({
        'text': data_list,
        'labels': label_list
    })
def getlabeldataloader(args, data):
    trl, val, tel = getdataloader(args, data)
    train_datasets = convert_all_clients(trl)
    val_datasets = convert_all_clients(val)
    test_datasets = convert_all_clients(tel)
    return train_datasets,val_datasets,test_datasets

def pubmed(args):
    num_labels = 5
    data_dir = '/mnt/dataset/pubmed-rct/PubMed_200k_RCT/'
    train_data=preprocessing_text_with_line_number(data_dir,'train.txt')
    test_data=preprocessing_text_with_line_number(data_dir,'test.txt')

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_data['labels'])
    test_labels_encoded = label_encoder.transform(test_data['labels'])

    train_data['labels'] = train_labels_encoded.tolist()
    test_data['labels'] = test_labels_encoded.tolist()

    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    dataset = concatenate_datasets([train_dataset, test_dataset])

    subset_size = int(len(dataset) * args.datapercent) 
    indices = args.random_state.choice(len(dataset), subset_size, replace=False)  
    dataset = dataset.select(indices)

    trd, vad, ted = getlabeldataloader(args, dataset)
    return trd, vad, ted, num_labels

def med_abs(args):
    num_labels = 5
    data_files = {
        'train': '/mnt/dataset/Medical-Abstracts-TC-Corpus/medical_tc_train.csv',
        'test': '/mnt/dataset/Medical-Abstracts-TC-Corpus/medical_tc_test.csv'
    }
    dataset = load_dataset('csv',  data_files=data_files)
    dataset = dataset.map(lambda example: {'labels': example['condition_label']-1, 'text': example['medical_abstract']})
    
    dataset = concatenate_datasets([dataset['train'], dataset['test']])

    subset_size = int(len(dataset) * args.datapercent) 
    indices = args.random_state.choice(len(dataset), subset_size, replace=False)  
    dataset = dataset.select(indices)

    trd, vad, ted = getlabeldataloader(args, dataset)
    return trd, vad, ted, num_labels

def preprocessing_text_with_line_number(data_dir,filename):
    """
        Takes a filename and extract all lines then make a list of dictionaries,
        For each instance each one contains :
            line_number
            target
            text
            total_lines = number of lines in each paragraph
    """
    def get_lines(filename):
        """
            reads a text file and return all lines in a list.
        """
        with open (filename,'r') as file:
            return file.readlines()
        
    input_lines = get_lines(data_dir + filename)
    text_blog=""
    samples = {'labels': [], 'text': []}

    for line in input_lines:
        if line.startswith('###'):
            text_blog=""
        elif line.isspace():
            all_lines=text_blog.splitlines()
            for curr_line in all_lines:
                label,text=curr_line.split('\t')
                samples['labels'].append(label.strip())
                samples['text'].append(text.strip().lower())
        else:
            text_blog += line 
    return samples