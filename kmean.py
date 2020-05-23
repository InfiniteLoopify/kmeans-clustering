# import time
import numpy as np
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random
from statistics import mode


class Indexer:
    def __init__(self):
        self.set = None
        self.label = None
        self.label_id = None

        self.classes_names = ['tennis', 'football',
                              'athletics', 'cricket', 'rugby']

        self.filter = 0
        self.rand_seed = 1
        self.rand_seed_cluster = 29
        self.class_count = 5

        self.vsm = None
        self.features = {}
        self.vsm_init_size = 14000
        self.lemmatize = WordNetLemmatizer()

        self.result = None
        self.param = None
        self.purity = None

    def randomize_array(self):
        """ randomize entries to get better set distribution and split them """

        random.seed(self.rand_seed)
        for i in range(0, self.vsm.shape[0]):
            rand = random.randint(0, self.vsm.shape[0]-1)
            self.vsm[[i, rand]] = self.vsm[[rand, i]]
            self.label[[i, rand]] = self.label[[rand, i]]
            self.label_id[[i, rand]] = self.label_id[[rand, i]]

    def read_file(self, docs_path, file_path):
        """ read vsm and other data files, if they exist, else read from documents and create vsm model"""

        if os.path.isfile(file_path+'result.npy') and\
           os.path.isfile(file_path+'param.npy') and\
           os.path.isfile(file_path+'label.npy') and\
           os.path.isfile(file_path+'label_id.npy'):
            self.result = np.load(file_path+'result.npy')
            self.param = np.load(file_path+'param.npy')
            self.label = np.load(file_path+'label.npy')
            self.label_id = np.load(file_path+'label_id.npy')

            if os.path.isfile(file_path+'set.npy'):
                self.set = np.load(file_path+'set.npy')
        else:
            if os.path.isfile(file_path+'set.npy') and\
                    os.path.isfile(file_path+'label.npy') and\
                    os.path.isfile(file_path+'label_id.npy'):
                self.set = np.load(file_path+'set.npy')
                self.label = np.load(file_path+'label.npy')
                self.label_id = np.load(file_path+'label_id.npy')

            else:
                self.read_from_documents(docs_path, file_path)
            self.calculate(file_path)

    def read_from_documents(self, docs_path, file_path):
        """ read all documents one by one and tokenize words to create vsm """

        # read and lemmatize stopwords
        stop_words = [re.sub('[,\'\n]', '', self.lemmatize.lemmatize(
            word)) for word in set(stopwords.words('english'))]

        dir_lists = os.listdir(docs_path)
        total_files = sum([len(os.listdir(docs_path+pth))
                           for pth in dir_lists])

        # initialize vector space model and labels with zeros
        self.vsm = np.zeros(shape=(total_files, self.vsm_init_size))
        self.label = np.zeros(total_files)
        self.label_id = np.zeros(total_files)

        # open every document, and index and tokenize the words in them to create vsm
        tokenize_regex = r"[^\w]"
        file_index = 0
        for dir_no, dir_list in enumerate(dir_lists):
            file_list = os.listdir(docs_path+dir_list)
            file_list = sorted(file_list, key=lambda x: int(
                "".join([i for i in x if i.isdigit()])))
            for doc_id, file_name in enumerate(file_list):
                self.label[file_index] = dir_no
                self.label_id[file_index] = doc_id+1
                with open(docs_path+dir_list+"/"+file_name, 'r', encoding='UTF-8') as file_data:
                    for line in file_data:
                        # split and tokenize word by given charecters
                        for word in re.split(tokenize_regex, line):
                            self.tokenize(word, file_index, stop_words)
                file_index += 1
        self.store_calculation(file_path)

    def store_calculation(self, file_path):
        """ store data into file """

        # delete unused columns in the vector
        self.delete_extra_cols()

        # filter out less used features
        temp_df = np.count_nonzero(self.vsm > 0, axis=0)
        del_index = [i for i, x in enumerate(temp_df) if x < self.filter]
        self.vsm = np.delete(self.vsm, del_index, 1)

        # randomize entries to get better set distribution and split them
        self.randomize_array()

        # calculate document frequency and idf using df
        df = np.count_nonzero(self.vsm > 0, axis=0) + 1
        N = len(self.vsm)
        idf = np.log10(N / df)
        self.set = np.multiply(self.vsm, idf)

        # write set data to file
        np.save(file_path+'set.npy', self.set)
        np.save(file_path+'label.npy', self.label)
        np.save(file_path+'label_id.npy', self.label_id)

    def tokenize(self, word, doc_id, stop_words):
        """ tokenize and insert term frequency in vsm """

        # remove trailing commas and apostrophes and lower word
        word = re.sub('[,\'\n]', '', word)
        word = word.lower()

        # apply lemmatization algo to each word
        word = self.lemmatize.lemmatize(word)

        # if word is not stopword, increment vsm value
        if word and word not in stop_words:
            if word not in self.features:
                self.features[word] = len(self.features)

            # resize vsm if feature count exceed its' current size
            if len(self.features) > len(self.vsm[0]):
                self.insert_extra_cols()
            self.vsm[doc_id][self.features[word]] += 1

    def delete_extra_cols(self):
        """ delete empty and unused columns from vsm """

        start = len(self.features)
        stop = len(self.vsm[0])
        self.vsm = np.delete(
            self.vsm, [i for i in range(start, stop)], axis=1)

    def insert_extra_cols(self):
        """ resize vsm if features start to overflow """

        rows = len(self.vsm)
        cols = self.vsm_init_size
        self.vsm = np.concatenate(
            (self.vsm, np.zeros(shape=(rows, cols))), axis=1)

    def calculate(self, result_path):
        """ calculate and predict the class of set queries """

        # create mean value clusters
        clusters = self.clustering()

        total = len(self.set)
        correct = 0
        self.result = []
        self.param = [0, self.filter, self.rand_seed]

        clusters_max = [0 for i in range(self.class_count)]
        clusters_total = [0 for i in range(self.class_count)]

        # for every test query, compute similarity with every cluster and predict the one with most similarity
        for query_index, query in enumerate(self.set):
            sim_list = []
            for cluster in clusters:
                sim = np.dot(cluster, query) / \
                    (np.linalg.norm(cluster) * np.linalg.norm(query))
                # sim = np.linalg.norm(cluster-query)
                sim_list.append(sim)
            min_index = sim_list.index(max(sim_list))
            computed = min_index
            expected = self.label[query_index]
            self.result.append(computed)
            clusters_total[computed] += 1
            if expected == computed:
                clusters_max[computed] += 1
        self.purity = sum([clusters_max[i]/clusters_total[i]
                           for i in range(len(clusters_total))])/self.class_count * 100
        self.param.append(self.purity)

        np.save(result_path+'result.npy', self.result)
        np.save(result_path+'param.npy', self.param)

        return self.purity

    def clustering(self):
        """ select random centroids and find mean value by clustering until optimal result reached """

        # select random centroids
        random.seed(self.rand_seed_cluster)
        clusters = np.zeros(shape=(self.class_count, self.set.shape[1]))
        rand = random.randint(0, self.set.shape[0] - 1)
        for i in range(self.class_count):
            while self.label[rand] != i:
                rand = random.randint(0, self.set.shape[0] - 1)
            clusters[i] = self.set[rand]
        old_clusters = [[] for i in range(self.class_count)]

        # find mean of cluster until centroid is unchanged
        while not np.array_equal(old_clusters, clusters):
            old_clusters = np.copy(clusters)
            clusters_doc = [[] for i in range(self.class_count)]
            for doc_id, doc in enumerate(self.set):
                sim_list = []
                for cluster in clusters:
                    sim = np.dot(cluster, doc) / \
                        (np.linalg.norm(cluster) * np.linalg.norm(doc))
                    # sim = np.linalg.norm(query-cluster)
                    sim_list.append(sim)
                min_index = sim_list.index(max(sim_list))
                clusters_doc[min_index].append(doc_id)
            for index, cluster_doc in enumerate(clusters_doc):
                clusters[index] = np.mean([self.set[i]
                                           for i in cluster_doc], axis=0)

        return clusters


if __name__ == "__main__":
    indexer = Indexer()
    indexer.read_file('bbcsport/', 'files/')
    print("purity: ", indexer.param[3])


# 0 > 1 > 29 > 97.72
