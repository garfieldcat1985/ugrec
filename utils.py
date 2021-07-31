from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix, lil_matrix

np.random.seed(10)

def get_test_negative(negative_file):
    test_negtavie = defaultdict(set)
    with open(negative_file, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            eee = arr[0]
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            test_negtavie[eee] = negatives
            line = f.readline()
    return test_negtavie


def load_data_direct(file):
    item_list = []
    category_list = []
    item_count = -1
    category_count = -1
    for line in open(file).readlines():
        line = line.replace("\n", "")
        arr = line.split("\t")
        item_list.append(int(arr[0]))
        category_list.append(int(arr[1]))
        if int(arr[0]) + 1 > item_count:
            item_count = int(arr[0]) + 1
        if int(arr[1]) + 1 > category_count:
            category_count = int(arr[1]) + 1

    relation_matrix = dok_matrix((item_count, category_count), dtype=np.int32)
    for index in range(len(item_list)):
        relation_matrix[item_list[index], category_list[index]] = 1
    return relation_matrix


def load_data_undirect(ITEM_NUM, file):
    item_list1 = []
    item_list2 = []
    item1_count = -1
    item2_count = -1
    for line in open(file).readlines():
        line = line.replace("\n", "")
        arr = line.split("\t")
        item_list1.append(int(arr[0]))
        item_list2.append(int(arr[1]))
        if int(arr[0]) + 1 > item1_count:
            item1_count = int(arr[0]) + 1
        if int(arr[1]) + 1 > item2_count:
            item2_count = int(arr[1]) + 1
    item_also_view_matrix = dok_matrix((ITEM_NUM, ITEM_NUM), dtype=np.int32)
    for index in range(len(item_list1)):
        item_also_view_matrix[item_list1[index], item_list2[index]] = 1
        item_also_view_matrix[item_list2[index], item_list1[index]] = 1
    return item_also_view_matrix


def load_data_matrix(training_simple_file, test_simple_file):
    user_dict = defaultdict(set)
    user_rate_dict = defaultdict(set)
    for line in open(training_simple_file).readlines():
        line = line.replace("\n", "")
        arr = line.split(" ")
        for e in arr[1:]:
            user_dict[int(arr[0])].add(int(e))

    n_users = len(user_dict)
    n_items = max([item for items in user_dict.values() for item in items]) + 1
    user_item_matrix = dok_matrix((n_users, n_items), dtype=np.int32)
    for e in user_dict.keys():
        items = user_dict[e]
        for item in items:
            user_item_matrix[e, int(item)] = 1
    ####################
    user_dict_1 = defaultdict(set)
    for line in open(test_simple_file).readlines():
        line = line.replace("\n", "")
        arr = line.split(" ")
        for e in arr[1:]:
            user_dict_1[int(arr[0])].add(int(e))
    user_item_matrix_test = dok_matrix((n_users, n_items), dtype=np.int32)
    for e in user_dict_1.keys():
        items = user_dict_1[e]
        for item in items:
            user_item_matrix_test[e, int(item)] = 1
    return user_item_matrix, user_item_matrix_test


def split_data(user_item_matrix, user_item_matrix_test):
    train = dok_matrix(user_item_matrix.shape)
    test = dok_matrix(user_item_matrix.shape)

    user_item_matrix = lil_matrix(user_item_matrix)
    for user in range(user_item_matrix.shape[0]):
        items = list(user_item_matrix.rows[user])
        np.random.shuffle(items)
        for i in items:
            train[user, i] = 1
    user_item_matrix_validation = lil_matrix(user_item_matrix_test)
    for user in range(user_item_matrix_validation.shape[0]):
        items = list(user_item_matrix_validation.rows[user])
        np.random.shuffle(items)
        for i in items:
            test[user, i] = 1

    print("{}/{} train/test samples".format(
        len(train.nonzero()[0]),
        len(test.nonzero()[0])))
    return train, test


def split_data_side_inf(side_information_matrix):
    train = dok_matrix(side_information_matrix.shape)
    side_information_matrix = lil_matrix(side_information_matrix)
    for item in range(side_information_matrix.shape[0]):
        items = list(side_information_matrix.rows[item])
        np.random.shuffle(items)
        for i in items:
            train[item, i] = 1
    return train
