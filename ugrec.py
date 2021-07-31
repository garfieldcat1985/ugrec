import functools
import numpy
import tensorflow as tf
import os
from concurrent.futures.process import ProcessPoolExecutor

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.compat.v1.disable_eager_execution()
#tf.random.set_seed(1)
#numpy.random.seed(10)
from sampler import WarpSampler
from side_inf_sampler import SideInfWarpSampler
from utils import *
from para_parser import *

KS = 20
BATCH_SIZE = 0
BATCH_SIZE_TEST = 50
N_NEGATIVE = 20
EVALUATION_EVERY_N_BATCHES = 0
EMBED_DIM = 0
USER_NUM = 0
ITEM_NUM = 0
CATEGORY_NUM = 0
BRAND_NUM = 0
EPOCH_NUM = 0
ALL_item_list = []

#############################################################
def doublewrap(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.compat.v1.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


##############################################################


class UGRec(object):
    def __init__(self,
                 element_counts,
                 embed_dim=64,
                 margin_major_inf=1.0,
                 margin_side_inf_d=1.0,
                 margin_side_inf_u=1.0,
                 embedding_index=None,
                 master_learning_rate=0.01,
                 ):
        self.n_users = element_counts[0]
        self.n_items = element_counts[1]
        self.space_size = int(len(embedding_index) / 2)
        self.embedding_index = embedding_index
        self.embed_dim = embed_dim
        self.margin_major_inf = margin_major_inf
        self.margin_side_inf_d = margin_side_inf_d
        self.margin_side_inf_u = margin_side_inf_u
        self.master_learning_rate = master_learning_rate

        self.cons_resize_neg = tf.constant(1.0, tf.float32, [N_NEGATIVE, EMBED_DIM])
        self.cons_resize_test = tf.constant(1.0, tf.float32, [ITEM_NUM, EMBED_DIM])

        self.dropout_istrain = tf.compat.v1.placeholder(tf.bool, shape=())
        self.user_positive_item_pairs = tf.compat.v1.placeholder(tf.int32, [None, None, 2])
        self.negative_samples = tf.compat.v1.placeholder(tf.int32, [None, None, None])
        self.test_user_ids = tf.compat.v1.placeholder(tf.int32, [None])
        self.test_item_ids = tf.compat.v1.placeholder(tf.int32, [None, ITEM_NUM])



        self.rels = tf.Variable(tf.random.normal([self.space_size, EMBED_DIM],
                                                 stddev=1 / (EMBED_DIM ** 0.5), dtype=tf.float32))
        self.rels_project = tf.Variable(tf.random.normal([self.space_size, EMBED_DIM],
                                                   stddev=1 / (EMBED_DIM ** 0.5), dtype=tf.float32))

        self.regu_losses = []
        self.vars = []
        self.element_embeddings = []
        self.element_embeddings_project = []

        self.element_embeddings.append(self.deploy_embeddings('user', self.n_users))
        self.element_embeddings.append(self.deploy_embeddings('item', self.n_items))

        for i in range(len(element_counts) - 2):
            self.element_embeddings.append(self.deploy_embeddings('sideinf_' + str(i), element_counts[i + 2]))
        self.element_embeddings_project.append(self.deploy_embeddings('user_proj_', self.n_users))
        self.element_embeddings_project.append(self.deploy_embeddings('item_user_proj_', self.n_items))
        for i in range(len(element_counts) - 2):
            self.element_embeddings_project.append(self.deploy_embeddings('sideinf_user_proj_' + str(i), element_counts[i + 2]))

        for i in range(len(element_counts)):
            self.vars.append(self.element_embeddings[i])
        for i in range(len(element_counts)):
            self.vars.append(self.element_embeddings_project[i])

        self.vars.append(self.rels)
        self.vars.append(self.rels_project)

        self.loss = self.deploy_loss()
        self.test_process = self.deploy_test_process()
        self.optimize

    def deploy_embeddings(self, name, item_count):
        return tf.Variable(tf.random.normal([item_count, self.embed_dim],
                                            stddev=1 / (self.embed_dim ** 0.5), name=name, dtype=tf.float32))

    def atten_MLP_deploy(self, space_index, unit0, unit1, dropoutRate=0.1):
        try:
            self.atten_x1
        except:
            self.atten_x1 = {}
            self.atten_x3 = {}
            self.atten_x4 = {}
        x1 = tf.keras.layers.Dense(unit1, activation='relu', input_shape=[None, unit0],
                                   kernel_regularizer=tf.keras.regularizers.l2(1),
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,
                                                                                         stddev=numpy.sqrt(
                                                                                             2.0 / (
                                                                                                     unit0 + unit1))))
        x3 = tf.keras.layers.Dropout(dropoutRate)
        x4 = tf.keras.layers.Softmax()
        x1.build(input_shape=[None, EMBED_DIM * 2])
        x3.build(input_shape=[None, EMBED_DIM])
        x4.build(input_shape=[None, EMBED_DIM])
        self.vars.extend(x1.trainable_variables)
        self.regu_losses.append(x1.losses)
        self.atten_x1[space_index] = x1
        self.atten_x3[space_index] = x3
        self.atten_x4[space_index] = x4

    def atten_MLP(self, space_index, x):
        re = self.atten_x1[space_index](x)
        re = self.atten_x3[space_index](re, training=self.dropout_istrain)
        re = self.atten_x4[space_index](re)
        return re

    def trans_space(self, x, x_trans, r_index=0, dim=2):
        return tf.nn.l2_normalize(x + tf.reduce_sum(x * x_trans, dim - 1, keepdims=True) * self.rels_project[r_index],
                                  dim - 1)

    def deyloy_embedding_loss_direct(self, space_index):
        left_elements = tf.nn.embedding_lookup(params=self.element_embeddings[self.embedding_index[space_index * 2]],
                                               ids=self.user_positive_item_pairs[space_index][:, 0])
        left_elements_project = tf.nn.embedding_lookup(
            params=self.element_embeddings_project[self.embedding_index[space_index * 2]],
            ids=self.user_positive_item_pairs[space_index][:, 0])

        right_elements = tf.nn.embedding_lookup(
            params=self.element_embeddings[self.embedding_index[space_index * 2 + 1]],
            ids=self.user_positive_item_pairs[space_index][:, 1])
        right_elements_project = tf.nn.embedding_lookup(
            params=self.element_embeddings_project[self.embedding_index[space_index * 2 + 1]],
            ids=self.user_positive_item_pairs[space_index][:, 1])

        left_elements_r1 = self.trans_space(left_elements, left_elements_project, space_index)

        pos_items_r1 = self.trans_space(right_elements, right_elements_project, space_index)

        att_v = tf.concat([left_elements_r1, pos_items_r1], axis=1)
        att_v = self.atten_MLP(space_index, att_v)
        left_elements_r1_plus_r1 = left_elements_r1 + self.rels[space_index] * att_v


        pos_distances = tf.reduce_sum(input_tensor=tf.math.squared_difference(left_elements_r1_plus_r1, pos_items_r1),
                                      axis=1)

        neg_items0 = tf.nn.embedding_lookup(params=self.element_embeddings[self.embedding_index[space_index * 2 + 1]],
                                            ids=self.negative_samples[space_index])
        neg_items0_trans = tf.nn.embedding_lookup(
            params=self.element_embeddings_project[self.embedding_index[space_index * 2 + 1]],
            ids=self.negative_samples[space_index])

        neg_items1 = self.trans_space(neg_items0, neg_items0_trans, space_index, 3)

        left_elements_r1_0 = tf.expand_dims(left_elements_r1, axis=1)
        left_elements_r1_0 = left_elements_r1_0 * self.cons_resize_neg
        left_right_0 = tf.concat([left_elements_r1_0, neg_items1], axis=2)
        left_right_0 = tf.reshape(left_right_0, shape=[-1, EMBED_DIM * 2])
        att_v2 = self.atten_MLP(space_index, left_right_0)
        att_v2 = tf.reshape(att_v2, [-1, N_NEGATIVE, EMBED_DIM])

        neg_items_r1 = tf.transpose(a=neg_items1,
                                    perm=(0, 2, 1))

        left_elements_r1_plus_r2 = left_elements_r1_0 + self.rels[space_index] * att_v2
        left_elements_r1_plus_r2 = tf.transpose(a=left_elements_r1_plus_r2,
                                                perm=(0, 2, 1))


        distance_to_neg_items = tf.reduce_sum(
            input_tensor=tf.math.squared_difference(left_elements_r1_plus_r2, neg_items_r1), axis=1)
        closest_negative_item_distances = tf.reduce_min(input_tensor=distance_to_neg_items, axis=1)

        margin = None
        if (space_index == 0):
            margin = self.margin_major_inf
        else:
            margin = self.margin_side_inf_d
        cal1 = pos_distances - closest_negative_item_distances + margin
        loss_per_pair = tf.maximum(cal1, 0)
        loss = tf.reduce_sum(input_tensor=loss_per_pair)
        return loss

    #############################
    def hyperplane_trans(self, e, n, dim=2):
        v1 = tf.norm(n, 2, axis=dim - 1, keepdims=True)
        v1 = tf.square(v1)
        v2 = tf.reduce_sum(e * n, axis=dim - 1, keepdims=True) * n
        v3 = tf.divide(v2, v1)
        return tf.nn.l2_normalize(e - v3, axis=dim - 1)

    def deyloy_embedding_loss_undirect(self, space_index):
        left_elements = tf.nn.embedding_lookup(params=self.element_embeddings[self.embedding_index[space_index * 2]],
                                               ids=self.user_positive_item_pairs[space_index][:, 0])
        right_elements = tf.nn.embedding_lookup(
            params=self.element_embeddings[self.embedding_index[space_index * 2 + 1]],
            ids=self.user_positive_item_pairs[space_index][:, 1])

        left_elements_r1 = left_elements
        left_elements_r1 = tf.nn.l2_normalize(left_elements_r1, axis=1)

        pos_items_r1 = right_elements
        pos_items_r1 = tf.nn.l2_normalize(pos_items_r1, axis=1)

        att_v = tf.concat([left_elements_r1, pos_items_r1], axis=1)
        att_v = self.atten_MLP(space_index, att_v)

        r_2 = self.rels[space_index] * att_v

        left_elements_r1_plus_r1 = self.hyperplane_trans(left_elements_r1, r_2)
        pos_items_r1 = self.hyperplane_trans(pos_items_r1, r_2)
        pos_distances = tf.reduce_sum(input_tensor=tf.math.squared_difference(left_elements_r1_plus_r1, pos_items_r1),
                                      axis=1)
        ###########################################################
        neg_items0 = tf.nn.embedding_lookup(params=self.element_embeddings[self.embedding_index[space_index * 2 + 1]],
                                            ids=self.negative_samples[space_index])
        neg_items1 = neg_items0
        neg_items1 = tf.nn.l2_normalize(neg_items1, axis=2)

        left_elements_r1_0 = tf.expand_dims(left_elements_r1, axis=1)
        left_elements_r1_0 = left_elements_r1_0 * self.cons_resize_neg
        left_right_0 = tf.concat([left_elements_r1_0, neg_items1], axis=2)
        left_right_0 = tf.reshape(left_right_0, shape=[-1, EMBED_DIM * 2])
        att_v2 = self.atten_MLP(space_index, left_right_0)
        att_v2 = tf.reshape(att_v2, [-1, N_NEGATIVE, EMBED_DIM])
        r_3 = self.rels[space_index] * att_v2

        neg_items_r1 = self.hyperplane_trans(neg_items1, r_3, 3)
        neg_items_r1 = tf.transpose(a=neg_items_r1,
                                    perm=(0, 2, 1))

        left_elements_r1_plus_r2 = self.hyperplane_trans(left_elements_r1_0, r_3, 3)
        left_elements_r1_plus_r2 = tf.transpose(a=left_elements_r1_plus_r2,
                                                perm=(0, 2, 1))

        distance_to_neg_items = tf.reduce_sum(
            input_tensor=tf.math.squared_difference(left_elements_r1_plus_r2, neg_items_r1), axis=1)

        closest_negative_item_distances = tf.reduce_min(input_tensor=distance_to_neg_items, axis=1)

        margin = self.margin_side_inf_u
        cal1 = pos_distances - closest_negative_item_distances + margin
        loss_per_pair = tf.maximum(cal1, 0)

        loss = tf.reduce_sum(input_tensor=loss_per_pair)
        return loss

    def deploy_loss(self):
        for i in range(self.space_size):
            self.atten_MLP_deploy(i, EMBED_DIM * 2, EMBED_DIM)
        loss_total = self.deyloy_embedding_loss_direct(0)
        self.loss_major = loss_total
        for i in range(1, self.space_size):
            if self.embedding_index[i * 2] != self.embedding_index[i * 2 + 1]:
                loss_total = loss_total + self.deyloy_embedding_loss_direct(i)
            else:
                loss_total = loss_total + self.deyloy_embedding_loss_undirect(i)
        for item in self.regu_losses:
            loss_total = loss_total + item
        return loss_total

    @define_scope
    def clip_by_norm_op(self):
        clip_obj = []
        entity_count = len(self.element_embeddings)
        for i in range(entity_count):
            clip_obj.append(tf.compat.v1.assign(self.element_embeddings[i],
                                                tf.clip_by_norm(self.element_embeddings[i], 1, axes=[1])))
        for i in range(entity_count):
            clip_obj.append(tf.compat.v1.assign(self.element_embeddings_project[i],
                                                tf.clip_by_norm(self.element_embeddings_project[i], 1,
                                                                axes=[1])))
        return clip_obj

    @define_scope
    def optimize(self):
        gds = []
        gds.append(
            tf.keras.optimizers.Adagrad(learning_rate=self.master_learning_rate).get_updates(self.loss, self.vars)[
                0])
        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    def deploy_test_process(self):
        user = tf.expand_dims(tf.nn.embedding_lookup(params=self.element_embeddings[0], ids=self.test_user_ids), 1)
        item = tf.nn.embedding_lookup(params=self.element_embeddings[1], ids=self.test_item_ids)

        user_0 = tf.expand_dims(tf.nn.embedding_lookup(params=self.element_embeddings_project[0], ids=self.test_user_ids), 1)
        item_0 = tf.nn.embedding_lookup(params=self.element_embeddings_project[1], ids=self.test_item_ids)
        user_r1 = self.trans_space(user, user_0, 0, 3)
        item_r1 = self.trans_space(item, item_0, 0, 3)
        user_r1 = user_r1 * self.cons_resize_test
        att_v = tf.concat([user_r1, item_r1], axis=2)
        att_v = tf.reshape(att_v, [-1, EMBED_DIM * 2])
        att_v = self.atten_MLP(0, att_v)
        att_v = tf.reshape(att_v, [-1, ITEM_NUM, EMBED_DIM])
        user_r1_plus_r1 = user_r1 + self.rels[0] * att_v
        result = tf.reduce_sum(input_tensor=tf.math.squared_difference(user_r1_plus_r1, item_r1), axis=2)
        return result



def calc_ndcg(rank, ground_truth):
    len_rank = len(rank)
    len_gt = len(ground_truth)
    idcg_len = min(len_gt, len_rank)
    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len - 1]
    dcg = np.cumsum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg / idcg
    return result[len(result) - 1]

def eval_sub(para):
    score = para[0]
    positive_simples = para[1]
    ks = para[2]
    index0 = numpy.argsort(score)
    index0 = index0[0:ks]
    s = 0
    for element in positive_simples:
        if element in index0:
            s += 1
    hr = 0
    if s > 0:
        hr = 1
    ndcg = calc_ndcg(index0, positive_simples)
    return hr, ndcg


def eval_for_test(model, sess, test_user_simples, item_already_in_train, test_item_positive_simples):
    score = []
    test_bench_size = int(np.ceil(len(test_user_simples) / BATCH_SIZE_TEST))
    executor = ProcessPoolExecutor()
    for i in range(test_bench_size):
        user_list = test_user_simples[i * BATCH_SIZE_TEST:(i + 1) * BATCH_SIZE_TEST]
        _item_already_in_train = item_already_in_train[i * BATCH_SIZE_TEST:(i + 1) * BATCH_SIZE_TEST]

        score_tmp = sess.run(model.test_process,
                          {model.test_user_ids: user_list,
                           model.test_item_ids: ALL_item_list[:len(user_list)],
                           model.dropout_istrain: False
                           })

        for index, element in enumerate(_item_already_in_train):
            for e2 in element:
                score_tmp[index][e2] = np.inf
        score.extend(score_tmp.copy())

    paras = []
    for i in range(len(score)):
        tmp = []
        tmp.append(score[i])
        tmp.append(test_item_positive_simples[i])
        tmp.append(KS)
        paras.append(tmp)
    part_data = executor.map(eval_sub, paras)
    hr_data = []
    ndcg_data = []
    for result in part_data:
        hr_data.append(result[0])
        ndcg_data.append(result[1])

    hr = np.sum(hr_data) / len(score)
    ndcg = np.sum(ndcg_data) / len(score)
    return hr, ndcg


def optimize(model, samplers, train, test):
    item_already_in_train_dic = {}
    test_item_positive_simples_dic = {}
    item_already_in_train = []
    test_user_simples = []
    test_item_positive_simples = []

    for item in train.keys():
        if item[0] not in item_already_in_train_dic.keys():
            item_already_in_train_dic[item[0]] = []
        item_already_in_train_dic[item[0]].append(item[1])

    for item in test.keys():
        if item[0] not in test_item_positive_simples_dic.keys():
            test_item_positive_simples_dic[item[0]] = []
        test_item_positive_simples_dic[item[0]].append(item[1])

    for key in test_item_positive_simples_dic.keys():
        test_user_simples.append(key)
        test_item_positive_simples.append(test_item_positive_simples_dic[key])
        item_already_in_train.append(
            list(set(item_already_in_train_dic[key]) - set(test_item_positive_simples_dic[key])))

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    count = 0
    while True:
        print('Epoch {}:'.format(count))
        losses = []
        for nowPatch in range(EVALUATION_EVERY_N_BATCHES):
            left_pos = []
            right_neg = []
            user_pos, neg = samplers[0].next_batch()
            left_pos.append(user_pos)
            right_neg.append(neg)
            for index in range(len(samplers) - 1):
                pos_tmp, neg_tmp = samplers[index + 1].next_batch()
                left_pos.append(pos_tmp)
                right_neg.append(neg_tmp)
            _, loss = sess.run((model.optimize, model.loss_major),
                               {model.user_positive_item_pairs: left_pos,
                                model.negative_samples: right_neg,
                                model.dropout_istrain: True})
            losses.append(loss)
        print("Training loss in major user/item interaction space: {}".format(numpy.mean(losses)))
        count += 1
        if count > 0 and count % 10 == 0:
            result = eval_for_test(model, sess, test_user_simples, item_already_in_train, test_item_positive_simples)
            print("HR: {}  NDCG: {}".format(result[0], result[1]))
        if count > EPOCH_NUM:
            exit()

if __name__ == '__main__':
    args = parse_args()
    EMBED_DIM = args.embed_size
    EPOCH_NUM = args.epoch

    user_item_matrix, user_item_matrix_test = load_data_matrix(args.data_path + args.dataset + '/train_dataset.txt',
                                                               args.data_path + args.dataset + '/test_dataset.txt')
    USER_NUM, ITEM_NUM = user_item_matrix.shape
    BATCH_SIZE = args.batch_size
    KS = args.Ks
    TEST_SIMPLE_NUM = len(user_item_matrix_test)
    EVALUATION_EVERY_N_BATCHES = int(len(user_item_matrix) / BATCH_SIZE)
    ALL_item_list = [x for x in range(ITEM_NUM)]
    ALL_item_list = np.tile(ALL_item_list, (BATCH_SIZE_TEST, 1))

    side_information_matrixs = []
    side_information_matrixs.append(load_data_direct(args.data_path + args.dataset + '/item_category_triple.txt'))
    side_information_matrixs.append(load_data_direct(args.data_path + args.dataset + '/item_brand_triple.txt'))
    CATEGORY_NUM = side_information_matrixs[0].shape[1]
    BRAND_NUM = side_information_matrixs[1].shape[1]
    side_information_matrixs.append(
        load_data_undirect(ITEM_NUM, args.data_path + args.dataset + '/item_also_buy_triple.txt'))
    side_information_matrixs.append(
        load_data_undirect(ITEM_NUM, args.data_path + args.dataset + '/item_also_view_triple.txt'))

    element_counts = []
    element_counts.append(USER_NUM)
    element_counts.append(ITEM_NUM)
    element_counts.append(CATEGORY_NUM)
    element_counts.append(BRAND_NUM)

    train, test = split_data(user_item_matrix, user_item_matrix_test)

    sampler = WarpSampler(train, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE, check_negative=True)
    info_simplers = []
    info_simplers.append(sampler)

    for i in range(len(side_information_matrixs)):
        info_simplers.append(
            SideInfWarpSampler(side_information_matrixs[i], batch_size=BATCH_SIZE, n_negative=N_NEGATIVE,
                               check_negative=True))
    margins=eval(args.margins)
    model = UGRec(element_counts,
                  embed_dim=EMBED_DIM,
                  embedding_index=[0, 1, 1, 2, 1, 3, 1, 1, 1, 1],
                  margin_major_inf=margins[0],
                  margin_side_inf_d=margins[1],
                  margin_side_inf_u=margins[2],
                  master_learning_rate=args.lr
                  )
    optimize(model, info_simplers, train, test)