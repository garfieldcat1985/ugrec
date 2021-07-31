import numpy
from multiprocessing import Process, Queue
from scipy.sparse import lil_matrix
numpy.random.seed(10)

def sample_function(item_sideinf_matrix, batch_size, n_negative, result_queue, check_negative=True):

    item_sideinf_matrix = lil_matrix(item_sideinf_matrix)
    item_sideinf_pairs = numpy.asarray(item_sideinf_matrix.nonzero()).T
    item_to_positive_set = {item1: set(row) for item1, row in enumerate(item_sideinf_matrix.rows)}
    while True:
        numpy.random.shuffle(item_sideinf_pairs)
        for i in range(int(len(item_sideinf_pairs) / batch_size)):

            user_positive_items_pairs = item_sideinf_pairs[i * batch_size: (i + 1) * batch_size, :]

            # sample negative samples
            negative_samples = numpy.random.randint(
                0,
                item_sideinf_matrix.shape[1],
                size=(batch_size, n_negative))

            # Check if we sample any positive items as negative samples.
            # Note: this step can be optional as the chance that we sample a positive item is fairly low given a
            # large item set.
            if check_negative:
                for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                       negative_samples,
                                                       range(len(negative_samples))):
                    user = user_positive[0]
                    for j, neg in enumerate(negatives):
                        while neg in item_to_positive_set[user]:
                            negative_samples[i, j] = neg = numpy.random.randint(0, item_sideinf_matrix.shape[1])
            result_queue.put((user_positive_items_pairs, negative_samples))


class SideInfWarpSampler(object):
    def __init__(self, item_sideinf_matrix, batch_size=10000, n_negative=1, n_workers=1, check_negative=True):
        self.result_queue = Queue(maxsize=n_workers * 2)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(item_sideinf_matrix,
                                                       batch_size,
                                                       n_negative,
                                                       self.result_queue,
                                                       check_negative)))
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:  # type: Process
            p.terminate()
            p.join()
