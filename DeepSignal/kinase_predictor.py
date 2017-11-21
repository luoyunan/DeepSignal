import torch
from torch.autograd import Variable

class KinasePredictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab


    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        softmax_list, _, other = self.model(src_id_seq, [len(src_seq)])
        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq, softmax_list

    def predict_batch(self, src_seq_list):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        # src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
        #                       volatile=True).view(1, -1)

        src_id_seq_list = []
        for src_seq in src_seq_list:
            src_id_seq = [self.src_vocab.stoi[tok] for tok in src_seq]
            src_id_seq_list.append(src_id_seq)
        src_id_seq_list = Variable(torch.LongTensor(src_id_seq_list), volatile=True)

        if torch.cuda.is_available():
            src_id_seq_list = src_id_seq_list.cuda()

        softmax_list, _, other = self.model(src_id_seq_list, [len(src_seq) for src_seq in src_seq_list])

        # length = other['length'][0]
        # tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        # tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        tgt_id_seq_list = []
        tgt_seq_list = []
        for bi in range(len(src_seq_list)):
            tgt_id_seq = [other['sequence'][di][bi].data[0] for di in range(other['length'][bi])]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            tgt_id_seq_list.append(tgt_id_seq)
            tgt_seq_list.append(tgt_seq)
        return tgt_seq_list, softmax_list