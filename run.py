import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
sys.path.append('DeepSignal')
from kinase_model import EncoderKinase, DecoderKinase, KinaseModel
from kinase_predictor import KinasePredictor
import seq2seq
from seq2seq.util.checkpoint import Checkpoint


def save_result(prob, output_vocab):
	prob = np.exp(prob)
	substrate_len = prob.shape[0]
	aa_pos = np.arange(substrate_len) - int(substrate_len / 2)
	df = pd.DataFrame(prob.T, index=output_vocab.itos, columns=aa_pos)
	aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
		  'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_']
	df = df.reindex(aa)
	df.to_csv('output/result.tsv', sep='\t', float_format='%.6f')

def prediction(predictor, output_vocab):
	record = SeqIO.read('data/input.fasta', 'fasta')
	pred_seq, pred_prob = predictor.predict(record.seq)
	substrate_len = len(pred_prob)
	num_aa = pred_prob[0].size(1)

	prob = np.zeros((substrate_len, num_aa))
	for ix_pos in range(substrate_len):
		for ix_aa in range(num_aa):
			prob[ix_pos, ix_aa] = pred_prob[ix_pos][0, ix_aa].data[0]
	save_result(prob, output_vocab)

def test():
	checkpoint_path = 'model'
	checkpoint = Checkpoint.load(checkpoint_path)
	model = checkpoint.model
	input_vocab = checkpoint.input_vocab
	output_vocab = checkpoint.output_vocab
	predictor = KinasePredictor(model, input_vocab, output_vocab)
	prediction(predictor, output_vocab)

def main():
	test()

if __name__ == '__main__':
	main()