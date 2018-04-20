import numpy as np
import random
import json

def read_data(trX,trY,batch_size):
		rg_x = range(trX.shape[0])
		random.shuffle(rg_x)
		x_collection = [np.array(trX[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)] 
		y_collection = [np.array(trY[rg_x[x:x+batch_size]]) for x in range(0,len(rg_x),batch_size)]
		
		return x_collection,y_collection





def normalization(seq):
		"""
		x_max = max(seq)
		x_min = min(seq)
		epilson = 1e-6
		new_seq = [10000 * (epilson + x - x_min )/(epilson + x_max - x_min) for x in seq] 
		"""
		new_seq = [6.3578286171 * x for x in seq]
		return new_seq

def convert_onefile_to_oneline(path, ID, step_increment):	
		data = np.loadtxt(path, delimiter=',')

		idx = range(0, data.shape[1], step_increment)
		slice_idx1 = idx[0:-1] 
		slice_idx2 = idx[1:]

		x_seq = ""
		for st, ed in zip(slice_idx1, slice_idx2):
			seq = np.array([normalization(x) for x in data[:, st:ed]])
			#tmp= [ID] + [str(x) for x in seq.flatten().astype(np.float32).tolist()]
			tmp= [ID] + [str(x) for x in np.round(seq,3).flatten().tolist()]
			x_seq += ','.join(tmp) + '\n' 

		#print np.array(x_seq).shape
		return x_seq


def save_batch_to_csv(filelist_path, seq_len):
		with open(filelist_path,'r') as f1:
			filepaths = f1.readlines()

		for line in filepaths:
			out = ""
			path = line.strip()

			ID = path.split('/')[-1].strip('.csv')

			res = convert_onefile_to_oneline(path, ID, seq_len)
			out += res

			print ID
			with open("data/gsf_no_norm_data.txt", 'a') as f2:
				f2.writelines(out)



seq_len = 4096

if __name__ == "__main__":
		#filelist_path = "data/csv_data.list"
		filelist_path = "data/total.csv"
		#filelist_path = "s5k.csv"

		"""
		anno_json = json.load(open("data/anno.json",'r'))
		
		normal_json = {}
		with open("data/normal_ID.txt",'r') as f1:
			txt = f1.readlines()
		for line in txt:
				normal_json[line.strip()] = [30]

		anno_json.update(normal_json)
		"""

		save_batch_to_csv(filelist_path, seq_len)
		#save_batch_to_csv(filelist_path, normal_json, seq_len, "normal")

