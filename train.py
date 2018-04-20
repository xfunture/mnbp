
from sys import argv
from model import *
from read_data import *
from tensorflow.python.framework import graph_util
from tfrc import read_and_decode 

batch_size = 256
dropout = 0.2



train_path = "data/train.tfrecords"
test_path = "data/test.tfrecords"

def get_train_test_data(train_path, test_path, batch_size):
		ecgdata, label = read_and_decode(train_path)
		test_data_batch, test_label_batch = read_and_decode(test_path)


		train_data, train_label = tf.train.shuffle_batch([ecgdata, label],
													batch_size=batch_size, capacity=50000,
													num_threads=3,
													min_after_dequeue=1000)

		test_data, test_label = tf.train.shuffle_batch([test_data_batch, test_label_batch],
													batch_size=256, capacity=50000,
													num_threads=3,
													min_after_dequeue=1000)

		return train_data, train_label, test_data, test_label 

train_data, train_label, test_data, test_label = get_train_test_data(train_path, test_path, batch_size)

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
version = tf.constant('v1.0.0', name='version')

with tf.device('/gpu:1'):
	model = build_network()


saver=tf.train.Saver()
config.gpu_options.per_process_gpu_memory_fraction = 0.4


read_log = True
prefix = 'mnbp_v1'
epochs = 500000
predict = True

if predict:
		tx = np.loadtxt("data/test_x0.csv", delimiter=',', usecols=range(1,22))
		ph = np.array([[125.3645,77.016500,1.606788,1.372469,2.768998] for x in range(256)])

init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()
with tf.Session(config=config) as sess:
	sess.run(init_global)
	#sess.run(init_local)

	#tf.summary.FileWriter('log', sess.graph)

	if read_log:			
		with open("log/checkpoint",'r') as f1:
				txt = f1.readline()
				point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
				print point
				saver.restore(sess,"log/%s"%point)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	print 'training start...'
	for step in range(epochs):
		batch_x, batch_y = sess.run([train_data, train_label])
		sess.run(model.optimizer, feed_dict={model.x: batch_x, model.labels: batch_y, model.dropout: dropout})


		if step % 10 == 0:	
			loss = sess.run(model.acc, feed_dict={model.x: batch_x, model.labels: batch_y, model.dropout: dropout})

			test_x, test_y = sess.run([test_data, test_label])
			mid, logits, val_loss, val_acc = sess.run([model.mid, model.logits, model.loss,model.acc], feed_dict={model.x: test_x, model.labels: test_y, model.dropout: 0})
			rand_acc = np.mean(np.square(np.log1p(ph) - np.log1p(test_y)))
			
			print "Epoch %d/%d - loss: %f \tval_loss: %f\tval_acc: %f\trand_acc: %f"  % (step+1,epochs, loss, val_loss, val_acc, rand_acc) 

			

		if step % 2000 == 0:	
			print logits[10:30]
			print '\n'
			print mid[10:30]
			print '\n'
			print np.mean(logits,0)
		if step % 500 == 199:	
			checkpoint_filepath='log/step-%d.ckpt' % step
			saver.save(sess,checkpoint_filepath)
			print '\n~~~~checkpoint saved!~~~~~\n'

		if step % 1000 == 1 and step > epochs - 2000 and False:	
			output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
											output_node_names=["x", "y", 'dropout', 'version'])
			with tf.gfile.FastGFile('./load_pb/%s_%d.pb' %(prefix,step), mode='wb') as f:
				f.write(output_graph_def.SerializeToString())

		if val_acc<0.042:
			pred_out = sess.run(model.logits, feed_dict={model.x: tx, model.dropout: 0})
			np.savetxt("data/submit_test_%d.csv" % val_acc, pred_out,  delimiter=',', fmt='%f')
			

	coord.request_stop()
	coord.join(threads)



#json.dump(val_acc_log, open("training_log.json",'w'))




