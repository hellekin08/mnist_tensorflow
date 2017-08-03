import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

#hl1,hl2,....
all_nodes = [500,500,500]
n_classes = 10
start_val = 784
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):
	layer_out = None
	for i in range(len(all_nodes)):
		if i == 0:			
			layer_out = create_layer(start_val,all_nodes[i],data)
		else:
			layer_out = create_layer(all_nodes[i-1],all_nodes[i],layer_out)

	return tf.matmul(layer_out,tf.Variable(tf.random_normal([all_nodes[-1],n_classes]))) \
		   + tf.Variable(tf.random_normal([ n_classes ]) )  

def create_layer(init_nodes,n_nodes_this_layer,data):
	weight = tf.Variable(tf.random_normal([init_nodes,n_nodes_this_layer]))
	biases = tf.Variable(tf.random_normal([ n_nodes_this_layer ]) )
	layer = tf.add(tf.matmul(data,weight) , biases)			
	layer = tf.nn.relu(layer)	
	return layer

#trainieren des models input data = x
def train_neural_network(x):
	#Die schötzung ist das Ergebnis unseres models als Array z.b. (1,0,0,0,0,0,0,0,0,0)
	prediction = neural_network_model(x)
	# cost = lost softmax_cross_entropy_with_logits ist die cost-function
	# berechnet die differenz zwischen prediction und einem bekannten label!!!!
	# one_hot definiert dabei den output
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels=y) )

	#minimieren des cost mit optimizier
	#AdamOptimizer nutzt gradient descent ??
	#optional learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#Anzahl der Epochen festlegen
	#durchläuft 10mal feedforward + backprop
	hm_epochs = 10

	#starten des Session
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		# durchlaufen aller epochen
		for epoch in range(hm_epochs):
			epoch_loss = 0
			# total number of samples / batch_sizes sagt uns wie of wir 
			# durch eine epoche iterieren müssen
			for _ in range(int(mnist.train.num_examples/batch_size)):
				#lädt die daten des mnist-datensatzes als 100ter Stapel
				#die funktion muss für eigene daten selber geschrieben werden!
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				#Starten des Session mit dem Optimizer und der cost
				#feed_dict ist die übergabe der Daten 
				#nun weiß tf das es die weights und biases anpassen muss 
				_, c = sess.run([optimizer,cost],feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print("Epoch",epoch,"completed out of",hm_epochs,"loss:",epoch_loss)

		#argmax???????? returns index of maximum value in den Arrays!! prediction und y
		#überprüpft ob der wert des maximalen index gleich ist!
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		#Changes the variable to float
		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		#überprüfen aller Accurancy der bilder zu den labels
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
