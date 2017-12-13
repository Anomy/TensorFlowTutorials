from __future__ import print_function

import tensorflow as tf


node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2) #Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)

sess = tf.Session()
print(sess.run([node1, node2])) #[3.0, 4.0]

node3 = tf.add(node1, node2)
print("node3:", node3) #node3: Tensor("Add:0", shape=(), dtype=float32)
print("sess.run(node3):", sess.run(node3)) #sess.run(node3): 7.0

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5})) #7.5
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]})) #[ 3.  7.]
