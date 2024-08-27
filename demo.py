import tensorflow as tf
tensor=tf.constant([[1,2,3],[4,5,6]])
tensor2=tf.constant([[11,12,13],[14,15,16]])
added=tf.add(tensor,tensor2)
print(tensor)
print(tensor2)
print(added)