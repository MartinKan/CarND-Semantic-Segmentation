#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Loading the model and weights from a pretrained VGG Model into Tensorflow
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return vgg_input_image, vgg_keep_prob, vgg_layer3, vgg_layer4, vgg_layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    # CHANGE CODE BELOW

    # The code below implements the FCN-8 model.  This model involves an encoder portion that 
    # downsamples the input image to smaller dimensions so that it will be computationally efficient 
    # to process.  This portion is represented by the pretrained VGG model that we imported. Next, 
    # a decoder portion is implemented that upsamples the output of the encoder and restores
    # the processed image to the original image size (so that segmantation on the original image can be done).  
    # The shape of the tensor after the final convolutional transpose layer will be 4-dimensional: 
    # (batch_size, original_height, original_width, num_classes).

    # Added to avoid ResourceExhaustedError
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)    

    # Perform 1x1 convolution on the relevant vgg layers.  The height and width of the layers will stay the same 
    # after the convolution but the number of filters on the layers will all be reduced to 1 - thus ensuring that
    # they will have the same shape when performing the skip connections later on
    vgg_layer7_out = tf.layers.conv2d(
        vgg_layer7_out, num_classes, kernel_size=1, name='vgg_layer7_out')
    vgg_layer4_out = tf.layers.conv2d(
        vgg_layer4_out, num_classes, kernel_size=1, name='vgg_layer4_out')
    vgg_layer3_out = tf.layers.conv2d(
        vgg_layer3_out, num_classes, kernel_size=1, name='vgg_layer3_out')

    # Begin upsampling the layers using transposed convolutions as instructed in the class tutorial.
    # The first layer to upsample is the final layer of the VGG model
    fcn8_decoder_layer1 = tf.layers.conv2d_transpose(
        vgg_layer7_out, num_classes, kernel_size=4, strides=(2, 2),
        padding='same', name='fcn8_decoder_layer1')

    # After layer 7 has been upsampled, it will be the same size as vgg_layer4_out.  Next, we will 
    # add the first skip connection from vgg_layer4_out
    fcn8_decoder_layer2 = tf.add(
        fcn8_decoder_layer1, vgg_layer4_out, name='fcn8_decoder_layer2')

    # Perform another upsampling step using transposed convolution to create a new layer that has
    # the same shape as vgg_layer3_out
    fcn8_decoder_layer3 = tf.layers.conv2d_transpose(
        fcn8_decoder_layer2, num_classes, kernel_size=4, strides=(2, 2),
        padding='same', name='fcn8_decoder_layer3')

    # Next, we will add the second skip connection from vgg_layer3_out
    fcn8_decoder_layer4 = tf.add(
        fcn8_decoder_layer3, vgg_layer3_out, name='fcn8_decoder_layer4')

    # Finally, we will upsample fcn8_decoder_layer4 to make it the same shape as the input image 
    fcn8_decoder_output = tf.layers.conv2d_transpose(
        fcn8_decoder_layer4, num_classes, kernel_size=16, strides=(8, 8),
        padding='same', name='fcn8_decoder_layer4')

    return fcn8_decoder_output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    # the code below follows closely the instructions that were given in the class tutorials
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    loss_operation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    return logits, training_operation, loss_operation

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    #pass

    sess.run(tf.global_variables_initializer())

    # Trains the network using batches of training data provided by a helper function. The loss is calculated 
    # and displayed at each training step
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.001})
            print("Loss = {:.3f}".format(loss))

tests.test_train_nn(train_nn)

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)
    EPOCHS = 25
    BATCH_SIZE = 10

    learning_rate = tf.placeholder(tf.float32, (None), name='learning_rate')
    correct_label = tf.placeholder(tf.int32, (None, None, None, num_classes), name='correct_label')
    
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, training_operation, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, training_operation, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
