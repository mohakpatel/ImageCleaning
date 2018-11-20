""" 
Model for microscope images cleaning. 
"""
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 
import tensorflow as tf
import os
import sklearn
import numpy as np

class ImageCleaner(object):
    def __init__(self):
        
        self.lr = 0.001
               
        self.build()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build(self):
        
        # Input placeholders
        self.input_img = tf.placeholder(tf.float32, (None,64,64,1), name="input")
        self.target_img = tf.placeholder(tf.float32, (None,64,64,1), name="input")

        # Layer 1: Conv to 32x32x15
        conv1 = tf.layers.conv2d(inputs=self.input_img, 
                             filters=15, 
                             kernel_size=(5,5),
                             strides=(2,2),
                             padding='same',
                             activation=tf.nn.relu)
        
        # Layer 2: Conv to 16x16x30
        conv2 = tf.layers.conv2d(inputs=conv1, 
                             filters=30, 
                             kernel_size=(5,5),
                             strides=(2,2),
                             padding='same',
                             activation=tf.nn.relu)
        
        # Layer 3: Conv to 8x8x45
        conv3 = tf.layers.conv2d(inputs=conv2, 
                             filters=45, 
                             kernel_size=(5,5),
                             strides=(2,2),
                             padding='same',
                             activation=tf.nn.relu)
        
        # Layer 4: Conv_transpose to 16x16x45
        conv_t4 = tf.layers.conv2d_transpose(inputs=conv3, 
                                            filters=45, 
                                            kernel_size=(3,3),
                                            strides=(2,2),
                                            padding='same',
                                            activation=tf.nn.relu)
        
        # Layer 5: Conv_transpose to 32x32x30
        conv_t5 = tf.layers.conv2d_transpose(inputs=conv_t4, 
                                            filters=30, 
                                            kernel_size=(5,5),
                                            strides=(2,2),
                                            padding='same',
                                            activation=tf.nn.relu)
        
        # Layer 6: Conv_transpose to 64x64x15
        conv_t6 = tf.layers.conv2d_transpose(inputs=conv_t5, 
                                            filters=15, 
                                            kernel_size=(5,5),
                                            strides=(2,2),
                                            padding='same',
                                            activation=tf.nn.relu)

        # Layer 7: Conv_transpose to 128x128x10
        conv_t7 = tf.layers.conv2d_transpose(inputs=conv_t6, 
                                            filters=10, 
                                            kernel_size=(5,5),
                                            strides=(1,1),
                                            padding='same',
                                            activation=tf.nn.relu)
        
        # Layer 8: Conv to 64x64x1
        conv8 = tf.layers.conv2d(inputs=conv_t7, 
                                filters=1, 
                                kernel_size=(3,3),
                                strides =(1,1),
                                padding='same', 
                                activation=None)
        
        # Make logits 
        logits = tf.slice(conv8, [0, 0, 0, 0], [-1, 64, 64, 1])

        # Pass logits through sigmoid to get reconstructed image
        self.decoded_img = tf.nn.sigmoid(logits)

        # Pass logits through sigmoid and calculate the cross-entropy
        entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_img, 
                                                            logits=logits)

        # Get loss and define the optimizer
        self.loss = tf.reduce_mean(entropy)
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # Accuracy (average image intensity difference)
        img_diff = tf.math.sqrt(tf.math.square(self.decoded_img
                                                - self.target_img))
        self.accuracy = tf.reduce_mean(img_diff)


    def evaluate(self, x=None, y=None, batch_size=20):
        '''
        Evaluate a model
        '''
        total_loss, total_accuracy = 0, 0
        for i in range(x.shape[0]//batch_size):

            x_batch = x[i*batch_size:(i+1)*batch_size,:,:,:]
            y_batch = y[i*batch_size:(i+1)*batch_size,:,:,:]

            l, acc = self.sess.run([self.loss, self.accuracy], 
                                feed_dict={self.input_img:x_batch,
                                        self.target_img:y_batch})

            total_loss += l
            total_accuracy += acc

        mean_loss = total_loss/(i+1)
        mean_acc = total_accuracy/(i+1)
        print('Validation loss: {:.3f} - Validation accuracy: {:.3f}'.format(
                                                        mean_loss, mean_acc))



    def fit(self, x=None, y=None, batch_size=20, epochs=1, verbose=1, 
            shuffle=True, val_x=None, val_y=None, initial_epoch=0, 
            save_path=None):
        '''
        The train function based on Keras api
        '''
        
        # Make dir for checkpoints if doesn't exist
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            saver = tf.train.Saver()


        for e in range(initial_epoch, epochs+initial_epoch):
            
            # start_time = time.time()

            if shuffle:
                x, y = sklearn.utils.shuffle(x, y)
            
            print('Epoch {:}/{:}'.format(e, epochs+initial_epoch))
            progbar = tf.keras.utils.Progbar(x.shape[0], verbose=verbose)
            
            for i in range(x.shape[0]//batch_size):
                
                x_batch = x[i*batch_size:(i+1)*batch_size,:,:,:]
                y_batch = y[i*batch_size:(i+1)*batch_size,:,:,:]

                l, _, acc = self.sess.run([self.loss, self.opt, self.accuracy], 
                                    feed_dict={self.input_img:x_batch,
                                                self.target_img:y_batch})

                progbar.add(x_batch.shape[0], values=[("Loss", l), 
                                            ("Accuracy", acc)])
            

            # Perform Validation
            if np.any(val_x):
                self.evaluate(x=val_x, y=val_y, 
                            batch_size=batch_size)
            if save_path:
                saver.save(self.sess, save_path,  global_step=e)
            # print('Time elapsed in current iteration: {:.2f} seconds'.format(
            #                                         time.time() - start_time))

    def predict(self, x=None, batch_size=20):
        '''
        Predict a model
        '''

        for i in range(x.shape[0]//batch_size):

            x_batch = x[i*batch_size:(i+1)*batch_size,:,:,:]
            
            predicted_img = self.sess.run([self.decoded_img], 
                                feed_dict={self.input_img:x_batch})

            x[i*batch_size:(i+1)*batch_size,:,:,:] = predicted_img
    
        return x


    def save(self, save_path='./checkpoints/my_model.ckpt'):

        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        print("Model saved in path: %s" % save_path)

    
    def restore(self, model_path='./checkpoints/my_model.ckpt'):
        
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        print("Model restored.")


# if __name__ == '__main__':
#     model = ImageCleaner()
#     # model.train(n_epochs=15)