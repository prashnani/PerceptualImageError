import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from keras.models import Model
import keras.layers as L

class PieAPP(object):

    def __init__(self):
        # build networks
        self.extract_features = self._build_extract_features()
        self.compute_scores   = self._build_compute_scores()
    
    def _build_extract_features(self):
        '''
        Extract features
        '''

        input_img = L.Input((64,64,3), name='input_img')
        # conv1
        conv1 = L.Conv2D(64, 3, padding='same', activation='relu', name='conv1') (input_img)
        # conv2
        conv2 = L.Conv2D(64, 3, padding='same', activation='relu', name='conv2') (conv1)
        pool2 = L.MaxPooling2D(2, 2, padding='same', name='pool2') (conv2)
        # conv3
        conv3 = L.Conv2D(64, 3, padding='same', activation='relu', name='conv3') (pool2)
        f3    = L.Flatten() (conv3)
        # conv4
        conv4 = L.Conv2D(64*2, 3, padding='same', activation='relu', name='conv4') (conv3)
        pool4 = L.MaxPooling2D(2, 2, padding='same', name='pool4') (conv4)
        # conv5
        conv5 = L.Conv2D(64*2, 3, padding='same', activation='relu', name='conv5') (pool4)
        f5    = L.Flatten() (conv5)
        # conv6
        conv6 = L.Conv2D(64*2, 3, padding='same', activation='relu', name='conv6') (conv5)
        pool6 = L.MaxPooling2D(2, 2, padding='same', name='pool6') (conv6)
        # conv7
        conv7 = L.Conv2D(64*4, 3, padding='same', activation='relu', name='conv7') (pool6)
        f7    = L.Flatten() (conv7)
        # conv8
        conv8 = L.Conv2D(64*4, 3, padding='same', activation='relu', name='conv8') (conv7)
        pool8 = L.MaxPooling2D(2, 2, padding='same', name='pool8') (conv8)
        # conv9
        conv9 = L.Conv2D(64*4, 3, padding='same', activation='relu', name='conv9') (pool8)
        f9    = L.Flatten() (conv9)
        # conv10
        conv10 = L.Conv2D(64*8, 3, padding='same', activation='relu', name='conv10') (conv9)
        pool10 = L.MaxPooling2D(2, 2, padding='same', name='pool10') (conv10)
        # conv11
        conv11 = L.Conv2D(64*8, 3, padding='same', activation='relu', name='conv11') (pool10)
        f11    = L.Flatten() (conv11)
        ### flattening and concatenation
        feature_A_multiscale = L.concatenate([f3, f5, f7, f9, f11], axis=1)

        return Model(input_img, [feature_A_multiscale, f11], name='extract_features')

    def _build_compute_scores(self):
        '''
        Compute scores
        '''

        img_ref = L.Input((64,64,3), name='input_ref')
        img_A   = L.Input((64,64,3), name='input_compare')

        # feature extraction
        A_multiscale_feature, A_last_layer_feature     = self.extract_features(img_A)
        ref_multiscale_feature, ref_last_layer_feature = self.extract_features(img_ref)

        # feature difference
        diff_A_ref_ms   = L.Subtract() ([ref_multiscale_feature, A_multiscale_feature])
        diff_A_ref_last = L.Subtract() ([ref_last_layer_feature, A_last_layer_feature])

        # score computation
        # fc1
        fc1_A_ref = L.Dense(512, activation='relu', name='fc1') (diff_A_ref_ms)
        dropout1_A_ref = L.Dropout(0.2) (fc1_A_ref)
        # fc2
        fc2_A_ref = L.Dense(1, name='fc2') (dropout1_A_ref)
        multiply_const = L.Lambda(lambda x:tf.multiply(x, tf.constant(0.01))) (fc2_A_ref)
        per_patch_score = L.Lambda(lambda x:tf.reshape(x, [-1])) (multiply_const)

        ### weighing subnetwork image A and ref
        # fc1w
        fc1_A_ref_w = L.Dense(512, activation='relu', name='fc1w') (diff_A_ref_last)
        dropout1_A_ref_w = L.Dropout(0.2) (fc1_A_ref_w)
        # fc2w
        fc2_A_ref_w = L.Dense(1, name='fc2w') (dropout1_A_ref_w)
        add_const = L.Lambda(lambda x:tf.add(x, tf.constant(0.000001))) (fc2_A_ref_w)
        per_patch_weight = L.Lambda(lambda x:tf.reshape(x, [-1])) (add_const)

        # weighted average of scores
        product_score_weights_A = L.Multiply() ([per_patch_weight, per_patch_score])
        norm_factor_A = L.Lambda(lambda x:tf.reduce_sum(x)) (per_patch_weight)
        final_score_A = L.Lambda(lambda x:tf.divide(tf.reduce_sum(x[0]),x[1])) ([product_score_weights_A, norm_factor_A])

        return Model([img_ref, img_A], [final_score_A, per_patch_score, per_patch_weight], name='compute_scores')
    
    def get_models(self):
        '''
        Return extract feature and compare score models.
        '''
        return self.extract_features, self.compute_scores
    
    def load_weights(self, path_weights):
        '''
        Load weights in to the network.
        
        Paramenters:
        ------------
            path_weights (string): path to the weights (tensorflow checkpoint can be used)
        '''
        
        # load model from tensorflow check point
        if path_weights.endswith(('.ckpt', '.meta')):
            # start tensorflow session and get all layers weights and biases
            with tf.Session() as sess:
                # import graph
                saver = tf.train.import_meta_graph(path_weights)

                # load weights for graph
                saver.restore(sess, path_weights[:-5])

                # get all global variables (including model variables)
                vars_global = tf.global_variables()

                # get their name and value and put them into dictionary
                sess.as_default()
                model_vars = {}
                for var in vars_global:
                    try:
                        layer_name = var.name.split('/')[0]
                        layer_type = var.name.split('/')[-1].split(':')[0]

                        if not layer_name in model_vars:
                            model_vars[layer_name] = {layer_type:var.eval()}
                        else:
                            model_vars[layer_name].update({layer_type:var.eval()})
                    except Exception as e:
                        print("For var={}, an exception occurred".format(var.name))
                        print(str(e))
                        
            # load weights and biases in the keras layers
            def load_weights(model):
                for layer in model.layers:
                    try:
                        if layer.name in model_vars:
                            layer.set_weights([model_vars[layer.name]['weights'], model_vars[layer.name]['biases']])
                    except:
                        print(f'Layer {layer.name} weights could not be loaded!')
                        
            load_weights(self.extract_features)
            load_weights(self.compute_scores)
            
        #TODO: load from keras .h5 file
        else:
            raise Exception('Weights must be in .ckpt or .meta format!')
    
    # TODO: training code
    def train(self):
        pass
    
    def predict(self, imRef, imComp, strides=64):
        '''
        Predict PIE value over images.
        
        Paramenters:
        ------------
            imRef (np.array): reference image
            imComp (np.array): image to be compared
            strides (int): used strides to get batches from images (higher values will return less accurate, but faster results)
        '''
        
        # slinding window to feed network with batches of 64x64x3 images
        def sliding_window(matrix, winSize, pad):
            windows = []
            for i in range(0,matrix.shape[0]-winSize[0],pad):
                for j in range(0,matrix.shape[1]-winSize[1],pad):
                    win = matrix[i:i+winSize[0],j:j+winSize[1],...]
                    windows.append(win)

            return np.array(windows)
        
        # get batches of 64x64x3 images
        imRef  = sliding_window(imRef, (64,64), strides)
        imComp = sliding_window(imComp, (64,64), strides)
        
        # computate pie value
        _, PieAPP_patchwise_errors, PieAPP_patchwise_weights = self.compute_scores.predict([imRef, imComp])
        score_accum  = np.sum(np.multiply(PieAPP_patchwise_errors,PieAPP_patchwise_weights),axis=0)
        weight_accum = np.sum(PieAPP_patchwise_weights, axis=0)
        
        return(score_accum/weight_accum)