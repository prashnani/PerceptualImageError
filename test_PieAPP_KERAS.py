import os
import cv2
import sys
import glob
sys.path.append('model/')
from model.PieAPPv0pt1_KERAS import PieAPP
import argparse
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

####### check for model and download if not present
if not len(glob.glob('weights/PieAPP_model_v0.1.ckpt.*')) == 3:
	print ("downloading dataset")
	os.system("bash scripts/download_PieAPPv0.1_TF_weights.sh")
	if not len(glob.glob('weights/PieAPP_model_v0.1.ckpt.*')) == 3:
		print ("PieAPP_model_v0.1.ckpt files not downloaded")
		sys.exit()
        
def compare_PieAPP(imageRef, imageA, gpu_id='', sampling_mode='sparse'):
    assert sampling_mode=='sparse' or sampling_mode=='dense', 'sampling_mode must be sparce or dense, received '+sampling_mode
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    
    if sampling_mode == 'sparse':	
        stride_val = 27
    if sampling_mode == 'dense':
        stride_val = 6
        
    pieModel = PieAPP()
    pieModel.load_weights('weights/PieAPP_model_v0.1.ckpt.meta')
    return pieModel.predict(imageRef, imageA, strides=stride_val)
    
if __name__=='__main__':
    ######## input args
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_path", dest='ref_path', type=str, default='imgs/ref.png', help="specify input reference")
    parser.add_argument("--A_path", dest='A_path', type=str, default='imgs/A.png', help="specify input image")
    parser.add_argument("--sampling_mode", dest='sampling_mode', type=str, default='sparse', help="specify sparse or dense sampling of patches to compte PieAPP")
    parser.add_argument("--gpu_id", dest='gpu_id', type=str, default='', help="specify which GPU to use (don't specify this argument if using CPU only)")
    
    args = parser.parse_args()
    
    # open images
    imageRef = cv2.imread(args.ref_path)
    imageA   = cv2.imread(args.A_path)
    
    compared = compare_PieAPP(imageRef, imageA, args.gpu_id, args.sampling_mode)
    
    print  ('PieAPP value of '+args.A_path+ ' with respect to: '+str(compared))