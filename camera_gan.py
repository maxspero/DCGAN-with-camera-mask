import sys
print(sys.version)
import numpy as np
import tensorflow as tf
from time import gmtime, strftime
import time
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import model
import utils
import cv2
print(cv2.__version__)
from IPython import display

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

config = type("Foo", (object,), {})()
config.dataset = 'celebA'
config.batch_size = 64

#with tf.Session(config=run_config) as sess:
sess = tf.Session(config=run_config) 
dcgan = model.DCGAN(
    sess,
    input_height=108,
    input_width=108,
    output_width=64,
    output_height=64,
    batch_size=config.batch_size,
    sample_num=64,
    dataset_name='celebA',
    input_fname_pattern='*.jpg',
    crop=True, #true for training
    checkpoint_dir='checkpoint',
    sample_dir='samples'
)

if not dcgan.load('checkpoint')[0]:
    print('Cannot find checkpoint!')

utils.show_all_variables()

def get_mask(file):
    image = imread(file)
    a = image[:,:,0]
    b = a/np.max(a)
    c = np.rint(b)
    return c

def deprocess_image(img):
    return np.clip(255 * (img+0.5), 0.0, 255.0).astype(np.uint8)

z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))

z_mask = np.ones([dcgan.z_dim])
h0_mask = np.ones([int(dcgan.output_height/16), int(dcgan.output_width/16)]) # 4, 4
h1_mask = np.ones([int(dcgan.output_height/8), int(dcgan.output_width/8)]) # 8, 8
h2_mask = np.ones([int(dcgan.output_height/4), int(dcgan.output_width/4)]) # 16, 16
h3_mask = np.ones([int(dcgan.output_height/2), int(dcgan.output_width/2)]) # 32, 32
h4_mask = np.ones([int(dcgan.output_height), int(dcgan.output_width)]) # 64, 64

# vc.release()
vc = cv2.VideoCapture(1)

if vc.isOpened(): # try to get the first frame
    is_capturing, frame = vc.read()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # makes the blues image look real colored
        webcam_preview = plt.imshow(frame)    
else:
    is_capturing = False
    
print(is_capturing)
while is_capturing:
    try:    # Lookout for a keyboardInterrupt to stop the script
        is_capturing, frame = vc.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # makes the blues image look real colored
            #print(frame.shape) # (480, 640) * (1/15, 1/20) = (32, 32)
            #res = cv2.resize(frame, None, fx=1/20, fy=1/15, interpolation=cv2.INTER_AREA) # 32, 32
            #res = cv2.resize(frame, None, fx=1/40, fy=1/30, interpolation=cv2.INTER_AREA) # 16, 16
            res = cv2.resize(frame, None, fx=1/80, fy=1/60, interpolation=cv2.INTER_AREA) # 8, 8
            #print(res.shape)
            res_min = np.min(res)
            res_max = np.max(res)
            scaled = (res - res_min)/float(res_max - res_min)
            flipped = cv2.flip(scaled, 1)
            flipped[flipped>.5] = 1
            flipped[flipped<=.5] = 0
            h1_mask = flipped
            
            feed_dict = {
                dcgan.z: z_sample, 
                dcgan.z_mask: z_mask,
                dcgan.h0_mask: h0_mask,
                dcgan.h1_mask: h1_mask,
                dcgan.h2_mask: h2_mask,
                dcgan.h3_mask: h3_mask,
                dcgan.h4_mask: h4_mask,
            }
            print(int(round(time.time() * 1000)))
            samples = sess.run(dcgan.sampler, feed_dict=feed_dict)
            print(int(round(time.time() * 1000))) # about 60ms on gpu to run 
            #utils.save_images(samples, [8,8], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
            #for image in samples[:3]:
            #    cv2.imshow('img', deprocess_image(image))
            #resized = cv2.resize(flipped, (800, 800), interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('img', resized)
            img = cv2.cvtColor(deprocess_image(samples[0]), cv2.COLOR_RGB2BGR)
            resized = cv2.resize(img, (800, 800), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('img', resized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
              print('q pressed...')
              vc.release()
              cv2.destroyAllWindows()
              break
            
            #webcam_preview = plt.imshow(flipped)
            #webcam_preview.set_data(flipped)
            #plt.draw()
            #break
            
    except:
        print('Exception!')
        vc.release()


