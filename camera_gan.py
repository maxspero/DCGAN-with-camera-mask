# don't forget to open the tensorflow conda virtual environment first!
# source tensorflow

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
config.batch_size = 1
seconds_per_random_sample = 4

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

def process_webcam_image(img, x_scale, y_scale, flip=True, threshold=0.5):
  res = cv2.resize(img, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_AREA) 
  res_min = np.min(res)
  res_max = np.max(res)
  scaled = (res - res_min)/float(res_max - res_min)
  flipped = cv2.flip(scaled, 1) if flip else scaled
  flipped[flipped>threshold] = 1
  flipped[flipped<=threshold] = 0
  return flipped

def bucket(frame, axis, num_buckets, invert): # for grayscale
  if invert:
    frame = 1-frame
  length = frame.shape[axis]
  bucket_length = length/num_buckets
  buckets = np.zeros(num_buckets)
  for i in range(num_buckets):
    if axis == 0:
      buckets[i] = np.sum(frame[int(i*bucket_length):int((i+1)*bucket_length),:])
    elif axis == 1:
      buckets[i] = np.sum(frame[:,int(i*bucket_length):int((i+1)*bucket_length)])
  print(frame)
  print(buckets)
  return np.argmax(buckets)

z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))

z_mask_empty = np.ones([dcgan.z_dim])
h0_mask_empty = np.ones([int(dcgan.output_height/16), int(dcgan.output_width/16)]) # 4, 4
h1_mask_empty = np.ones([int(dcgan.output_height/8), int(dcgan.output_width/8)]) # 8, 8
h2_mask_empty = np.ones([int(dcgan.output_height/4), int(dcgan.output_width/4)]) # 16, 16
h3_mask_empty = np.ones([int(dcgan.output_height/2), int(dcgan.output_width/2)]) # 32, 32
h4_mask_empty = np.ones([int(dcgan.output_height), int(dcgan.output_width)]) # 64, 64

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
time_to_switch = time.time() + seconds_per_random_sample

prev_output_window = np.zeros((3, 64, 64, 3))
window_iter = 0
reset_window = True

while is_capturing:
    try:    # Lookout for a keyboardInterrupt to stop the script
        is_capturing, frame = vc.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # makes the blues image look real colored
            #print(frame.shape) # (640, 480) 
            # resize key:
            #   32, 32 -> fx=1/20, fy=1/15
            #   16, 16 -> fx=1/40, fy=1/30
            #   8, 8   -> fx=1/80, fy=1/60
            #   4, 4   -> fx=1/160, fy=1/48
            #   10, 10 -> fx=1/64, fy=1/48
            if(time.time() > time_to_switch):
              time_to_switch += seconds_per_random_sample
              z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
              reset_window = True
            z_mask = z_mask_empty
            h0_mask = h0_mask_empty
            h1_mask = h1_mask_empty
            h2_mask = h2_mask_empty
            h3_mask = h3_mask_empty
            h4_mask = h4_mask_empty


            #bucket_choice = bucket(process_webcam_image(frame, 1/20, 1/15, threshold=.3), 1, 4, True)
            bucket_choice = max(bucket(process_webcam_image(frame, 1/20, 1/15, threshold=.3), 1, 5, True) - 1, 0)
            
            mask_choice = 'z' + str(bucket_choice)
            #mask_choice = 'z012'
            display_mask = True
            mask_size = (200, 200)
            face_size = (600, 600)
            if 'z' in mask_choice:
              z_mask_10x10 = process_webcam_image(frame, 1/64, 1/48, False)
              z_mask = np.reshape(z_mask_10x10, (100))
              if display_mask:
                resized_z_mask = cv2.resize(z_mask_10x10, mask_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('z_mask', resized_z_mask)
            if '0' in mask_choice:
              h0_mask = process_webcam_image(frame, 1/160, 1/120, threshold=.3)
              if display_mask:
                resized_h0_mask = cv2.resize(h0_mask, mask_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('h0_mask', resized_h0_mask)
            if '1' in mask_choice:
              h1_mask = process_webcam_image(frame, 1/80, 1/60, threshold=.3)
              if display_mask:
                resized_h1_mask = cv2.resize(h1_mask, mask_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('h1_mask', resized_h1_mask)
            if '2' in mask_choice:
              h2_mask = process_webcam_image(frame, 1/40, 1/30, threshold=.3)
              if display_mask:
                resized_h2_mask = cv2.resize(h2_mask, mask_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('h2_mask', resized_h2_mask)
            if '3' in mask_choice:
              h3_mask = process_webcam_image(frame, 1/20, 1/15, threshold=.3)
              if display_mask:
                resized_h3_mask = cv2.resize(h3_mask, mask_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('h3_mask', resized_h3_mask)

            #h1_mask = process_webcam_image(frame, 1/80, 1/60)
            
            feed_dict = {
                dcgan.z: z_sample, 
                dcgan.z_mask: z_mask,
                dcgan.h0_mask: h0_mask,
                dcgan.h1_mask: h1_mask,
                dcgan.h2_mask: h2_mask,
                dcgan.h3_mask: h3_mask,
                dcgan.h4_mask: h4_mask,
            }
            samples = sess.run(dcgan.sampler, feed_dict=feed_dict)
            prev_output_window[window_iter%3] = samples[0]
            if reset_window:
              prev_output_window[:] += samples[0]
              reset_window = False
              window_iter = 0
            window_iter += 1
            #avg = np.sum(samples, axis=0)/5
            avg = (prev_output_window[0] + prev_output_window[1] + prev_output_window[2])/3
            img = cv2.cvtColor(deprocess_image(avg), cv2.COLOR_RGB2BGR)
            resized = cv2.resize(img, face_size, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('img', resized)
            feed_dict_empty = {
                dcgan.z: z_sample, 
                dcgan.z_mask: z_mask_empty,
                dcgan.h0_mask: h0_mask_empty,
                dcgan.h1_mask: h1_mask_empty,
                dcgan.h2_mask: h2_mask_empty,
                dcgan.h3_mask: h3_mask_empty,
                dcgan.h4_mask: h4_mask_empty,
            }
            samples_empty = sess.run(dcgan.sampler, feed_dict=feed_dict_empty)
            img_empty = cv2.cvtColor(deprocess_image(samples_empty[0]), cv2.COLOR_RGB2BGR)
            resized_empty = cv2.resize(img_empty, face_size, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('img_empty', resized_empty)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
              print('q pressed...')
              vc.release()
              cv2.destroyAllWindows()
              break
            
            #webcam_preview = plt.imshow(flipped)
            #webcam_preview.set_data(flipped)
            #plt.draw()
            #break
            
    except(e):
        vc.release()
        print('Exception!')
        print(e)


