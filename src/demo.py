from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      #print(ret['results'][6][0][4])
      #Removing the null/none scored class from the score dictionary(ret)
      dic_out = {x:y for x,y in ret['results'].items() if y.any()}
      #print(dic_out)
      #Removing the predicted scores<0.5 classes - ret is a value sorted dictionary so, y[0] has the largest prediction score object of the class
      new_ret = {x:y for x,y in dic_out.items() if y[0][4]>0.5}
      #print(new_ret)
      count=0
      my_dict={}
      for i in new_ret.items():
        #print(i[0])
        for j in i[1]:
          if(j[4]>0.4):
            count+=1
        #print("count",count)  
        my_dict[i[0]]=count 
      print(my_dict)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
