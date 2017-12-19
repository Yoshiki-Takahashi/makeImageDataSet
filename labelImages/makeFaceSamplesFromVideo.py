#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
passenger Tracker
====================

Usage
-----
track_passengers.py [<video_source>] (<fps> or <fps:every_x_frame>) (global_scale) (output_folder) (rotation) (track_window)


Keys
----
ESC - exit
'''

import numpy as np
import cv2
import skvideo.io
import math
from common import draw_str
from time import clock

import sys
import time
import os
import threading
from Queue import Queue
from ast import literal_eval
import argparse

# detection network accept a fixed size image
#width = 240
#width = 480
width = 640
#width = 1248
half_width = width / 2
#height = 360
height = 480
#height = 384
half_height = height / 2
grid_size = 32

class Passenger:

    def __init__(self, passenger_class, occlusion, left, top, right, bottom):
        self.passenger_class = passenger_class
        self.occlusion = occlusion
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def describe(self, passenger_class_label):
        return passenger_class_label + \
                ' 0.00 ' + str(self.occlusion) + ' -10 ' + \
                str('{0:.2f}'.format(self.left)) + ' ' + \
                str('{0:.2f}'.format(self.top)) + ' ' + \
                str('{0:.2f}'.format(self.right)) + ' ' + \
                str('{0:.2f}'.format(self.bottom)) + \
                ' -1 -1 -1 -1000 -1000 -1000 -10'

class CameraReader(threading.Thread):
    def __init__(self, camera_queue, video_src, fps, every_x_frame):
        super(CameraReader, self).__init__()
        self.camera_queue = camera_queue
        self.video_src = video_src
        self.fps = fps
        self.finished = False
        self.paused = False
        self.frame_idx = 0
        self.every_x_frame = every_x_frame
        self.busy = False
        self.frame_rates = [0.0]
        self.resume_temporary = False

    def setup(self):
        try:
            self.cam = skvideo.io.VideoCapture(int(self.video_src))
        except:
            self.cam = skvideo.io.VideoCapture(self.video_src)        
        if self.cam is None or not self.cam.isOpened():
            print(self.video_src + ' is not available')
            sys.exit(1)
        # 0: 0.0
        # VIDEOIO ERROR: V4L2: getting property #1 is not supported
        # 1: -1.0
        # VIDEOIO ERROR: V4L2: getting property #2 is not supported
        # 2: -1.0
        # 3: 640.0 <== WIDTH
        # 4: 480.0 <== HEIGHT
        # 5: 30.0  <== FPS
        # 6: 1448695129.0
        # VIDEOIO ERROR: V4L2: getting property #7 is not supported
        # 7: -1.0
        # 8: 16.0
        # 9: 1448695129.0
        # 10: 0.21568627451
        # 11: 0.333333333333
        # 12: 0.501960784314
        # 13: 0.5
        # 14: 0.0
        # VIDEOIO ERROR: V4L2: Exposure is not supported by your device
        # 15: -1.0
        # 16: 1.0
        # VIDEOIO ERROR: V4L2: getting property #17 is not supported
        # 17: -1.0
        # VIDEOIO ERROR: V4L2: getting property #18 is not supported
        # 18: -1.0
        # set camera properties
        # Full HD
        #self.cam.set(3, 1920)
        #self.cam.set(4, 1200)
        # HD
        #self.cam.set(3, 1280)
        #self.cam.set(4, 720)
        #self.cam.set(5, self.fps)

    def run(self):
        self.setup()
        # read frames
        frame = None
        while not self.finished:
            # measure frame rate - start
            start = cv2.getTickCount()
            #print('reading cam')
            if not self.paused or self.resume_temporary:
                if self.resume_temporary:
                    print('reading camera temporarily')
                ret, frame = self.cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #print('read cam')

            if self.resume_temporary:
                self.resume_temporary = False

            # finished
            if not ret:
                print('no frame is avaiable')
                break

            if self.every_x_frame > 1 and self.frame_idx % self.every_x_frame != 0:
                if self.paused:
                    ret, frame = self.cam.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_idx += 1
                continue

            # return the previous frame if paused
            try:
                self.camera_queue.put_nowait(frame)
            except:
                #print('cameraqueue is full')
                self.busy = True

                        # measure frame rate - end
            #print('measuring performance...')
            duration = (cv2.getTickCount() - start)/ cv2.getTickFrequency()
            rate = int(1 / duration)
            current_fps = self.fps
            if self.paused:
                current_fps = 5
            while current_fps < rate:
                time.sleep(0.01)
                # compute again
                duration = (cv2.getTickCount() - start)/ cv2.getTickFrequency()
                rate = int(1 / duration)
            self.frame_rates.append(rate)
            if len(self.frame_rates) > 10:
                del self.frame_rates[0]

            self.frame_idx += 1


        self.finished = True
        print('finished reading camera')

class App:

    def __init__(self, video_src, target_fps, every_x_frame, global_scale, output_folder, rotation, track_window, skip_to_frame):
        self.output_folder = output_folder
        self.fps = target_fps
        self.camera_queue = Queue(2)
        self.camera_reader = CameraReader(self.camera_queue, video_src, self.fps, every_x_frame)
        self.global_scale = global_scale
        self.rotation = rotation

        cv2.namedWindow('Passenger Tracker')
        cv2.setMouseCallback('Passenger Tracker', self.settracker)
        self.drag_start = None
        if track_window is None:
            self.track_window = None
            self.selection = None
        else:
            self.track_window = track_window
            self.selection = [track_window[0], track_window[1], track_window[0] + track_window[2], track_window[1] + track_window[3]]

        self.skip_to_frame = skip_to_frame

        # to be set
        self.rows = 0
        self.cols = 0

        # label generation
        self.passengers = []
        self.occlusion = 0
        self.passenger_class = 'shudo'
        self.passenger_classes = ['shudo', 'ohnishi', 'iwasaki', 'soga', 'aoki', 'yoshiki', 'matsumura', 'kanda', 'ken', 'nakajima', 'weiliang','all']
        self.imgdir = self.output_folder + 'images/'
        self.labelsdir = self.output_folder + 'labels/'
        for vc in self.passenger_classes:
            if not os.path.exists(self.imgdir + vc):
                os.makedirs(self.imgdir + vc)
            if not os.path.exists(self.labelsdir + vc):
                os.makedirs(self.labelsdir + vc)
        self.alias = video_src[video_src.rfind('/')+1:video_src.rfind('.')]
        self.current = None

        # augumentation
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # counter mode
        self.counter_mode = False
        self.counters = None

    def settracker(self, event, x, y, flags, param):
        if self.camera_reader.paused and self.track_window is not None:
            self.setlabels(event, x, y, flags, param)
            return
        #print('mouse event = ' + str(event))
        #print('flags = ' + str(flags))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.track_window = None
            self.selection = None
            self.count_line_candidate = None
            self.count_lines = []
            self.drag_start = (x, y)
            return
        if self.drag_start is not None:
            xmin = max(0, x - half_width)
            ymin = max(0, y - half_height)
            xmax = min(self.cols, x + half_width)
            ymax = min(self.rows, y + half_height)
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            if self.selection is None:
                return
            if (xmax - xmin == 0) or (ymax - ymin == 0):
                return
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
            print('tracking area=' + str(self.track_window))

    def setlabels(self, event, x, y, flags, param):
        self.current = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selection = None
            self.drag_start = (x, y)
            return
        if self.drag_start is not None:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            if self.selection is None:
                return
            if (xmax - xmin == 0) or (ymax - ymin == 0):
                return
            xmin = max(0, xmin-self.track_window[0])
            ymin = max(0, ymin-self.track_window[1])
            xmax = min(width, xmax-self.track_window[0])
            ymax = min(height, ymax-self.track_window[1])
            v = Passenger(self.passenger_class, self.occlusion, xmin, ymin, xmax, ymax)
            print('adding: ' + v.describe(v.passenger_class))
            self.passengers.append(v)
            self.selection = None

    def run(self):
        scale_x = 1.0
        scale_y = 1.0
        frame_rates = []
        isReady = False
        # detection sample generation
        paused = False
        scene = None
        offset_y = 0
        offset_x = 0

        self.camera_reader.start()
        while not self.camera_reader.finished:
            # measure frame rate - start
            start = cv2.getTickCount()

            # block 1 second
            try:
                frame = self.camera_queue.get(True, 60)
            except:
                print('failed to read camera frame from camera queue within 60 second, exitting...')
                break

            while self.camera_reader.frame_idx < self.skip_to_frame:
                if (self.skip_to_frame - self.camera_reader.frame_idx) % 100 == 0:
                    print('skipping to ' + str(self.skip_to_frame) + ', now at ' + str(self.camera_reader.frame_idx))
                self.camera_reader.frame_idx = self.camera_reader.frame_idx + 1
                frame = self.camera_queue.get(True, 60)
                continue

            # rotate if required
            if self.rotation != 0:
                rows, cols, channels = frame.shape
                M = cv2.getRotationMatrix2D((cols/2, rows/2), self.rotation, 1)
                frame = cv2.warpAffine(frame, M, (cols, rows))
                #print('rotated ' + str(self.rotation) + ' degrees of (' + str(rows) + ',' + str(cols) + ')')

            resized = frame
            if self.global_scale != 1.0:
                #print('resizing...')
                resized = cv2.resize(frame,None, fx=self.global_scale, fy=self.global_scale, interpolation=cv2.INTER_AREA)
            #print('copying...')
            vis = resized.copy()
            self.rows, self.cols = vis.shape[:2]
            if self.track_window is None:

                if self.selection is not None:
                    # draw selected region
                    cv2.rectangle(vis, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]), (0, 0, 255), 3)

            else:

                if self.camera_reader.paused:
                    # draw the current selection
                    if self.selection is None:
                        if self.current is not None:
                            # draw horizontal
                            cv2.line(vis, (0, self.current [1]), (len(vis[0]), self.current [1]), (255,0,0), 1)
                            # draw vertical
                            cv2.line(vis, (self.current [0], 0), (self.current [0], len(vis[1])), (255,0,0), 1)
                    else:
                        cv2.rectangle(vis, (self.selection[0], self.selection[1]), (self.selection[2], self.selection[3]), (0, 0, 255), 1)

                    # draw bboxes
                    for v in self.passengers:
                        cv2.rectangle(vis, (v.left+self.track_window[0], v.top+self.track_window[1]), (v.right+self.track_window[0], v.bottom+self.track_window[1]), (255, 0, 255), 2)

                #print('processing image...')
                from_x = self.track_window[0]
                from_y = self.track_window[1]
                to_x = from_x + self.track_window[2]
                to_y = from_y + self.track_window[3]

                # crop
                cropped = resized[from_y:to_y, from_x:to_x, :]

                # draw tracked region
                cv2.rectangle(vis, (from_x, from_y), (to_x, to_y), (0, 255, 0), 3)
                resized_grid_size = grid_size
                current = from_x + resized_grid_size
                while current < to_x:
                    # draw vertical lines
                    cv2.line(vis,(current, from_y),(current, to_y), (0,255,0), 1)
                    current += resized_grid_size
                current = from_y + resized_grid_size
                while current < to_y:
                    # draw horizontal lines
                    cv2.line(vis,(from_x, current),(to_x, current), (0,255,0), 1)
                    current += resized_grid_size

                scene = cropped
                #print("offset x,y = " + str(offset_x) + "," + str(offset_y))

            # measure frame rate - end
            #print('measuring performance...')
            duration = (cv2.getTickCount() - start)/ cv2.getTickFrequency()
            rate = int(1 / duration)
            current_fps = self.fps
            if self.camera_reader.paused:
                current_fps = 5
            while current_fps < rate:
                time.sleep(0.01)
                # compute again
                duration = (cv2.getTickCount() - start)/ cv2.getTickFrequency()
                rate = int(1 / duration)
            frame_rates.append(rate)
            if len(frame_rates) > 10:
                del frame_rates[0]
            #m, s = divmod(int(self.camera_reader.cam.get(0)/1000), 60)

            # draw
            #draw_str(vis, (20, 40), '%02d:%02d frame index: %d' % (m, s, self.camera_reader.frame_idx))
            draw_str(vis, (20, 60), 'frame rate: %d/%d' % (np.mean(frame_rates), self.fps))

            if self.camera_reader.paused and self.track_window is not None:
                # label generations
                draw_str(vis, (20, 80), 'passenger class: ' + self.passenger_class)
            if self.counter_mode:
                draw_str(vis, (20, 80), 'Counter #1 (L, S, B) = (%d, %d, %d)' % (self.counters[0][0], self.counters[0][1], self.counters[0][2]))
                draw_str(vis, (20, 100), 'Counter #2 (L, S, B) = (%d, %d, %d)' % (self.counters[1][0], self.counters[1][1], self.counters[1][2]))
                draw_str(vis, (20, 120), 'Counter #3 (L, S, B) = (%d, %d, %d)' % (self.counters[2][0], self.counters[2][1], self.counters[2][2]))

            #print('finising up...')
            cv2.imshow('Passenger Tracker', vis)

            # key events
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            else:

                if self.camera_reader.paused:

                    if ch == 49: # 1(shudo)
                        passenger_class_index = 0
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 50: # 2(ohnishi)
                        passenger_class_index = 1
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 113: # q(iwasaki)
                        passenger_class_index = 2
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 119: # w(soga)
                        passenger_class_index = 3
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 97: # a(aoki)
                        passenger_class_index = 4
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 115: # s(yoshiki)
                        passenger_class_index = 5
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 100: # d(matsumura)
                        passenger_class_index = 6
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 122: # z(kanda)
                        passenger_class_index = 7
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 120: # x(ken)
                        passenger_class_index = 8
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 99: #c(nakajima)
                        passenger_class_index = 9
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 118: #v(weiliang)
                        passenger_class_index = 10
                        # update passenger class
                        self.passenger_class = self.passenger_classes[passenger_class_index]
                        print('passenger_class = ' + self.passenger_class)

                    elif ch == 102:
                        # up arrow, 
                        print('resuming temporarily')
                        self.camera_reader.resume_temporary = True
                        #self.skip_to_frame = self.camera_reader.frame_idx + 2
                        # down arrow(84), not used

                    elif ch == 116:
                        # resume
                        self.camera_reader.paused = False
                        self.passengers = []
                    elif ch == 114:
                        # save samples
                        frame_idx_str = '{0:04d}'.format(self.camera_reader.frame_idx)
                        if self.alias is not None:
                            # filename has to be a number
                            frame_idx_str = self.alias + '_' + frame_idx_str
                        subdirs = ['all/']
                        for v in self.passengers:
                            if (v.passenger_class + '/') not in subdirs:
                                subdirs.append(v.passenger_class + '/')
                        print('storing in ' + str(subdirs))
                        for subdir in subdirs:
                            # write an image
                            img1 = self.imgdir + subdir + frame_idx_str + '.jpg'
                            cv2.imwrite(img1, scene)
                            # equalize
                            #ycb = cv2.cvtColor(scene, cv2.COLOR_BGR2YCrCb)
                            ycb = cv2.cvtColor(scene, cv2.COLOR_BGR2YCR_CB)
                            y,c,b = cv2.split(ycb)
                            y_eq2 = self.clahe.apply(y)
                            y_eq3 = cv2.equalizeHist(y)
                            ycb_eq2 = cv2.merge((y_eq2, c, b))
                            #scene2 = cv2.cvtColor(ycb_eq2, cv2.COLOR_YCrCb2BGR)
                            scene2 = cv2.cvtColor(ycb_eq2, cv2.COLOR_YCR_CB2BGR)
                            img2 = self.imgdir + subdir + frame_idx_str + '_adaeq' + '.jpg'
                            cv2.imwrite(img2, scene2)
                            ycb_eq3 = cv2.merge((y_eq3, c, b))
                            #scene3 = cv2.cvtColor(ycb_eq3, cv2.COLOR_YCrCb2BGR)
                            scene3 = cv2.cvtColor(ycb_eq3, cv2.COLOR_YCR_CB2BGR)
                            img3 = self.imgdir + subdir + frame_idx_str + '_globaleq' + '.jpg'
                            cv2.imwrite(img3, scene3)
                            # write a label file
                            filename = self.labelsdir + subdir + frame_idx_str + '.txt'
                            f = open(filename, 'w')
                            filename2 = self.labelsdir + subdir + frame_idx_str + '_adaeq' + '.txt'
                            f2 = open(filename2, 'w')
                            filename3 = self.labelsdir + subdir + frame_idx_str + '_globaleq' + '.txt'
                            f3 = open(filename3, 'w')
                            for v in self.passengers:
                                pc = v.passenger_class
                                if not pc == 'dontcare' and subdir == 'all/':
                                    pc = 'passenger'
                                v_desc = v.describe(pc)
                                print(v_desc)
                                f.write(v_desc + '\n')
                                f2.write(v_desc + '\n')
                                f3.write(v_desc + '\n')
                            f.close()
                            f2.close()
                            f3.close()
                        # reset
                        self.passengers = []
                    #else:
                    #    print('pressed: ' + str(ch))

                elif self.counter_mode:

                    if ch == 109:
                        self.counter_mode = False
                        self.counters = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                        print('exitted counter mode')
                    elif ch == 116:

                        self.camera_reader.paused = True

                    # this is the mode to count passengers by hand
                    elif ch in (49, 50, 51): # 1, 2, 3
                        # the first counter
                        counter = self.counters[0]
                        if ch == 49:
                            counter[0] = counter[0] + 1
                        elif ch == 50:
                            counter[1] = counter[1] + 1
                        else:
                            counter[2] = counter[2] + 1
                    elif ch in (52, 53, 54): # 4, 5, 6
                        # the second counter
                        counter = self.counters[1]
                        if ch == 52:
                            counter[0] = counter[0] + 1
                        elif ch == 53:
                            counter[1] = counter[1] + 1
                        else:
                            counter[2] = counter[2] + 1
                    elif ch in (55, 56, 57): # 7, 8, 9
                        # the third counter
                        counter = self.counters[2]
                        if ch == 55:
                            counter[0] = counter[0] + 1
                        elif ch == 56:
                            counter[1] = counter[1] + 1
                        else:
                            counter[2] = counter[2] + 1
                else:

                    if ch == 116:

                        self.camera_reader.paused = True
                        
                    elif ch == 109: # manual count

                        self.counter_mode = True

                        self.counters = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                        print('entered counter mode')
                            

                    elif ch == -1:
                        continue
                    elif ch == 255:
                        # do nothing
                        continue
                    else:
                        print('pressed: ' + str(ch))
                    # ascii code(base 10)
                    # 27:  ESC
                    # 49:  1(shudo)
                    # 50:  2(ohnishi)
                    # 113: q(iwasaki)
                    # 119: w(soga)
                    # 97:  a(aoki)
                    # 115: s(yoshiki)
                    # 100: d(matsumura)
                    # 122: z(kanda)
                    # 120: x(ken)
                    # 99:  c(nakajima)
                    # 118: v(weiliang)
                    # 114: r(save)
                    # 102: f(next)
                    # 116: t(pause)
                    #109:  m(manual count)

        self.camera_reader.finished = True

        if self.counter_mode:
            print('counters #1: ' + str(self.counters[0]))
            print('counters #2: ' + str(self.counters[1]))
            print('counters #3: ' + str(self.counters[2]))

def main():
    parser = argparse.ArgumentParser(description='Labeling names to image frame from video.')
    parser.add_argument('video_src',       type=str, help='video file path')
    parser.add_argument('--fps',           type=int, default=10)
    parser.add_argument('--every_x_frame', type=int, default=1)
    parser.add_argument('--global_scale',  type=float, default=0.5)
    parser.add_argument('--output_folder', type=str, default='member/')
    parser.add_argument('--rotation',      type=int, default=0)
    parser.add_argument('--track_window',  type=str, default=None)
    parser.add_argument('--skip_to_frame',    type=int, default=0)
    args = parser.parse_args()

    try:
        track_window = literal_eval(args.track_window)
    except:
        track_window = None

    #print __doc__
    App(args.video_src, args.fps, args.every_x_frame, args.global_scale, args.output_folder, args.rotation, track_window, args.skip_to_frame).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
