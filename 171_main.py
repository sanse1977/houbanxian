import cv2
import numpy as np
import datetime
import time
def featrue_detection(frame,compare_image,threshold_1,threshold_2,a,b,i,j):
    frame = frame[a:b,i:j]
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame_gray, None)
    kp2, des2 = orb.detectAndCompute(compare_image, None)


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = list(filter(lambda x: x.distance < threshold_1, matches))  # 过滤不合格的相似点

    if len(matches) > threshold_2:
        return True
    else:
        return False

def background(frame,a,b,i,j):
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(background, (21, 21), 0)
    background_roi = background[a:b, i:j]
    return background_roi

def diff(frame,background_roi,threshold_1,a,b,i,j):
    es=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,4))

    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame=cv2.GaussianBlur(gray_frame,(21,21),0)
    gray_frame_roi=gray_frame[a:b,i:j]

    diff=cv2.absdiff(background_roi,gray_frame_roi)
    diff=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    diff=cv2.dilate(diff,es,iterations=2)
    diff_sum=np.sum(diff)

    if diff_sum > threshold_1:
        return True

    else:
        return False

def hsv_detection(frame_roi,background_mask_roi,threshold_1):
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))

    lower = np.array([100, 43, 46])
    upper = np.array([124, 255, 255])
    hsv = cv2.cvtColor(frame_roi,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower,upper)

    diff = cv2.absdiff(background_mask_roi, mask)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)
    diff_sum = np.sum(diff)
    if diff_sum > threshold_1:
        return True
    else:
        return False

def video_recogn171d(self):
    #ppeb动的条件1：蓝标动
    self.background_ppebn_workblue = cv2.imread('pictures/ppebDayOriginal.png')
    self.background_ppebn_workblue = cv2.resize(self.background_ppebn_workblue,(1080,720))
    self.background_ppebn_workblue = cv2.cvtColor(self.background_ppebn_workblue,cv2.COLOR_BGR2GRAY)
    self.background_ppebn_workblue_roi = self.background_ppebn_workblue[350:450,550:680]

    # #ppeb动的条件2：三角标动
    # background_ppebn_mach_worktri = cv2.imread('pictures/ppeb_work-0002.png')
    # background_ppebn_mach_worktri = cv2.cvtColor(background_ppebn_mach_worktri,cv2.COLOR_BGR2GRAY)
    # background_ppebn_mach_worktri = cv2.GaussianBlur(background_ppebn_mach_worktri,(21,21),0)
    # background_ppebn_worktri_roi = background_ppebn_mach_worktri[400:500,380:450]

    cap = cv2.VideoCapture('videos/ppeb_tiao.mp4')
    self.ppebn_work = [0]*50
    self.ppebn_angle_time = [0]*50
    self.ppebn_tuo_time = [0]*50
    self.ppebn_work_judge = False
    self.background_angle = None
    self.background_tuo = None
    self.background_ppebn_worktri_roi = None
    self.ppebn_tiao_judge = False
    self.ppebn_tuo_judge = False
    currenttime = time.time()
    while(True):
        ret,frame = cap.read()
        frame = cv2.resize(frame,(1080,720))
        if time.time()-currenttime >0.4:

            if self.background_ppebn_worktri_roi is None:
                self.background_ppebn_worktri_roi = background(frame,400,500,380,450)

            # frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ppebn_worktri = self.vision.diff(frame,self.background_ppebn_worktri_roi,40000,400,500,380,450)
            #print('worktri',ppebn_worktri)
            ppebn_workblue = self.vision.featrue_detection(frame,self.background_ppebn_workblue_roi,60,35,350,450,550,680)
            #print('worlblue',ppebn_workblue)
            if ppebn_worktri is True and ppebn_workblue is False :#机器动
                self.ppebn_work.pop(0)
                self.ppebn_work.append(1)

            else:#机器停止
                self.ppebn_work.pop(0)
                self.ppebn_work.append(0)
                if self.background_angle is None:
                    self.background_angle = self.vision.background(frame,520,680,0,150)

                if self.background_tuo is None:
                    self.background_tuo = self.vision.background(frame,550,950,1000,1700)

                ppebn_tiao_angle = self.vision.diff(frame,self.background_angle,10000,520,680,0,150)
                ppebn_tuo = self.vision.diff(frame,self.background_tuo,10000,550,950,1000,1700)
                if ppebn_tiao_angle is True:
                    self.ppebn_angle_time.pop(0)
                    self.ppebn_angle_time.append(1)

                else:
                    self.ppebn_angle_time.pop(0)
                    self.ppebn_angle_time.append(0)

                if ppebn_tuo is True:
                    self.ppebn_tuo_time.pop(0)
                    self.ppebn_tuo_time.append(1)

                else:
                    self.ppebn_tuo_time.pop(0)
                    self.ppebn_tuo_time.append(0)

                ppebnt = sum(self.ppebn_angle_time)
                ppebntuo = sum(self.ppebn_tuo_time)


                if ppebnt >40 and self.ppebn_tiao_judge is False:
                    self.ppebn_tiao_start = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    print(self.ppebn_tiao_start)
                    self.ppebn_tiao_judge = True

                elif ppebnt < 10 and self.ppebn_tiao_judge is True:
                    self.ppebn_tiao_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    self.ppebn_tiao_judge = False
                    print(self.ppebn_tiao_stop)

                if ppebntuo > 40 and self.ppebn_tuo_judge is False:
                    self.ppebn_tuo_start = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    self.ppebn_tuo_judge = True

                if ppebntuo < 10 and self.ppebn_tuo_judge is True:
                    self.ppebn_tuo_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    self.ppebn_tuo_judge = False


            ppebn_work_res = sum(self.ppebn_work)
            # print('机器',ppebn_work_res)
            if ppebn_work_res > 40 and self.ppebn_work_judge is False:
                self.ppebn_work_start = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                self.ppebn_work_judge = True
                print('机器工作',self.ppebn_work_start)

            if ppebn_work_res < 10 and self.ppebn_work_judge is True:
                self.ppebn_work_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                self.ppebn_work_judge = False
                print('机器停工',self.ppebn_work_stop)

            currenttime = time.time()


        cv2.imshow('frame',frame)
        cv2.waitKey(10)



video_recogn171d()



