import cv2
import numpy as np
import datetime
import time

#特征匹配
def featrue_detection(frame,compare_image,threshold_1,threshold_2,a,b,i,j):
#输入：视频帧，对比的背景图片，筛选的特征点最多个数，判断阈值，感兴趣区域坐标
#输出：false：匹配成功；True：匹配失败

    # 截取感兴趣区域
    frame = frame[a:b,i:j]
    # 图像灰度化
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 背景图片的截取和灰度化
    compare_image = compare_image[a:b, i:j]
    compare_image_gray = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
    # orb特征算子
    orb = cv2.ORB_create()
    # 计算视频帧和背景图上的特征点
    kp1, des1 = orb.detectAndCompute(frame_gray, None)
    kp2, des2 = orb.detectAndCompute(compare_image_gray, None)

    # 暴力匹配两张图的特征
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 筛选最相关的a个特征点(a=threshold_1)
    matches = list(filter(lambda x: x.distance < threshold_1, matches))  # 过滤不合格的相似点

    # 如果匹配到的特征数超过阈值(threshold_2)就匹配成功
    if len(matches) > threshold_2:
        return False
    else:
        return True

# 帧间查分法的背景图处理
def background_diff(frame,a,b,i,j):
    # 灰度化
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯平滑
    background = cv2.GaussianBlur(background, (21, 21), 0)
    background_roi = background[a:b, i:j]
    return background_roi

# 颜色检测的背景图处理
def background_hsv(frame,a,b,i,j,lower, upper):
    # 将图片转化为hsv颜色空间下的图片
    background = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 高斯平滑
    background = cv2.GaussianBlur(background, (21, 21), 0)
    background_roi = background[a:b, i:j]
    # 提取图中某一个颜色的区域
    background_roi = cv2.inRange(background_roi, lower, upper)
    return background_roi

# 帧间差分法
def diff(frame,background_roi,threshold_1,a,b,i,j):
    # 输入：视频帧，背景图，阈值1，感兴趣图像区域坐标
    # 输出：True:该区域运动；False：该区域静止

    es=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,4))

    # 图像处理：灰度化+去噪
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame=cv2.GaussianBlur(gray_frame,(21,21),0)
    gray_frame_roi=gray_frame[a:b,i:j]

    #  差分过程：1.差分2.转化为二值图3.膨胀（对缺陷进行归一化处理）
    diff=cv2.absdiff(background_roi,gray_frame_roi)
    diff=cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
    diff=cv2.dilate(diff,es,iterations=2)
    #量化差分值
    diff_sum=np.sum(diff)

    # 如果差分值大于某一阈值，则开始工作
    if diff_sum > threshold_1:
        return True

    else:
        return False

# 基于hsv颜色空间的检测
def hsv_detection(frame,background_mask_roi,threshold_1,a,b,i,j,lower,upper):
    # 输入：视频帧，背景图，阈值1，坐标，要提取颜色的范围

    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))

    # 图像处理:平滑+转化为hsv空间+提取颜色
    frame_roi = frame[a:b,i:j]
    frame_roi = cv2.GaussianBlur(frame_roi, (21, 21), 0)
    hsv = cv2.cvtColor(frame_roi,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower,upper)

    # 差分操作
    diff = cv2.absdiff(background_mask_roi, mask)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, es, iterations=2)
    diff_sum = np.sum(diff)
    if diff_sum > threshold_1:
        return True
    else:
        return False

def line_diff(frame1,frame2,frame3,a,b,i,j):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,3))
    frame1 = frame1[a:b,i:j]
    frame2 = frame2[a:b, i:j]
    frame3 = frame3[a:b, i:j]

    gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    d1 = cv2.absdiff(gray2,gray1)
    d2 = cv2.absdiff(gray3,gray2)

    blur1 = cv2.GaussianBlur(d1,(5,5),0)
    blur2 = cv2.GaussianBlur(d2,(5,5),0)

    th1 = cv2.threshold(blur1,10,255,cv2.THRESH_BINARY)
    th2 = cv2.threshold(blur2,10,255,cv2.THRESH_BINARY)

    gra = cv2.bitwise_and(th1,th2)
    segmetation = cv2.medianBlur(gra,5)

    opening = cv2.morphologyEx(segmetation,cv2.MORPH_CLOSE,kernel)

    lines = cv2.HoughLinesP(opening,50,np.pi/180,10,minLineLength=100,maxLineGap=5)

    if lines is None:
        return False
    else:
        return True


# 蓝色范围
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])

# 黄色范围
lower_yellow = np.array([26, 43, 46])
upper_yellow = np.array([34, 255, 255])

# 绿色范围
lower_green = np.array([35, 43, 46])
upper_green = np.array([77, 255, 255])

# cap = cv2.VideoCapture('videos/格式工厂houban173.avi')
cap = cv2.VideoCapture('G:\格式工厂格式工厂1~1.mp4')

M5000_work_background = None
op20_work_background = cv2.imread('pictures/173dayOriginal.jpg')
op20_work_background = cv2.resize(op20_work_background,(1280,720))
M5000_fix_backgrond = cv2.imread('pictures/fixM5000base.png')
M5000_fix_backgrond = cv2.resize(M5000_fix_backgrond,(1280,720))
op20_night_background = cv2.imread('pictures/173nightOriginal.png')
op20_night_background = cv2.resize(op20_night_background,(1280,720))
M5000_work_start_judge = False
M5000_work_time = [0]*50
M5000_jud_start_judge = False
M5000_jud_time= [0]*20
houban_huan_backgrond = None
houban_huan_judge = False
houban_huan_time= [0]*20
M5000_fix_judge = False
M5000_fix_time= [0]*20
houban_material_backgrond = None
houban_material_judge = False
houban_material_all_judge = False
houban_material_time= [0]*10
houban_material_all_time = [0]*10
currenttime = time.time()
while(True):
    ret,frame = cap.read()
    frame = cv2.resize(frame,(1280,720))
    # op20_night = featrue_detection(frame,op20_night_background,60,50,300,400,200,350)
    if time.time() - currenttime > 0.2:
        # print(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        op20_night = featrue_detection(frame, op20_night_background, 60, 50, 300, 400, 200, 350)

        if M5000_work_background is not None :
            M5000_work_background_1 = background_diff(M5000_work_background, 0, 100, 600, 910)
            M5000_work_background_2 = background_diff(M5000_work_background, 100, 230, 600, 910)
            M5000_work_1 = diff(frame, M5000_work_background_1, 100000, 0,100,600,910)
            M5000_work_2 = diff(frame, M5000_work_background_2, 100000, 100, 230, 600, 910)
            op20_work = featrue_detection(frame,op20_work_background,60,20,260,360,500,650)
            # print('a1',M5000_work_1)
            # print('a2',M5000_work_2)
            # if (M5000_work_1 is True and M5000_work_2 is True and op20_night is True) or (op20_work is True and op20_night is True):
            #机器人op20在中间位置时的工作状态
            if op20_night is True and (M5000_work_1 is True and M5000_work_2 is True and op20_work is True):
                M5000_work_time.pop(0)
                M5000_work_time.append(1)
                # print(op20_work)
                # print('白天工作')
            # 机器人op20在左边位置时的工作状态
            if op20_night is False and M5000_work_1 is True and M5000_work_2 is True:
                M5000_work_time.pop(0)
                M5000_work_time.append(1)
                # print('夜晚工作')
            #机器停工
            else:
                # print('机器停工')
                M5000_work_time.pop(0)
                M5000_work_time.append(0)

                #夜晚工作的时候切换材料由工人做
                if op20_night is False:
                    if M5000_work_2 is True:
                        houban_material_time.pop(0)
                        houban_material_time.append(1)

                    elif M5000_work_2 is False:
                        houban_material_time.pop(0)
                        houban_material_time.append(0)



                # #拖车换料/黑线换料
                # if houban_huan_backgrond is None:
                #     houban_huan_backgrond = background_diff(houban_huan_backgrond,300,720,400,1280)
                #
                # houban_huan = diff(frame,houban_huan_backgrond,100000,300,720,400,1280)
                #
                # if houban_huan is True:
                #     houban_huan_time.pop(0)
                #     houban_huan_time.append(1)
                #
                # else:
                #     houban_huan_time.pop(0)
                #     houban_huan_time.append(0)
                #
                # houban_huan_time_sum = sum(houban_huan_time)
                # if houban_huan_judge is False and houban_huan_time_sum > 5:
                #     houban_huan_start = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                #
                # if houban_huan_judge is True and houban_huan_time_sum <= 2:
                #     houban_huan_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                #     houban_huan_judge = False



                #修理M5000
                M5000_fix = featrue_detection(frame,M5000_fix_backgrond,60,10,0,65,350,520)
                if M5000_fix is True:
                    M5000_fix_time.pop(0)
                    M5000_fix_time.append(0)

                else:
                    M5000_fix_time.pop(0)
                    M5000_fix_time.append(1)

                M5000_fix_time_sum = sum(M5000_fix_time)
                if M5000_fix_judge is False and M5000_fix_time_sum > 4:
                    M5000_fix_start = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    print('M5000_fix_start',M5000_fix_start)
                    M5000_fix_judge = True

                if M5000_fix_judge is True and M5000_fix_time_sum <= 2:
                    M5000_fix_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                    print('M5000_fix_stop',M5000_fix_stop)
                    M5000_fix_judge = False

                # 调整材料
                if houban_material_backgrond is not None:
                    houban_material_backgrond_lower = background_diff(houban_material_backgrond,  550,720,430,750)
                    houban_material_backgrond_all = background_diff(houban_material_backgrond,190,350,970,1100)

                    houban_material = diff(frame, houban_material_backgrond_lower, 1000000, 550,720,430,750)
                    houban_material_all = diff(frame,houban_material_backgrond_all,100000,190,350,970,1100)
                    # print('shang',houban_material_upper)
                    # print('xia',houban_material)
                    if houban_material is True:
                        houban_material_time.pop(0)
                        houban_material_time.append(1)

                    else:
                        houban_material_time.pop(0)
                        houban_material_time.append(0)
                        if houban_material_all is True:
                            houban_material_all_time.pop(0)
                            houban_material_all_time.append(1)
                        if houban_material_all is False:
                            houban_material_all_time.pop(0)
                            houban_material_all_time.append(0)

                        houban_material_all_sum = sum(houban_material_all_time)

                        if houban_material_all_judge is False and houban_material_all_sum > 6:
                            houban_material_all_start = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                            print('houban_material_all_start', houban_material_all_start)
                            houban_material_all_judge = True

                        if houban_material_all_judge is True and houban_material_all_sum <= 4:
                            houban_material_all_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                            print('houban_material_all_stop', houban_material_all_stop)
                            houban_material_all_judge = False

                    houban_material_time_sum = sum(houban_material_time)
                    if houban_material_judge is False and houban_material_time_sum > 8:
                        houban_material_start = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                        print('houban_material_start',houban_material_start)
                        houban_material_judge = True

                    if houban_material_judge is True and houban_material_time_sum <= 5:
                        houban_material_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                        print('houban_material_stop',houban_material_stop)
                        houban_material_judge = False

                houban_material_backgrond = frame






            M5000_work_time_sum = sum(M5000_work_time)
            if M5000_work_start_judge is False and M5000_work_time_sum > 4:
                M5000_work_start =  datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                M5000_work_start_judge = True
                print('M5000_work_start',M5000_work_start)
            if M5000_work_start_judge is True and M5000_work_time_sum <= 2:
                M5000_work_stop = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                print('M5000_work_stop',M5000_work_stop)
                M5000_work_start_judge = False

        cv2.imshow('frame',frame)
        cv2.waitKey(25)
        M5000_work_background = frame
        currenttime = time.time()



cv2.destroyAllWindows()
cap.release()