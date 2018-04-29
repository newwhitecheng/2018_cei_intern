#coding:utf-8
usage_text = """
LPIRC Client 
This is a sample client for illustration purpose only.
This sample client is written in python but a client program can be written in any language, as long as it can communicate using HTTP GET and POST.

====================
@2015 - HELPS, Purdue University

Main Tasks:
-----------
1. Login to the Server and store the token from the server.
2. Request images from the server and store the images in a directory.
3. Send POST messages to the server as the results.
4. Request total number of valid images from server.

Rules:
1. If a single image has multiple bounding boxes, the client can send the bounding boxes in the same POST message.
2. The client may send multiple POST messages for different bounding boxes of the same image.
3. Two different images need to be sent in the different POST messages.
4. The POST messages for different images may be out of order (for example, the bounding boxes for image 5 may be sent before the bounding boxes for image 3)


Steps to Follow to Run the client script:
1. Download client.py and golden_output.csv files from https://github.com/ieeelpirc/sampleclient
2. Keep both the files in the same directory.
3. Run the command:

   >python client.py -w 128.46.75.108 --im_dir images --temp_dir temp
    
   This command will start the script by connecting it to the server hosted by lpirc.ecn.purdue.edu
   The script will create two new directories, "temp" and "images", in the same directory as client.py, if they are not already present.
   All the images received from the server will be stored in the "images" directory.

   If testing on a local server, replace the ip with the ip of your local server.  You may also need to use the -p/--port flag to appropriately set the port.  

Requirements:
-------------
1. Python v2.7.3
2. PycURL (To support HTTP)

Usage:
------
client.py [OPTION]...
Options:
         -w, --ip
                IP address of the server in format <xxx.xxx.xxx.xxx>
                Default: 128.46.75.108

         -p, --port
                Port number of the server.
                Default: 80

         --user
                Username
                The user must send a mail to lpirc@ecn.purdue.edu to request 
        for a username and password.

         --pass
                Password for the username
                The user must send a mail to lpirc@ecn.purdue.edu to request 
        for a username and password.
        
         --im_dir
        Directory with respect to the client.py 
                where received images are stored
        Default: ../images

     --temp_dir
        Directory with respect to the client.py
        where temporary data is stored
        Default: ../temp

         --in
        Name of the csv file to take the input with respect to source directory
        Default: golden_output.csv
        (This is for testing purpose only. It should not be in the real client.)

         --score
        Score that you want to have. The client corrupts 
        the golden input with probability (100 - score)/100.
        Default: 100
        (This is for testing purpose only. It should not be in the real client.)

         -h, --help
                Displays all the available option


"""
import detection
from multiprocessing import Process,Pool,Queue,Manager,Lock,Value
import traceback
import time
import sys
import threading
import os
###import caffe 

import numpy as np 
import thread
import zipfile
## <-| add by Alan YU

from random import randint
import pycurl
import csv,shutil,os
from collections import defaultdict
import getopt,sys, time

try:
    # python 3
    from urllib.parse import urlencode
except ImportError:
    # python 2
    from urllib import urlencode

from StringIO import StringIO as BytesIO
from StringIO import StringIO

last_down_w = -1
finished = False


def detection(w):
    # TODO  implement coding here
    detection(w)
    print 'handling %d.jpg'%w

def producer(q1,q,namespace):
    global token
    global last_down_w
    global finished

    print os.getpid(), 'producer start'
    while True:
        try:
            while q.full(): time.sleep(0.5)

            if q1.qsize == 0: break
            w = q1.get(False)
            print 'get w',w
            if not get_images(token,w):
                q1.put(w,False)
            else:
                unzip_images(w)
                for i in xrange(100):
                    q.put(i+w,False)
                    #print 'put',i+w
                if w == last_down_w:
                    finished = True

        except:
            traceback.print_exc()
            break
    print os.getpid(), 'producer done'


def consumer(q1, q,namespace):
    global finished
    global token

    print os.getpid(), 'consumer start'
    while True:
        try:
            if finished: break
            if q.qsize() == 0: 
               time.sleep(2.0)
               continue
            w = q.get(False)
            try:
                detection(w)
                res = csv_filename
                post_result(token,res)

            except:
                traceback.print_exc()
                q.put(w,False)

        except:
            traceback.print_exc()
    print os.getpid(), 'consumer done'


def multi_run(producer,consumer,pool_cnt=20,producer_cnt=10,consumer_cnt=10):
    global token
    global last_down_w
    global finished

    manager = Manager()
    q1 = manager.Queue(10000)
    ws = [i for i in range(1, int(no_of_images)+1, 100)]
    for i in ws: q1.put(i)
    last_down_w = ws[-1]

    q = manager.Queue(10000)
    p = Pool(pool_cnt)
    nm = manager.Namespace()
    nm.running = True
    for i in xrange(producer_cnt):
        pw = p.apply_async(producer,args=(q1,q,nm,))
        print 'new producer'
        sys.stdout.flush()
        time.sleep(0.1)
    for i in xrange(consumer_cnt):
        p.apply_async(consumer,args=(q1,q,nm,))
        print 'new consumer'
        sys.stdout.flush()
    p.close()
    p.join()


## |-> add by Alan YU

## |-> add by Alan YU ########
# Define a function for the thread of download the images.zip
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def unzip_images(w):
     zip_ref = zipfile.ZipFile(image_directory+'/'+str(w)+'.zip', 'r')
     zip_ref.extractall(image_directory+'/'+str(w))
     zip_ref.close()

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#++++++++++++++++++++++++++++ Test_get_token: Can be used by the participant directly ++++++++++++++++++++
# 
# Functionality : 
# Sends request to the server to log in with username and password and returns the token and status. 
# Token needs to be used in all the communication with the server in the session.
# If the username and password are invalid or the session has expired, status returned is 0.
# If the token is is valid, status returned is 1.
# 
# This must be the first message to the server.
#
# Usage: [token, status] = get_token(username, password)
# 
# Inputs: 
#         1. username
#         2. password
#
# Outputs:
#         1. token
#     2. status
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def Test(img):
      
    net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network 
       
    #图片预处理设置 
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28) 
    transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28) 
    #transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用 
    transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间 
    transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR 
       
    im=caffe.io.load_image(img)                   #加载图片 
    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中 
       
    #执行测试 
    out = net.forward() 
       
    labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件 
    prob= net.blobs['prob'].data[0].flatten() #取出最后一层（prob）属于某个类别的概率值，并打印,'prob'为最后一层的名称
    print prob 
    order=prob.argsort()[4]  #将概率值排序，取出最大值所在的序号 ,9指的是分为0-9十类 
    #argsort()函数是从小到大排列 
    print 'the class is:',labels[order]   #将该序号转换成对应的类别名称，并打印 
    f=file("/home/liuyun/caffe/examples/DR_grade/label.txt","a+")
    f.writelines(img+' '+labels[order]+'\n')
 


#++++++++++++++++++++++++++++ get_token: Can be used by the participant directly ++++++++++++++++++++
# 
# Functionality : 
# Sends request to the server to log in with username and password and returns the token and status. 
# Token needs to be used in all the communication with the server in the session.
# If the username and password are invalid or the session has expired, status returned is 0.
# If the token is is valid, status returned is 1.
# 
# This must be the first message to the server.
#
# Usage: [token, status] = get_token(username, password)
# 
# Inputs: 
#         1. username
#         2. password
#
# Outputs:
#         1. token
#     2. status
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_token (username,password):

    buffer = StringIO()
    c = pycurl.Curl()
    c.setopt(c.URL, host_ipaddress+':'+host_port+'/login')
    post_data = {'username':username,'password':password}
    postfields = urlencode(post_data)
    c.setopt(c.POSTFIELDS,postfields)
    c.setopt(c.WRITEFUNCTION, buffer.write)
    c.perform()
    status = c.getinfo(pycurl.HTTP_CODE)
    c.close()
    if status == 200:
        return [buffer.getvalue(),1]
    else:
    #   print status
        print "Unauthorised Access\n"
        return [buffer.getvalue(),0]


#++++++++++++++++++++++++++++ get_image: Can be used by the participant directly ++++++++++++++++++++++++++++++++++
# 
# Functionality : 
# Sends request to the server for an image with its token number and the image number.
# 'status' is 1 if the image transfer succeeded. If the transfer failed, 'status' will be set to 0. 
# Transfer can fail because of two reasons:-
# 1. The image_number request is out of the valid range [1,total_image_number] (inclusive)
# 2. The token is not valid.
#
# Usage: status = get_image(token, image_number) 
# Total number of images can be queried from server using get_no_of_images(token).
# If image number is outside the permitted range, penalty will be assigned. 
#
# Inputs: 
#         1. token : Obtained from Log in ( get_token() )
#         2. image_number : Index of image client needs.
#
# Output:
#     1. status  
#
# Note:
#         The image is buffered in the temp_directory. If the POST message succeeds, 
#         the file is moved to the image_directory. 
#     This movement can be avoided by buffering the file to image_directory and removing the same if HTTP status is not 200 (OK). 
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_image(token, image_number):
    global image_directory
    global temp_directory
    c = pycurl.Curl()
    c.setopt(c.URL, host_ipaddress+':'+host_port+'/image')#/?image='+str(image_number))
    post_data = {'token':token, 'image_name':str(image_number)}
    postfields = urlencode(post_data)
    c.setopt(c.POSTFIELDS,postfields)
    try:
            os.stat(temp_directory)
    except:
            os.mkdir(temp_directory)
    try:
            os.stat(image_directory)
    except:
            os.mkdir(image_directory)
    # Image will be saved as a file
    with open(temp_directory+'/'+str(image_number)+'.jpg', 'wb') as f:
        c.setopt(c.WRITEDATA, f)
        c.perform()
        status = c.getinfo(pycurl.HTTP_CODE)
        c.close()
    if status == 200:
        # Server replied OK so, copy the image from temp_directory to image_directory
        shutil.move(temp_directory+'/'+str(image_number)+'.jpg',image_directory+'/'+str(image_number)+'.jpg')
        return 1
    elif status == 401:
        # Server replied 401, Unauthorized Access, remove the temporary file
        os.remove(temp_directory+'/'+str(image_number)+'.jpg')
        print "Invalid Token\n"
        return 0
    else:
        # Server replied 406, Not Acceptable, remove the temporary file
        os.remove(temp_directory+'/'+str(image_number)+'.jpg')
        print "The image number is not Acceptable\n" 
        return 0

#++++++++++++++++++++++++++++ get_images: Can be used by the participant directly ++++++++++++++++++++++++++++++++++
# 
# Functionality : 
# Sends request to the server for a zip file with 100 images with its token number and the starting image number.
# 'status' is 1 if the image transfer succeeded. If the transfer failed, 'status' will be set to 0. 
# Transfer can fail because of two reasons:-
# 1. The image_number requested is out of the valid range [1,total_image_number] (inclusive)
# 2. The image_number requested is not of the form 1, 101, 201, etc.
# 3. The token is not valid.
#
# Usage: status = get_images(token, image_number) 
# Total number of images can be queried from server using get_no_of_images(token).
# If image number is outside the permitted range, penalty will be assigned. 
#
# Inputs: 
#         1. token : Obtained from Log in ( get_token() )
#         2. image_number : Index of image client needs.
#
# Output:
#     1. status  
#
# Note:
#         The image is buffered in the temp_directory. If the POST message succeeds, 
#         the file is moved to the image_directory. 
#     This movement can be avoided by buffering the file to image_directory and removing the same if HTTP status is not 200 (OK). 
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_images(token, image_number):
    global image_directory
    global temp_directory
    c = pycurl.Curl()
    c.setopt(c.URL, host_ipaddress+':'+host_port+'/zipimages')#/?image='+str(image_number))
    post_data = {'token':token, 'image_name':str(image_number)}
    postfields = urlencode(post_data)
    c.setopt(c.POSTFIELDS,postfields)
    try:
            os.stat(temp_directory)
    except:
            os.mkdir(temp_directory)
    try:
            os.stat(image_directory)
    except:
            os.mkdir(image_directory)
    # Zip file will be saved
    with open(temp_directory+'/'+str(image_number)+'.zip', 'wb') as f:
        c.setopt(c.WRITEDATA, f)
        c.perform()
        status = c.getinfo(pycurl.HTTP_CODE)
        c.close()
    if status == 200:
        # Server replied OK so, copy the zip from temp_directory to image_directory
        shutil.move(temp_directory+'/'+str(image_number)+'.zip',image_directory+'/'+str(image_number)+'.zip')
        return 1
    elif status == 401:
        # Server replied 401, Unauthorized Access, remove the temporary file
        os.remove(temp_directory+'/'+str(image_number)+'.zip')
        print "Invalid Token\n"
        return 0
    else:
        # Server replied 406, Not Acceptable, remove the temporary file
        os.remove(temp_directory+'/'+str(image_number)+'.zip')
        print "The image number is not Acceptable\n" 
        return 0


#++++++++++++++++++++++++++++ get_camera_image: Can be used by the participant directly ++++++++++++++++++++++++++++++++++
# 
# Functionality : 
# Sends request to the server for a zip file with 100 images with its token number and the starting image number.
# 'status' is 1 if the image transfer succeeded. If the transfer failed, 'status' will be set to 0. 
# Transfer can fail because of two reasons:-
# 1. The image_number requested is out of the valid range [1,total_image_number] (inclusive)
# 2. The image_number requested is not of the form 1, 101, 201, etc.
# 3. The token is not valid.
#
# Usage: status = get_camera_image(token, image_number) 
# Total number of images can be queried from server using get_no_of_images(token).
# If image number is outside the permitted range, penalty will be assigned. 
#
# Inputs: 
#         1. token : Obtained from Log in ( get_token() )
#         2. image_number : Index of image client needs.
#
# Output:
#     1. status  
# 
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_camera_image(token, image_number):
    #global image_directory
    #global temp_directory
    buffer = StringIO ()
    c = pycurl.Curl()
    c.setopt(c.URL, host_ipaddress+':'+host_port+'/image_camera')#/?image='+str(image_number))
    post_data = {'token':token, 'image_name':str(image_number)}
    postfields = urlencode(post_data)
    c.setopt(c.POSTFIELDS,postfields)
    # try:
            # os.stat(temp_directory)
    # except:
            # os.mkdir(temp_directory)
    # try:
            # os.stat(image_directory)
    # except:
            # os.mkdir(image_directory)

    c.setopt(c.WRITEFUNCTION, buffer.write)
    c.perform()
    status = c.getinfo(pycurl.HTTP_CODE)
    c.close()
    if status == 200:
        # Server replied OK so, copy the image from temp_directory to image_directory
        print buffer.getvalue ()
        return 1
    elif status == 401:
        # Server replied 401, Unauthorized Access, remove the temporary file
        print "Invalid Token\n"
        return 0
    else:
        # Server replied 406, Not Acceptable, remove the temporary file
        print "The image number is not Acceptable\n" 
        return 0

#++++++++++++++++++++++++++++ post_result: Can be used by the participant directly ++++++++++++++++++++++++++++++++++
# 
# Functionality : 
# POSTS the bounding box information corresponding to an image back to the server. If the POST message to the server 
# succeeded, status = 1. If the POST message to the server failed, the status is set as 0.
# The POST message can fail because of 2 reasons:-
# 1. The token is not valid
# 2. The format of 'data' is incorrect
#
# Usage: post_result(token, data)
# 
# Inputs:
#         1. token: Obtained from Log in
#         2. data :
#           data is a dictionary container with:
#           key:     'image_name', 'CLASS_ID', 'confidence', 'ymax', 'xmax', 'xmin', 'ymin'
#           values:  list of values corresponding to the keys
#
#           Eg: data for 2 bounding boxes of images 1 could be:-
#
#           data = {'image_name': ['1', '1'], 'CLASS_ID': ['58', '10'],'confidence': ['0.529047', '0.184961'],
#               'ymax': ['271.055408', '225.339863'],  'xmax': ['351.519712', '194.408771'],
#               'xmin': ['291.439033', '184.804591'], 'ymin': ['237.148035', '212.047943']}
#                   
# Outputs:
#     1. status
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def post_result(token, data):
    c = pycurl.Curl()
    c.setopt(c.URL, host_ipaddress+':'+host_port+'/result')
    post_data = {'token':token}
    postfields = urlencode(post_data)+'&'+urlencode(data,True)
    c.setopt(c.POSTFIELDS,postfields)
    c.perform()
    status = c.getinfo(pycurl.HTTP_CODE)
    c.close()
    if status == 200:
        # Server replied 200, OK, Result stored
        return 1
    elif (status == 401):
        # Server replied 401, Unauthorized Access
        print "Unauthorized Access\n"
        return 0
    else:
        # Server replied 406, In correct format of 'data'
        print "Not Acceptable. Incorrect Format of result data\n"
        return 0

def post_logout(token):
    c = pycurl.Curl()
    c.setopt(c.URL, host_ipaddress+':'+host_port+'/logout')
    post_data = {'token':token}
    postfields = urlencode(post_data)
    c.setopt(c.POSTFIELDS,postfields)
    c.perform()
    status = c.getinfo(pycurl.HTTP_CODE)
    c.close()
    if status == 200:
        # Server replied 200, OK, Result stored
        return 1
    else:
        # Server replied 401, Unauthorized Access
        print "Unauthorized Access\n"
        return 0

#++++++++++++++++++++++++++++ get number of images: Can be used by the participant directly ++++++++++++++++++++++++++++++++++
# Functionality : 
# POSTS the message to the server and gets back the total number of images.
# If the server sends back OK status (200), status=1 and 'number_of_images' is valid
# If the server sends Unauthorized Access (401), status=0 and 'number_of_images' is invalid.
# 
# Usage: no_of_images = get_no_of_images(token)
# 
# Inputs:
#         1. token: Obtained from Log in
#
# Output:
#         1. number_of_images
#         2. status
#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_no_of_images(token):
    buffer = StringIO()
    c = pycurl.Curl()
    c.setopt(c.URL, host_ipaddress+':'+host_port+'/no_of_images')
    post_data = {'token':token}
    postfields = urlencode(post_data)
    c.setopt(c.POSTFIELDS,postfields)
    c.setopt(c.WRITEFUNCTION, buffer.write)
    c.perform()
    status = c.getinfo(pycurl.HTTP_CODE)
    c.close()
    if status == 200:
        return [buffer.getvalue(), 1]
    else:
        return [buffer.getvalue(), 0]



# The following functions are for testing purpose only. They should not be in the actual client.

#++++++++++++++++++++++++++++ get_lines: Internal Function ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#
# Functionality : 
# Pops bounding box lines from the directory and returns 
# 
# Usage: get_lines(no_of_lines)
# 
# Inputs: 
#         1. no_of_lines: Number of lines to pop and return
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#def get_lines (no_of_lines):
#    global score
#    global level
#    if (level+no_of_lines>len(columns[0]) and level<len(columns[0])):
#        no_of_lines = len(columns[0])-level
#        lines = {'image_name':columns[0][level:level+no_of_lines],\
#                'CLASS_ID':columns[1][level:level+no_of_lines],\
#                'confidence':columns[2][level:level+no_of_lines],\
#                'xmin':columns[3][level:level+no_of_lines],\
#                'ymin':columns[4][level:level+no_of_lines],\
#                'xmax':columns[5][level:level+no_of_lines],\
#                'ymax':columns[6][level:level+no_of_lines]\
#            }
#        level = len(columns[0])
#    elif (level+no_of_lines<=len(columns[0])):
#        lines = {'image_name':columns[0][level:level+no_of_lines],\
#                'CLASS_ID':columns[1][level:level+no_of_lines],\
#                'confidence':columns[2][level:level+no_of_lines],\
#                'xmin':columns[3][level:level+no_of_lines],\
#                'ymin':columns[4][level:level+no_of_lines],\
#                'xmax':columns[5][level:level+no_of_lines],\
#                'ymax':columns[6][level:level+no_of_lines]\
#            }
#        level = level + no_of_lines
#
#    return lines
#
#
#++++++++++++++++++++++++++++ read_csv: Internal Function ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This function simulates a recognition program. It should be replaced.  
#
# Functionality : 
# 
# Reads a file with the results in the form separated with a space and generates database.
# Format of the file:-
# 
# <image_name> <CLASS_ID> <confidence> <xmin> <ymin> <xmax> <ymax>
# <image_name> <CLASS_ID> <confidence> <xmin> <ymin> <xmax> <ymax>
# ...
# ...
# 
# Usage: get_lines(no_of_lines)
# 
# Inputs: 
#         1. no_of_lines: Number of lines to pop and return
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#def read_csv(csv_filename):
#    global csv_data
#    with open(csv_filename) as csvfile:
#        databuf = csv.reader(csvfile, delimiter=' ')
#        for row in databuf:
#            for (i,v) in enumerate(row):
#                columns[i].append(v)
#    level = len(columns[0])
#
#
#++++++++++++++++++++++++++++ simulate_score: Internal Function ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# This function simulates recognition of different scores.
#
# Functionality : 
# Corrupts the database: It changes the CLASS_ID field of a line randomly picked with probability (1 - score/100)
# 
# Usage: simulate_score(score)
# 
# Inputs: 
#         1. score: Score [0,100] which needs to be obtained.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#def simulate_score (score):
#    no_of_lines = len(columns[0])
#    global level
#    global lines
#    for w in range(0,no_of_lines):
#        rand = randint(1,100)
#        if (rand >= score):
#            columns[1][w]=str(int(columns[1][w])+5) # adding class number by 5 to corrupt the line
#
#
#+++++++++++++++++++++++++++ Script usage function +++++++++++++++++++++++++++++++++++++++++++++++++++
def usage():
    print usage_text

#++++++++++++++++++++ Main function to parse command-line input and run server ++++++++++++++++++++++++++++
def parse_cmd_line():

    global host_ipaddress
    global host_port
    global score
    global username
    global password
    global csv_filename
    global image_directory
    global temp_directory
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hw:p:", ["help", "ip=", "port=", "user=", "pass=", "in=", "im_dir=","temp_dir=","score="])
    except getopt.GetoptError as err:
        print str(err) 
        usage()
        sys.exit(2)
    for switch, val in opts:
        if switch in ("-h", "--help"):
            usage()
            sys.exit()
        elif switch in ("-w", "--ip"):
            host_ipaddress = val
        elif switch in ("-p", "--port"):
            host_port = val
        elif switch == "--user":
            username = val
        elif switch == "--pass":
            password = val
        elif switch in ("-i","--in"):
            csv_filename = val
        elif switch == "--im_dir":
            image_directory = val
        elif switch == "--temp_dir":
            temp_directory = val
        elif switch in ("-s","--score"):
            score = int(val)
        else:
            assert False, "unhandled option"

    print "\nhost = "+host_ipaddress+":"+host_port+"\nUsername = "+username+"\nPassword = "+password+"" 


## |-> add by Alan YU ########
# Define a function for the thread of download the images.zip
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#def unzip_images(w):
#     zip_ref = zipfile.ZipFile(temp_directory+"/"+str(w)+'.zip', 'r')
#     zip_ref.extractall(image_directory)
#     zip_ref.close()
#          
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## <-| add by Alan YU ########


#+++++++++++++++++++++++++++ Global Variables ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def test_and_post(w):
    '''
    解压图片，然后分析图片，然后上传结果
    '''
    unzip_images(w)
    for i in xrange(100):
        result =  test_reference(w+i)
        post_results(token,result)

def do_work(threadno,ws):
    print 'thread-%d begin work'%threadno
    while True:
        not_downloaded = []
        for w in ws:
            if not (get_images(token,w)):
                print 'get images %d.zip faild'%w
                #sys.exit()
                not_downloaded.append(w)
            else:
                thread.start_new_thread(test_and_post,(w,))
        if not not_downloaded: break
        ws = not_downloaded
    print 'thread-%d end work'%threadno

#+++++++++++++++++++++++++++ Start of the script +++++++++++++++++++++++++++++++++++++++++++++++

host_ipaddress = '128.46.75.27'
host_port = '8000'
password = 'sCUAtsDC4ehU'
score = 100
username = 'hsc38'
csv_filename = 'golden_output.csv'
image_directory = './images'
temp_directory = './temp'

'''

level = 0
columns = defaultdict(list)
lines=""

## |-> add by Alan YU
root='/home/liuyun/caffe/'   #根目录 
deploy=root + 'examples/DR_grade/deploy.prototxt'    #deploy文件 
caffe_model=root + 'models/DR/model1/DRnet_iter_40000.caffemodel'  #训练好的 caffemodel 

dir = root+'examples/DR_grade/test_512/'
filelist=[]
filenames = os.listdir(dir)
for fn in filenames:
   fullfilename = os.path.join(dir,fn)
   filelist.append(fullfilename)

labels_filename = root +'examples/DR_grade/DR.txt'    #类别名称文件，将数字标签转换回类别名称 

'''

imgs = True   #False     # Set to False if you want to use get_image
camera_imgs = False # Set to False if not using camera.
parse_cmd_line()
[token, status] = get_token(username,password)   # Login to server and obtain token

if status==0:
    print "Incorrect Username and Password. Bye!"
    sys.exit()
#read_csv(csv_filename)                    # Read the csv file to obtain the data    
#simulate_score(score)                     # Corrupt the databaseread to obtain a score of 'score'
[no_of_images, status] = get_no_of_images(token)

if status==0:
    print "Token, Incorrect or Expired. Bye!"
    sys.exit()     

print 'no_of_images', no_of_images
sys.stdout.flush()

# This is for illustration purpose
if camera_imgs:
    for w in range (1, int(no_of_images)+1, 1):
        if get_camera_image (token, w) == 0:
            print "Get Images Failed, Exiting, Bye!"
            sys.exit ()
        else:
            time.sleep(5)
        line = get_lines(1)
        if post_result(token,line)==0:        # If post_result failed, exit.
            print "Posting Result Failed, Exiting, Bye!"
            sys.exit()

        print
else:
    if imgs:
        multi_run(producer,consumer,pool_cnt=20,producer_cnt=1,consumer_cnt=10)
    else:
        for w in range (1,int(no_of_images)+1,1):
            if get_image(token,w)==0:             # If get_image failed, exit.
                print "Get Image Failed, Exiting, Bye!"
                sys.exit()
            else:
                print "Image Stored in client directory "+image_directory+"/"+str(w)+".jpg"
                time.sleep(2)
            line = get_lines(1)
            if post_result(token,line)==0:        # If post_result failed, exit.
                print "Posting Result Failed, Exiting, Bye!"
                sys.exit()

            print

post_logout (token)
