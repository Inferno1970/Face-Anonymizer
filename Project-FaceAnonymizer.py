import cv2
import mediapipe as mp
import os
import argparse


#Creting different modes of this code
args=argparse.ArgumentParser()
args.add_argument("--mode",default='webcam')
args.add_argument("--filePath",default='D:\Data Science\Python Assignment\Computer Vision\Data\Sample Image\FaceSample.jpg')
args=args.parse_args()




#Detect faces live

#Creating a function for video processing (multiple frames)
def process_image(img,face_detection):
    H,W,_=img.shape
    img_rgb= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out=face_detection.process(img_rgb)
        
    if out.detections is not None:
        for detection in out.detections:
            bbox=detection.location_data.relative_bounding_box
            x1,y1,w,h=bbox.xmin, bbox.ymin, bbox.width, bbox.height

            #The bounding box elemnts are ratio and relative
            x1=int(W*x1)
            y1=int(H*y1)
            w=int(W*w)
            h=int(H*h)


            #Blur faces
            img[y1:y1+h,x1:x1+w,:]=cv2.blur(img[y1:y1+h,x1:x1+w,:],(30,30))

    return img

#Save image directory
output_directory='D:\Data Science\Python Assignment\Computer Vision\Data'

mp_face_detection=mp.solutions.face_detection
with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
    if args.mode in ['image']:
        img=cv2.imread(args.filePath)
        print(img.shape)
        img=process_image(img,face_detection)

        cv2.imwrite(os.path.join(output_directory,'AnanomizedFace.png'),img)



        #Creating directory if not exsits
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
    #if the file is a saved video
    elif args.mode in ['video']:

        vid= cv2.VideoCapture(args.filePath)
        #ret: A boolean value indicating whether the frame was successfully read (True if successful, False if not).
        #frame: The actual frame data.
        ret,frame=vid.read()

        #path,format,frame rate,frame size
        output_video=cv2.VideoWriter(os.path.join(output_directory,'AnanomizedFaceVideo.mp4'),
                                        cv2.VideoWriter_fourcc(*'MP4'),
                                        25,
                                        (frame.shape[1],frame.shape(0)))
        while ret:
            frame=process_image(frame,face_detection)
            #Copying processed frames 
            output_video.write(frame)
            #Check whther ret is true or false whcih affects ths loop
            ret,frame=vid.read()



        vid.release()
        output_video.release()

    elif args.mode in ['webcam']:
        vid= cv2.VideoCapture(0)
        ret,frame=vid.read()

        while ret:
            frame=process_image(frame,face_detection)
            #Copying processed frames 
            cv2.imshow('frame',frame)
            #Waiting 30 seconds or press q to quit
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break  

            #Reading a new frame and check whether ret is true or false whcih affects ths loop
            ret,frame=vid.read()




        vid.release()      





#Detect faces in an image
    
# #Read image
# image_path='D:\Data Science\Python Assignment\Computer Vision\Data\Sample Image\FaceSample.jpg'
# img=cv2.imread(image_path)
# print(img.shape)
# H,W,_=img.shape

# #model selection referes to the range of the object from camera (0,1)
# mp_face_detection=mp.solutions.face_detection
# with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
#     img_rgb= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#     out=face_detection.process(img_rgb)

#     # print(out.detections)

#     #This model only detects human faces
#     if out.detections is not None:
#         for detection in out.detections:
#             bbox=detection.location_data.relative_bounding_box
#             x1,y1,w,h=bbox.xmin, bbox.ymin, bbox.width, bbox.height

#             #The bounding box elemnts are ratio and relative
#             x1=int(W*x1)
#             y1=int(H*y1)
#             w=int(W*w)
#             h=int(H*h)

#             #Drawing green bounding box 
#             # img=cv2.rectangle(img,(x1,y1),(x1+w,y1+h), (0,255,0),10)

#             #Blur faces
#             img[y1:y1+h,x1:x1+w,:]=cv2.blur(img[y1:y1+h,x1:x1+w,:],(30,30))

#     # cv2.imshow('img',img)
#     # cv2.waitKey(0)


# #Save image
# output_directory='D:\Data Science\Python Assignment\Computer Vision\Data\Sample Image'

# #Creating directory if not exsits
# if not os.path.exist(output_directory):
#     os.makedirs(output_directory)


# cv2.imwrite(os.path.join(output_directory,'AnanomizedFace.png'),img)




