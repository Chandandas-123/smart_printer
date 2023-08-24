# follow chandan das on Linkedin https://www.linkedin.com/in/chandan-das-713617251

import cv2
import os
import time
import numpy as np
import socket
import win32print
import win32ui
from PIL import Image, ImageWin

# for image joining
def vconcat_resize(img_list, interpolation 
                   = cv2.INTER_CUBIC):
      # take minimum width
    w_min = min(img.shape[1] 
                for img in img_list)
      
    # resizing images
    im_list_resize = [cv2.resize(img,
                      (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation = interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)

def print_out():
        
    HORZRES = 8
    VERTRES = 10
    #
    # LOGPIXELS = dots per inch
    #
    LOGPIXELSX = 88
    LOGPIXELSY = 90
    #
    # PHYSICALWIDTH/HEIGHT = total area
    #
    PHYSICALWIDTH = 110
    PHYSICALHEIGHT = 111
    #
    # PHYSICALOFFSETX/Y = left / top margin
    #
    PHYSICALOFFSETX = 112
    PHYSICALOFFSETY = 113

    printer_name = win32print.GetDefaultPrinter ()
    file_name = "final.jpg"

    #
    # You can only write a Device-independent bitmap
    #  directly to a Windows device context; therefore
    #  we need (for ease) to use the Python Imaging
    #  Library to manipulate the image.
    #
    # Create a device context from a named printer
    #  and assess the printable size of the paper.
    #
    hDC = win32ui.CreateDC ()
    hDC.CreatePrinterDC (printer_name)
    printable_area = hDC.GetDeviceCaps (HORZRES), hDC.GetDeviceCaps (VERTRES)
    printer_size = hDC.GetDeviceCaps (PHYSICALWIDTH), hDC.GetDeviceCaps (PHYSICALHEIGHT)
    printer_margins = hDC.GetDeviceCaps (PHYSICALOFFSETX), hDC.GetDeviceCaps (PHYSICALOFFSETY)


    bmp = Image.open (file_name)
    if bmp.size[0] > bmp.size[1]:
        bmp = bmp.rotate (90)

    ratios = [1.0 * printable_area[0] / bmp.size[0], 1.0 * printable_area[1] / bmp.size[1]]
    scale = min (ratios)


    hDC.StartDoc (file_name)
    hDC.StartPage ()

    dib = ImageWin.Dib (bmp)
    scaled_width, scaled_height = [int (scale * i) for i in bmp.size]
    x1 = int ((printer_size[0] - scaled_width) / 2)
    y1 = int ((printer_size[1] - scaled_height) / 2)
    x2 = x1 + scaled_width
    y2 = y1 + scaled_height
    dib.draw (hDC.GetHandleOutput (), (x1, y1, x2, y2))

    hDC.EndPage ()
    hDC.EndDoc ()
    hDC.DeleteDC ()




cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
capture = "stream"
show =  "output"
footer = cv2.imread("footer.png")
cv2.namedWindow(capture)
cv2.moveWindow(capture,350,100)


i=30;
while True:
    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frames = video_capture.read()
        imgcrop = frames
        img2 = imgcrop
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # print(frames.shape)

        # Draw a rectangle around the faces


        for (x, y, w, h) in faces:
            # print("faces ",x," ",y," ",w," ",h)
            cv2.rectangle(frames, (x-100, y-80), (x+w+100, y+h+180), (0, 255,0), 2)


            cv2.putText(frames, str(i//30), (30, 100), cv2.FONT_HERSHEY_SIMPLEX ,3, (255, 0, 0), 5, cv2.LINE_AA)
            i+=1
            # print(x-100," ",y-80," ",(x+w+100)," ",(y+h+180))
            imgcrop = imgcrop[(y-80):(y+h+180),(x-100):(x+w+100)]
            
            
        if len(faces)  < 1 or (x-100) < 52 or (y-80) < 5 or (x+w+100) > 635 or (y+h+180) > 475 :
            i=30
     
        if (imgcrop.shape[0] > 0 and imgcrop.shape[1] > 0) :
            if(i>100):
                cv2.imwrite("save.jpg",imgcrop)
             
                break
        cv2.imshow(capture, frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
            break

        # footer = cv2.resize(footer, (imgcrop.shape[0],imgcrop.shape[1]))

    
    
    video_capture.release()
    cv2.destroyAllWindows()
    img1 = cv2.imread("save.jpg")
    img2 = cv2.imread("footer.png")
    display = vconcat_resize([img1, img2])
    while True:
        
        
        # img1 = np.vstack((img1,footer))
        cv2.namedWindow(show)
        cv2.moveWindow(show,530,120)
        cv2.imshow(show,display)
        # print(img1.shape)
        img = np.hstack((img1,img1))
        img = np.vstack((img,img))
        cv2.imwrite("final.jpg",img)
        # client.close()
        if cv2.waitKey() & 0xFF == ord('y'):
            try :
                print_out()
                cv2.destroyAllWindows()
                break
            except:
                cv2.destroyAllWindows()
                break
        else :
            cv2.destroyAllWindows()
            break
            
        
    
       

            
    





