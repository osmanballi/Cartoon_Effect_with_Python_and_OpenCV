# Cartoon Effect with Python and OpenCV

![Orginal Image](https://raw.githubusercontent.com/osmanballi/Cartoon_Effect_with_Python_and_OpenCV/main/spiderman.jpg)
![Cartoon Image](https://raw.githubusercontent.com/osmanballi/Cartoon_Effect_with_Python_and_OpenCV/main/cartoon.jpg)

### Source: 
####http://datahacker.rs/002-opencv-projects-how-to-cartoonize-an-image-with-opencv-in-python/
####https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
####https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html


```python
import cv2
from tkinter.filedialog import * 

photo = askopenfilename()
img = cv2.imread(photo)
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(grey,5)
edges = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,5)
color = cv2.bilateralFilter(img,9,200,200)
cartoon = cv2.bitwise_and(color,color,mask=edges)
# cv2.imshow("cartoon",cartoon)

import numpy as np

def color_quantization(img, k):
    data = np.float32(img).reshape((-1,3))
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TermCriteria_MAX_ITER, 20, 1.0)
    ret, label,center = cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

img1 = color_quantization(cartoon,7)
# cv2.imshow("img1",img1)

blurred = cv2.medianBlur(img1,3)
cartoon1 = cv2.bitwise_and(blurred,blurred,mask=edges)
# cv2.imshow("cartoon1",cartoon1)

# cv2.imshow("color",color)
# cv2.imshow("edges",edges)
# cv2.imshow("median",median)
# cv2.imshow("grey",grey)
# cv2.imshow("spiderman",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("cartoon.jpg",cartoon1)
``` 
