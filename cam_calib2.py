import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.

imgpoints = [] # 2d points in image plane.
images = glob.glob("/home/pi/opencv-2.4.10/samples/python/new_images/*.jpg")
chrow=7
chcol=10
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret = False
    # Find the chess board c  orners
    ret, corners = cv2.findChessboardCorners(gray, (chrow,chcol))
    print ret, len(corners)
    # If found, add object points, image points (after refining them)
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners 
        cv2.drawChessboardCorners(img, (chrow,chcol), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(0)

cv2.waitKey(0)
for i in range (1,5):
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

imgpoints = np.array(imgpoints,'float32')

pattern_size = (chrow, chcol)
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2).astype(np.float32)
pattern_points = np.array(pattern_points,dtype=np.float32)
print len(corners), len(pattern_points)

ret, matrix, dist_coef, rvecs, tvecs = cv2.calibrateCamera([pattern_points], [corners], gray.shape[::-1], flags=cv2.CALIB_USE_INTRINSIC_GUESS)

img5 = cv2.imread('/home/pi/opencv-2.4.10/samples/python/new_images/chess07.jpg')
h,  w = img5.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(matrix,dist_coef,(w,h),1,(w,h))
print roi

################## Method 1 to Undistort ############################
# undistort
#dst = cv2.undistort(img5, matrix, dist_coef, None, newcameramtx)

# crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult.png',dst)
#cv2.imshow('calibresult',dst)
#cv2.waitKey(0)

################## Method 1 to Undistort ############################
# undistort
mapx,mapy = cv2.initUndistortRectifyMap(matrix,dist_coef,None,newcameramtx,(w,h),5)
dst = cv2.remap(img5,mapx,mapy,cv2.INTER_LINEAR)
#cv2.imwrite('calibresult.png',dst)
cv2.imshow('calibresult',dst)
cv2.waitKey(0)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
