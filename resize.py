import  cv2
img=cv2.imread(filename='C:\\Users\\jiguoqiang\\Desktop\\1.jpg')

img=cv2.resize(img,dsize=(358,441))
cv2.imwrite("1.jpg",img)