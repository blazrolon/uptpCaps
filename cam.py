import cv2
import os

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

path = "./pictures"

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        #img_name = "pictures/whole_frame_{}.png".format(img_counter)
        #cv2.imwrite(img_name, frame)
        #print("{} written!".format(img_name))

        #guardar
        num_elements = len([item for item in os.listdir(path) if os.path.isfile(os.path.join(path, item))])
        file_name = str(num_elements).zfill(5) + '.jpg'

        #Save the picture
        cv2.imwrite(os.path.join(path, file_name), frame)
        print("Saved")

        img_counter += 1
cam.release()

cv2.destroyAllWindows()
