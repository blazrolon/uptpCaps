
import cv2
import get_latest

img=cv2.imread(get_latest.get_last_image(),1)

sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "ESPCN_x4.pb"
sr.readModel(path)
sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio
result = sr.upsample(img) # upscale the input image
cv2.imwrite("output.jpeg", result)