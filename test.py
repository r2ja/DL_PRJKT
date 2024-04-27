import cv2
import dlib

#pretrained model
face_detector = dlib.get_frontal_face_detector()

#loading the pretrained predictor
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detect_faces_and_keypoints(image):
  #converts images to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #in-built function to detect faces
  faces = face_detector(gray, 1) 

  #list declarations
  keypoints_list = []
  bounding_boxes = []

  
  for face in faces:
    
    #variables for the bounding box
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    #creating a boundary box list
    bounding_box = (x1, y1, x2, y2)
    bounding_boxes.append(bounding_box)

    #landmark predictor function to get keypoints
    landmarks = landmark_predictor(gray, face)

    
    keypoints = [(p.x, p.y) for p in landmarks.parts()]
    keypoints_list.append(keypoints)
    print(f"Keypoints for this face: {keypoints}")  # Print keypoints for verification

    #drawing keypoints on image.
    for (x, y) in keypoints:
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)  # Larger radius, red color, filled circle
        
  return keypoints_list, bounding_boxes

image_path = 'test_img2.jpg'
image = cv2.imread(image_path)


keypoints_list, bounding_boxes = detect_faces_and_keypoints(image.copy())

#draw bounding boxes
for box in bounding_boxes:
  x1, y1, x2, y2 = box
  cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#display image.
cv2.imshow('Image with Bounding Boxes and Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
