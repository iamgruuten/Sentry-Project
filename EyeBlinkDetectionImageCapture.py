import cv2

# initialize face and eye cascade xml of opencv library to detect face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

first_read = False #initially is True

# Video Capturing by opening web-cam
cap = cv2.VideoCapture(0)

# to check for first instance of capturing it will return True and image
ret, image = cap.read()

count = 0

while ret:
    # this will keep the web-cam running and capturing the image for every loop
    ret, image = cap.read()
    
    # Convert the recorded image to grayscale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Applying filters to remove impurities
    gray_scale = cv2.bilateralFilter(gray_scale, 5, 1, 1)
    
    # to detect face and eye
    faces = face_cascade.detectMultiScale(gray_scale, 1.3, 5, minSize=(200, 200))

    # If face is detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:              
            cv2.imshow('image',image)
            cv2.waitKey(1)
            print("Blink Detected.....!!!!")
            print("Image " + str(count) + "saved")
            file="E:\Windows Computer Files\justi\Documents\Visual Studio Code files\Sentry\EyeBlinkDetector_imageCapture\SavedImages" + str(count) + ".jpg"
            cv2.imwrite(file, image)
            count +=1
            
    # If no face detected        
    else:
        cv2.putText(image, "No Face Detected.", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
    #Opens a window called image        
    cv2.imshow('image', image)

    #Wait for keystroke
    a = cv2.waitKey(1)
    
    # press q to Quit
    # ord(ch) returns the ascii of ch
    if a == ord('q'):
        break

# release the web-cam
cap.release()
# close the window
cv2.destroyAllWindows()
