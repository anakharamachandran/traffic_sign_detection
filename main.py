from flask import Flask
from flask import Flask, render_template,Response,request
import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
import pyttsx3
import classifier as cl
import json


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')


@app.route('/result', methods=['GET','POST'])
def result():
    print("in result")

    if request.method == 'POST':
        print("in post")

        image = request.files["pic"]
        image.save("static/uploads/"+image.filename)
        result = cl.classify(image.filename)
        r = result.index(max(result))
        classes = { 0:'Speed limit (20km/h)',
                1:'Speed limit (30km/h)',
                2:'Speed limit (50km/h)',
                3:'Speed limit (60km/h)',
                4:'Speed limit (70km/h)',
                5:'Speed limit (80km/h)',
                6:'End of speed limit (80km/h)',
                7:'Speed limit (100km/h)',
                8:'Speed limit (120km/h)',
                9:'No passing',
                10:'No passing veh over 3.5 tons',
                11:'Right-of-way at intersection',
                12:'Priority road',
                13:'Yield',
                14:'Stop',
                15:'No vehicles',
                16:'Veh > 3.5 tons prohibited',
                17:'No entry',
                18:'General caution',
                19:'Dangerous curve left',
                20:'Dangerous curve right',
                21:'Double curve',
                22:'Bumpy road',
                23:'Slippery road',
                24:'Road narrows on the right',
                25:'Road work',
                26:'Traffic signals',
                27:'Pedestrians',
                28:'Children crossing',
                29:'Bicycles crossing',
                30:'Beware of ice/snow',
                31:'Wild animals crossing',
                32:'End speed + passing limits',
                33:'Turn right ahead',
                34:'Turn left ahead',
                35:'Ahead only',
                36:'Go straight or right',
                37:'Go straight or left',
                38:'Keep right',
                39:'Keep left',
                40:'Roundabout mandatory',
                41:'End of no passing',
                42:'End no passing veh > 3.5 tons' }
        ans = classes[r]
        engine = pyttsx3.init()
        engine.say(ans)
        engine.runAndWait()
        return render_template('result.html', r = r, imgname = image.filename,result = result, ans = ans)

    else:

        return render_template('result.html', name="shine")

@app.route('/detect')
def detect():

    cv2.startWindowThread()
    cap = cv2.VideoCapture(0)
    num=0
    model = keras.models.load_model('traffic_classifier.h5')
    classes = { 0:'Speed limit (20km/h)',
                1:'Speed limit (30km/h)',
                2:'Speed limit (50km/h)',
                3:'Speed limit (60km/h)',
                4:'Speed limit (70km/h)',
                5:'Speed limit (80km/h)',
                6:'End of speed limit (80km/h)',
                7:'Speed limit (100km/h)',
                8:'Speed limit (120km/h)',
                9:'No passing',
                10:'No passing veh over 3.5 tons',
                11:'Right-of-way at intersection',
                12:'Priority road',
                13:'Yield',
                14:'Stop',
                15:'No vehicles',
                16:'Veh > 3.5 tons prohibited',
                17:'No entry',
                18:'General caution',
                19:'Dangerous curve left',
                20:'Dangerous curve right',
                21:'Double curve',
                22:'Bumpy road',
                23:'Slippery road',
                24:'Road narrows on the right',
                25:'Road work',
                26:'Traffic signals',
                27:'Pedestrians',
                28:'Children crossing',
                29:'Bicycles crossing',
                30:'Beware of ice/snow',
                31:'Wild animals crossing',
                32:'End speed + passing limits',
                33:'Turn right ahead',
                34:'Turn left ahead',
                35:'Ahead only',
                36:'Go straight or right',
                37:'Go straight or left',
                38:'Keep right',
                39:'Keep left',
                40:'Roundabout mandatory',
                41:'End of no passing',
                42:'End no passing veh > 3.5 tons' }
    data=np.ndarray(shape=(1,30,30,3),dtype=np.float32)
    size=(30,30)
    engine = pyttsx3.init()
    while(True):
        # reading the frame
        t="null"
        ret, img = cap.read()
        height, width, channels = img.shape
        scale_value = width / height
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img_array = np.asarray(img_resized)

        normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = img_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = classes[index]
        confidence_score = prediction[0][index]
        #print(class_name)
        if confidence_score * 100>70:

            cv2.putText(img, class_name, (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(img, str(float("{:.2f}".format(confidence_score * 100))) + "%", (75, 100), cv2.FONT_HERSHEY_SIMPLEX,1.5, (0, 255, 0), 2)

        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            image = cap.read()[1]
            cv2.imwrite("C://Users//Admin//Documents//demo//"+str(num)+".png", image)
            print("image saved!")
            img = cv2.imread("C://Users//Admin//Documents//demo//"+str(num)+".png")
            height, width, channels = img.shape
            scale_value = width / height
            img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            img_array = np.asarray(img_resized)

            normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1
            data[0] = img_array
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = classes[index]
            confidence_score = prediction[0][index]
            print(class_name)
            engine.say(class_name)
            engine.runAndWait()

            cv2.imshow('frame', image)

            num += 1

        elif k == ord('q'):
            # breaking the loop if the user types q
            # note that the video window must be highlighted!
            break





        # displaying the frame

        #cv2.rectangle(img, (25, 25), (300, 300), (0, 255, 0), 2)
        cv2.imshow('frame',img)


    cap.release()
    cv2.destroyAllWindows()
    # the following is necessary on the mac,
    # maybe not on other platforms:
    cv2.waitKey(1)