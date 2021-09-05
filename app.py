import cv2
import pose_estimation_class as pm
import mediapipe as mp
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods = ['GET', 'POST'])
def after():
    try:
        img = request.files['file1']
        img.save('static/file.jpg')
        
        img1 = cv2.imread('static/file.jpg')

        detector = pm.PoseDetector()

        img1, p_landmarks, p_connections = detector.findPose(img1, False)

        mp.solutions.drawing_utils.draw_landmarks(img1, p_landmarks, p_connections)

        cv2.imwrite('static/after.jpg', img1)

        return render_template('estimated.html')
    except:
        return  render_template('error.html')

if __name__ == "__main__":
    app.run(host = '0.0.0.0', port=5000, debug=True)







