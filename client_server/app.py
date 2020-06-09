import os

import cv2
import imutils
import numpy as np
from flask import Flask, redirect, render_template, request, Response
from flask_bootstrap import Bootstrap
# from flask_mail import Mail, Message
from flask_uploads import UploadSet, configure_uploads, extension, patch_request_class, IMAGES
from flask_wtf import FlaskForm
from imutils.object_detection import non_max_suppression
from wtforms import MultipleFileField
from wtforms.validators import InputRequired

from src.detection_api.base.pedestrian_detector import PedestrianDetector, Config

app = Flask(__name__)
app.config['SECRET_KEY'] = "#techup-studio-works@chingy"
app.config.from_pyfile('config.cfg')
Bootstrap(app)
# mail = Mail(app)
VIDEO = tuple('mp4 mkv flv gif mov ogv webm mpg avi'.split())
photos = UploadSet('photos', IMAGES)
videos = UploadSet('videos', VIDEO)
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/static/uploads/photos'
app.config['UPLOADED_VIDEOS_DEST'] = os.getcwd() + '/static/uploads/videos'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
configure_uploads(app, (photos, videos))
patch_request_class(app)

app.jinja_env.globals.update(len=len)


class DefaultConfig(Config):
    faster_rcnn_inception_v2_pb_path = os.path.abspath('../detection_api/models/frozen_inference_graph.pb')
    model_path = os.path.abspath('../detection_api/models/output.svc.model')
    # step_size = (6, 6)
    non_max_suppression_threshold = .2
    accuracy_threshold = .7
    img_resize = 600
    # visualize = True
    verbose = False


class StreamGenerator:
    def __init__(self, media=None):
        self.detector = PedestrianDetector(DefaultConfig())
        self.stream_source = None
        self.active = False
        self.media = media

    def stream(self, wait_time=1):
        if self.active:
            if str(self.media).__contains__('photos\\photo'):
                data = cv2.imread(self.media)
                image = imutils.resize(data, height=600)
                # image = cv2.resize(data_sets,(0,0), None, 1, 1)
                image = self.detect(image)
                frame = cv2.imencode('.jpg', image)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                while self.stream_source.isOpened():
                    ret, data = self.stream_source.read()
                    if not ret:
                        break

                    image = imutils.resize(data, height=600)
                    image = self.detect(image, model='cv2')

                    # image = cv2.resize(data_sets,(0,0), None, 1, 1)
                    frame = cv2.imencode('.jpg', image)[1].tobytes()

                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    cv2.waitKey(wait_time)

                    if not self.active:
                        self.stop()
            if not self.active:
                self.stop()

    def detect(self, frame, model: str='conv'):
        # (rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
        #
        # # Applies non-max supression from imutils package
        # # to kick-off overlapped boxes
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # result = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        if model == 'conv':
            _, boxes = self.detector.detect_with_conv_net(frame)
        elif model == 'hog':
            _, boxes = self.detector.detect_with_sliding_windows(frame)
        else:
            _, boxes = self.detector.detect_with_tp_cv2(frame)

        # draw the final bounding boxes
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

        return frame

    def stop(self):
        if self.is_active():
            if self.stream_source is not None:
                self.stream_source.release()
            self.set_active(False)

    def start(self):
        if not self.is_active() and self.get_media_source() is not None:
            self.stream_source = cv2.VideoCapture(self.media)
            self.set_active(True)

    def set_active(self, state):
        self.active = state

    def is_active(self):
        return self.active

    def set_media_source(self, media):
        self.media = media

    def get_media_source(self):
        return self.media


class UploadForm(FlaskForm):
    media = MultipleFileField('', validators=[InputRequired()])


streamer = StreamGenerator()


@app.route('/')
def home():
    form = UploadForm()
    if streamer.is_active():
        streamer.stop()
    return render_template("index.html", form=form)


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    if 'media' in request.files:
        file = request.files['media']
        # print(file)
        name = file.filename
        ext = extension(name)

        if ext.lower() in IMAGES:
            filename = photos.save(file, name=f'photo.')
            path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
            # print(f'processing image at path {path}')
        elif ext.lower() in VIDEO:
            filename = videos.save(file, name=f'video.')
            path = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
            # print(f'processing video at path {path}')
        else:
            return redirect('/')

        streamer.stop()
        streamer.set_media_source(path)
        return render_template("live.html", title="Detection Results")

    return redirect('/')


@app.route('/live', methods=['GET'])
def live():
    streamer.stop()
    streamer.set_media_source(0)
    return render_template("live.html", title="Live Feed Detection")


@app.route('/docs', methods=['GET'])
def docs():
    if streamer.is_active():
        streamer.stop()
    return render_template("docs.html")


@app.route('/authors', methods=['GET'])
def authors():
    if streamer.is_active():
        streamer.stop()
    return render_template("authors.html")


@app.route('/stream')
def stream():
    if not streamer.is_active() and streamer.get_media_source() is not None:
        streamer.start()
    return Response(streamer.stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=80, threaded=True)
