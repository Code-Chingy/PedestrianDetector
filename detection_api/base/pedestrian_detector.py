import logging
import os
import random
import sys
import time

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from src.detection_api.base.rcnn_pedestrian_detector import RCNNPedestrianDetector
from src.scripts.extract_persons_annotations import rects_from_image
from src.scripts.image_cropper import scale_image_section

# Main Logger Configurations
# logging.basicConfig(format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
#                     level=logging.DEBUG,
#                     filename="logs.txt"
#                     )

# basicConfg   datefmt="%d-%m-%Y %H:%M:%S"
# logging.getLogger(__name__)

logging.basicConfig(format="%(asctime)s %(levelname)-8s [%(lineno)d] %(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S",
                    level=logging.DEBUG)
# logging.disable(logging.DEBUG)
logger = logging.getLogger(__name__)


class Config:
    # model path configs
    faster_rcnn_inception_v2_pb_path: str = None
    model_path: str = ''
    class_dict: dict = {}

    # sliding windows configs
    step_size: tuple = (10, 10)
    min_wdw_sz: tuple = (64, 128)

    # hog feature extraction configs
    pixels_per_cell: tuple = (6, 6)
    cells_per_block: tuple = (2, 2)
    orientations: int = 9

    # constants
    max_hard_negatives: int = 20000
    max_sample: int = None
    img_resize: int = 700

    # non maximum suppression config
    non_max_suppression_threshold: float = .3
    # prediction result configs
    accuracy_threshold: float = .5
    # gaussian pyramid configs
    downscale: float = 1.6

    # switch configs
    visualize: bool = False
    normalize: bool = True
    verbose: bool = False


class Model:
    def fit(self, samples, labels):
        pass

    def predict(self, feat):
        pass

    def decision_function(self, feat):
        pass

    def get_model(self):
        pass


class ModelWrapper(Model):
    def __init__(self, model):
        self.model = model

    def fit(self, samples, labels):
        return self.model.fit(samples, labels)

    def predict(self, feat):
        return self.model.predict(feat)

    def decision_function(self, feat):
        return self.model.decision_function(feat)

    def get_model(self):
        return self.model


class SVMModelWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def fit(self, samples, labels):
        return self.get_model().train(samples, cv2.ml.ROW_SAMPLE, labels)


class SVMModel(SVMModelWrapper):
    def __init__(self):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)  # (cv2.ml.SVM_RBF)
        # svm.setDegree(0.0)
        svm.setGamma(5.383)
        # svm.setCoef0(0.0)
        svm.setC(2.67)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)
        super().__init__(svm)


class LinearSVCModel(ModelWrapper):
    def __init__(self):
        model = svm.LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose=1)
        super().__init__(model)


class LogisticRegressionModel(ModelWrapper):
    def __init__(self):
        model = LogisticRegression()
        super().__init__(model)

    def fit(self, samples, labels):
        samples = StandardScaler().fit_transform(samples)
        return self.get_model().fit(samples, labels)

    def predict(self, feat):
        feat = StandardScaler().fit_transform(feat)
        return self.get_model().predict(feat)


class PedestrianDetector:
    def __init__(self, config: Config):
        """

        :type config: Config
        """
        self.config = config

        self.constructed_model = None
        self.samples: list = []
        self.labels: list = []
        self.feat_size = 0

        # attempting to load rcnn model
        f_rcnn_path = self.config.faster_rcnn_inception_v2_pb_path
        if f_rcnn_path is not None:
            self.f_rcnn_detector = RCNNPedestrianDetector(f_rcnn_path)
            self.__verbose('loaded faster rcnn inception pb')

        # attempting to load hog desc model
        if os.path.isfile(self.config.model_path):
            try:
                self.load(self.config.model_path)
            except Exception as e:
                self.__verbose(str(e), level='error')

        # loading third party pedestrian detector
        self.cv2HOG = cv2.HOGDescriptor()
        self.cv2HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def set_class_dict(self, class_dict: dict):
        """

        :param class_dict: list of path and class number. {path: 'images path', class: 1}
        eg.
        >>> { 'person':{ 'path':'path_to_class_1_images', 'class': 1  },{ 'animal':{ 'path':'path_to_class_2_images', 'class': 2 }
        }
        :return:
        """

        if not isinstance(class_dict, dict):
            self.__verbose('class dict must be type dict!', level='error')
            raise TypeError('class dict must be type dict!')

        for key in class_dict.keys():
            if class_dict[f'{key}'].keys().__contains__('path'):
                path = class_dict[f'{key}']['path']
                if not os.path.exists(path) or not os.path.isdir(path):
                    self.__verbose('class images path is not valid', level='error')
                    raise FileNotFoundError('class images path is not valid')

        self.config.class_dict = class_dict

    def add_class(self, name: str, path: str, class_number: int):
        """

        :param name: name of class
        :param path: path to class image files directory
        :param class_number: int value for class group
        :return:
        """
        self.config.class_dict[f'{name}'] = {
            'path': path,
            'class_number': class_number
        }

    def _get_feature_for_image(self, img):
        if img is None:
            return

        hist = hog(img, orientations=self.config.orientations, pixels_per_cell=self.config.pixels_per_cell,
                   cells_per_block=self.config.cells_per_block, block_norm='L2-Hys', visualise=False,
                   transform_sqrt=False, feature_vector=True, normalise=None)

        self.__verbose(f'{type(hist)} {len(hist)}', level='debug')

        if self.feat_size == 0:
            self.feat_size = len(hist)
        else:
            if len(hist) != self.feat_size:
                raise ValueError(f'hist size changed from {self.feat_size} to {len(hist)}')

        return hist

    def _ten_random_windows(self, img):
        h, w = img.shape[:2]
        if w < self.config.min_wdw_sz[0] or h < self.config.min_wdw_sz[1]:
            return []

        w = w - self.config.min_wdw_sz[0]
        h = h - self.config.min_wdw_sz[1]

        windows = []

        for i in range(10):
            x = random.randint(0, w)
            y = random.randint(0, h)
            windows.append([x, y])

        return windows

    def _add_features_from_image_in_dir(self, dir_name: str, class_number: int = 0, name=None, max_sample=None):
        # Get positive samples
        self.__verbose(f'for class: {class_number} with files images in the directory {dir_name}')

        def add_data_point(feat):
            if feat is None:
                self.__verbose(f'failed extraction for file {filename}')
                return
            else:
                self.samples.append(feat)
                self.labels.append(class_number)
                self.__verbose(f'successful extraction for file {filename} class<name: {name}, class: {class_number}>')
                self.__verbose(f'data_sets set: samples= {len(self.samples)}, labels= {len(self.labels)}>')

        count = 0

        if name == 'negative':
            for filename in os.listdir(dir_name):
                # attempt read image in gray scale
                self.__verbose(f'attempting extraction for file {filename}')
                img = cv2.imread(os.path.join(dir_name, filename), 0)

                if self.config.img_resize is not None and self.config.img_resize > self.config.min_wdw_sz[1]:
                    img = imutils.resize(img, width=min(self.config.img_resize, img.shape[1]))

                windows = self._ten_random_windows(img)

                for win in windows:
                    try:
                        hist = self._get_feature_for_image(
                            scale_image_section(img, [win[0], win[1],
                                                      self.config.min_wdw_sz[0],
                                                      self.config.min_wdw_sz[1]],
                                                self.config.min_wdw_sz))
                        add_data_point(hist)
                    except ValueError as e:
                        self.__verbose(f'{e}', level='error')

                if max_sample is not None:
                    count += 1
                    if count == max_sample:
                        break

        else:
            for filename in os.listdir(dir_name):
                # attempt read image in gray scale
                self.__verbose(f'attempting extraction for file {filename}')

                img = cv2.imread(os.path.join(dir_name, filename), 0)

                if self.config.img_resize is not None and self.config.img_resize > self.config.min_wdw_sz[1]:
                    img = imutils.resize(img, width=min(self.config.img_resize, img.shape[1]))

                result = rects_from_image(img)

                for rect in result:
                    try:
                        hist = self._get_feature_for_image(
                            scale_image_section(img, rect, self.config.min_wdw_sz, gravity='top'))
                        add_data_point(hist)
                        hist = self._get_feature_for_image(
                            scale_image_section(img, rect, self.config.min_wdw_sz, gravity='center'))
                        add_data_point(hist)
                        hist = self._get_feature_for_image(
                            scale_image_section(img, rect, self.config.min_wdw_sz, gravity='bottom'))
                        add_data_point(hist)
                    except ValueError as e:
                        self.__verbose(f'{e}', level='error')

                try:
                    hist = self._get_feature_for_image(
                        scale_image_section(img, None, self.config.min_wdw_sz))
                    add_data_point(hist)
                except ValueError as e:
                    self.__verbose(f'{e}', level='error')

                if max_sample is not None:
                    count += 1
                    if count == max_sample:
                        break

    def __hard_negative_mine(self, trained_svm):

        hard_negatives = []
        hard_negative_labels = []

        count = 0

        negative_class: dict = self.config.class_dict['negative']
        min_wdw_sz = self.config.min_wdw_sz

        if negative_class.keys().__contains__('path'):
            dir_name = negative_class['path']
            if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
                self.__verbose('path to class image files is not valid!', level='error')
                return

            for item in os.listdir(dir_name):
                img = cv2.imread(os.path.join(dir_name, item), 0)
                for (x, y) in self.__sliding_window(img.copy()):

                    im_window = img[y:y + min_wdw_sz[1], x:x + min_wdw_sz[0]]
                    self.__verbose(f'x: {x}, y: {y}, window shape: {im_window.shape}')

                    if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                        self.__verbose(f'continue: {im_window.shape}')
                        continue
                    try:
                        hist = self._get_feature_for_image(im_window)
                        if hist is not None:
                            self.__verbose(f'hard negative features length: {len(hist)}')
                            if trained_svm.predict([hist]) == 1:
                                hard_negatives.append(hist)
                                hard_negative_labels.append(negative_class['class'])
                                count = count + 1

                            if count == self.config.max_hard_negatives:
                                return np.array(hard_negatives), np.array(hard_negative_labels)
                        else:
                            self.__verbose('failed to extract features', level='error')

                    except ValueError as e:
                        self.__verbose(f'{e}', level='error')

                if self.config.verbose:
                    self.__verbose("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(
                        round((count / float(self.config.max_hard_negatives)) * 100, 4)) + " %", level='sout')

        else:
            self.__verbose('class has no path to images!', level='error')
            raise FileNotFoundError('path to class images files does not exist!')

        return np.array(hard_negatives), np.array(hard_negative_labels)

    def construct(self, samples=None, labels=None, max_sample: int = None, base_model: Model = LinearSVCModel(),
                  pre_process_func=None, hard_mine=False, save_dataset: bool = False,
                  dev_save_name: str = 'output.dev', prod_save_name: str = 'output'):
        """
        train the model with dataset provided. [path or direct feed features]
        also saves model in specified path in config.model_path
        :param samples: list of features in type np.ndarray
        :param labels: list of labels in type np.ndarray
        :param max_sample: max number of data_sets points to process
        :param base_model: instance of the Model class to fit samples and labels
        :param pre_process_func: function that returns pre processed version of samples and labels
               format: >>> samples, labels = pre_process_func(samples, labels)
        :param hard_mine: mines for false positive features using the negative the points and the dev model
        :param save_dataset: saves current sample and features as numpy output objects /sample.npy, /labels.npy
        :param dev_save_name:  name to save output trained development model
        :param prod_save_name:  name to save output trained production model
        :return: instance of trained Model
        """
        if samples is None or labels is None:
            if not isinstance(self.config.class_dict, dict):
                self.__verbose('class list must be type dict!', level='error')
                raise TypeError('class list must be type dict!')

            self.__verbose('extracting features.')
            for k, v in self.config.class_dict.items():
                _class = self.config.class_dict[f'{k}']
                if not isinstance(_class, dict):
                    self.__verbose('class object must be type dict!')
                    continue
                self._add_features_from_image_in_dir(_class['path'], _class['class'], k, max_sample)

            self.__verbose('start building model')

            # Convert objects to Numpy Objects
            samples = np.float32(self.samples)
            labels = np.array(self.labels)
        else:
            if samples is not None and labels is not None and max_sample is not None and max_sample > 0:
                samples = samples[:max_sample]
                labels = labels[:max_sample]

        # Shuffle Samples
        samples, labels = shuffle(samples, labels, random_state=0)

        self.__verbose(f'data_sets set: data_sets={len(samples)}, labels={len(labels)}')

        # rand = np.random.RandomState(321)
        # orders = rand.permutation(len(samples))
        # samples = samples[orders]
        # labels = labels[orders]

        def get_model():
            if base_model is None or base_model == 'linear-svc':
                return LinearSVCModel()
            elif base_model == 'logistic-regression':
                return LogisticRegressionModel()
            elif base_model == 'svm':
                return SVMModel()
            else:
                return base_model

        model = get_model()

        _type = 'development' if hard_mine else 'production'

        self.__verbose(f'training {_type} model')
        if pre_process_func is not None:
            samples, labels = pre_process_func(samples, labels)

        model.fit(samples, labels)
        self.__verbose(f'done training {_type} model')
        self._save(model, dev_save_name if hard_mine else prod_save_name)

        if hard_mine:
            self.__verbose('gathering hard negatives')
            hn_feats, hn_labels = self.__hard_negative_mine(model)
            self.__verbose('finished gathering hard negatives')

            self.__verbose('adding new samples')
            np.concatenate((samples, hn_feats), axis=0)
            np.concatenate((labels, hn_labels), axis=0)

            # Shuffle Samples
            samples, labels = shuffle(samples, labels, random_state=0)

            model = get_model()

            self.__verbose("training production model")
            if pre_process_func is not None:
                samples, labels = pre_process_func(samples, labels)
            model.fit(samples, labels)
            self.__verbose("done training production model")
            self._save(model, prod_save_name)

        self.constructed_model = model

        if save_dataset:
            np.save(self.config.model_path + '/labels.npy', labels)
            np.save(self.config.model_path + '/samples.npy', samples)

        return model

    def _save(self, model, name=f'output{time.time()}'):

        if self.config.model_path:
            self.__verbose('saving model')

            if os.path.isdir(self.config.model_path):
                if not os.path.exists(self.config.model_path):
                    os.mkdir(path=self.config.model_path)
                save_path = os.path.join(self.config.model_path, f'{name}.model')
                if isinstance(model.get_model(), cv2.ml_SVM):
                    model.get_model().save(os.path.join(save_path))
                else:
                    joblib.dump(model, save_path)
                    self.__verbose(f'saved model to {save_path}')
            else:
                if isinstance(model.get_model(), cv2.ml_SVM):
                    model.get_model().save(self.config.model_path)
                else:
                    joblib.dump(model, self.config.model_path)
                    self.__verbose(f'saved model to {self.config.model_path}')

    def __verbose(self, param, level: str = 'log'):
        if self.config.verbose:
            if level == 'log':
                logger.info(param)
            elif level == 'debug':
                logger.debug(param)
            elif level == 'warning':
                logger.warning(param)
            elif level == 'error':
                logger.error(param)
            elif level == 'criticaL':
                logger.critical(param)
            elif level == 'sout':
                sys.stdout.write(param)
                sys.stdout.flush()

    def __sliding_window(self, image):
        """
        This function returns a patch of the input 'image' of size
        equal to 'window_size'. The first image returned top-left
        co-ordinate (0, 0) and are increment in both x and y directions
        by the 'step_size' supplied.

        :param image: numpy array
        :return: (x, y, im_window)
        """

        window_size = self.config.min_wdw_sz
        step_size = self.config.step_size

        height, width = image.shape
        stop_horizontal = width - window_size[0]
        stop_vertical = height - window_size[1]

        for y in range(0, height, step_size[1]):
            if y <= stop_vertical:
                for x in range(0, width, step_size[0]):
                    if x <= stop_horizontal:
                        yield (x, y)

    def detect_with_sliding_windows(self, image_or_image_path):
        """
        detects objects of interest in the image frame

        :param image: numpy array
        :return: scores, list of list denoting all square boxes around detected image
        """

        image = image_or_image_path

        if isinstance(image_or_image_path, str):
            image = cv2.imread(image_or_image_path)

        if image is None:
            self.__verbose(f'image path \'{image_or_image_path}\' is not valid')
            raise FileNotFoundError(f'Image path \'{image_or_image_path}\' does not exist')

        if not self.constructed_model:
            self.__verbose('model not constructed. run self.construct', level='error')
            raise Exception('model not constructed')
        gray_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        width = image.shape[1]

        if self.config.img_resize is not None and self.config.img_resize > self.config.min_wdw_sz[1]:
            width = min(self.config.img_resize, width)

        resized_gray_img = imutils.resize(gray_img, width=width)
        resized_clone_img = imutils.resize(image, width=width)
        self.__verbose('resizing image')  # TODO: convert to gray scale

        min_wdw_sz = self.config.min_wdw_sz
        downscale = self.config.downscale

        if self.constructed_model is None:
            if not self.config.model_path or not os.path.exists(self.config.model_path):
                self.__verbose('model path does not exist', level='error')
                return

            if os.path.isdir(self.config.model_path) and os.path.exists(
                    os.path.join(self.config.model_path, 'svmhogdetector.model')):
                model_path = os.path.join(self.config.model_path, 'svmhogdetector.model')
            else:
                model_path = self.config.model_path

            if os.path.exists(model_path):
                self.__verbose('model does not exist', level='error')
                return

            self.load(model_path)

        clf = self.constructed_model

        # List to store the detections
        detections = []
        # The current scale of the image
        scale = 0

        self.__verbose('running detection...')
        p_visual = resized_gray_img.copy()

        for im_scaled in pyramid_gaussian(resized_gray_img.copy(), downscale=downscale):
            # The list contains detections at the current scale
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                break

            im_scaled = imutils.resize(resized_gray_img.copy(), height=im_scaled.shape[0], width=im_scaled.shape[1])

            for (x, y) in self.__sliding_window(im_scaled.copy()):
                im_window = im_scaled[y:y + min_wdw_sz[1], x:x + min_wdw_sz[0]]
                # cv2.imshow("window", im_window)
                # cv2.waitKey(1)

                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    self.__verbose(f'continue: {im_window.shape}')
                    continue

                features = hog(im_window,
                               orientations=self.config.orientations,
                               pixels_per_cell=self.config.pixels_per_cell,
                               cells_per_block=self.config.cells_per_block, block_norm='L2-Hys',
                               visualise=False, transform_sqrt=False, feature_vector=True, normalise=None)

                features = features.reshape(1, -1)
                prediction = clf.predict(features)

                if self.config.visualize:
                    visual = im_scaled.copy()
                    cv2.rectangle(visual, (x, y), (x + min_wdw_sz[0], y + min_wdw_sz[1]), (0, 0, 255), 2)
                    cv2.imshow("sliding window", visual)
                    cv2.waitKey(1)

                if prediction == 1:
                    accuracy = clf.decision_function(features)
                    self.__verbose(f'[{prediction}] total-count:{len(detections)} x:{x}, y:{y}, accuracy:{accuracy}')
                    if clf.decision_function(features) > self.config.accuracy_threshold:
                        detections.append((int(x * (downscale ** scale)), int(y * (downscale ** scale)),
                                           accuracy, int(min_wdw_sz[0] * (downscale ** scale)),
                                           int(min_wdw_sz[1] * (downscale ** scale))))

                        if self.config.visualize:
                            i, j, _, w, h = detections[-1]
                            cv2.rectangle(p_visual, (i, j), (i + w, j + h), (0, 255, 0), thickness=2)
                            cv2.imshow("detections", p_visual)
                            cv2.waitKey(1)

            scale += 1

        self.__verbose('detection process complete')

        rect_list = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        scores = [score[0] for (x, y, score, w, h) in detections]
        scores = np.array(scores)
        picks = non_max_suppression(rect_list, probs=scores, overlapThresh=self.config.non_max_suppression_threshold)

        for (x, y, _, w, h) in detections:
            cv2.rectangle(resized_gray_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        for (xA, yA, xB, yB) in picks:
            w = resized_clone_img.shape
            y = image.shape
            width_scale = y[0] / w[0]
            height_scale = y[1] / w[1]
            cv2.rectangle(resized_clone_img, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.rectangle(image, (int(xA * width_scale), int(yA * width_scale)),
                          (int(xB * height_scale), int(yB * height_scale)), (0, 255, 0), 2)

        if self.config.visualize:
            cv2.destroyAllWindows()

            plt.axis("off")
            plt.imshow(cv2.cvtColor(resized_gray_img, cv2.COLOR_BGR2RGB))
            plt.title("Raw Detection before NMS")
            plt.show()

            # plt.axis("off")
            # plt.imshow(cv2.cvtColor(resized_clone_img, cv2.COLOR_BGR2RGB))
            # plt.title("Raw Detection after NMS")
            # plt.show()

            plt.axis("off")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Final Detections after applying NMS")
            plt.show()

        self.__verbose(f'detection complete')

        return scores, picks

    def detect_with_conv_net(self, image_or_image_path):
        if self.f_rcnn_detector is None:
            self.__verbose('path to faster_rcnn_inception_v2_pb not found!. set path to file in config', level='error')
            raise FileNotFoundError(f'path to faster_rcnn_inception_v2_pb not found!')
        else:
            original = image_or_image_path

            if isinstance(image_or_image_path, str):
                original = cv2.imread(image_or_image_path)

            if original is None:
                self.__verbose(f'image path \'{image_or_image_path}\' is not valid')
                raise FileNotFoundError(f'Image path \'{image_or_image_path}\' does not exist')

            image = original.copy()  # cv2.resize(original.copy(), (1280, 720))

            boxes, scores, classes, num = self.f_rcnn_detector.detect(image)

            picks = []
            pick_scores = []

            for i in range(len(boxes)):
                # Class 1 represents human
                if classes[i] == 1 and scores[i] > self.config.accuracy_threshold:
                    # y, x, h, w = boxes[i]
                    picks.append(boxes[i])
                    pick_scores.append(scores[i])

            if self.config.visualize:
                for (x, y, w, h) in picks:
                    cv2.rectangle(original, (x, y), (w, h), (0, 255, 0), 2)

                plt.axis("off")
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.title("Detected Results")
                plt.show()

            return pick_scores, picks

    def detect_with_tp_cv2(self, image_or_image_path):
        original = image_or_image_path

        if isinstance(image_or_image_path, str):
            original = cv2.imread(image_or_image_path)

        if original is None:
            self.__verbose(f'image path \'{image_or_image_path}\' is not valid')
            raise FileNotFoundError(f'Image path \'{image_or_image_path}\' does not exist')

        (boxes, _) = self.cv2HOG.detectMultiScale(original, winStride=(4, 4), padding=(8, 8), scale=1.05)
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        picks = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
        return None, picks

    def load(self, model_path):
        self.__verbose('attempting to load model')
        if os.path.isfile(model_path):
            try:
                model = joblib.load(model_path)
            except Exception as e:
                model = cv2.ml.SVM_load(model_path)

            if model is None:
                self.__verbose('failed to load model from file', level='error')
                raise ValueError(f'failed to load model from file \'{model_path}\'')
            else:
                self.constructed_model = SVMModelWrapper(model)
                self.__verbose('success loading model')
        else:
            self.__verbose('failed to load model!. check if file exists.', level='error')
            raise FileNotFoundError(f'failed to load model at \'{model_path}\'')
