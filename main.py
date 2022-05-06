from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import sys

import os
import pickle

import cv2
import joblib
import numpy as np

import Utils
from ImageUtils import vector_quantization

global path


class img_viewed(QWidget):

    def __init__(self, parent=None):
        super(img_viewed, self).__init__(parent)
        self.parent = parent
        self.width = 960
        self.height = 800
        self.setWindowTitle("Image Search")

        self.scroll_ares_images = QScrollArea(self)
        self.scroll_ares_images.setWidgetResizable(True)

        self.scrollAreaWidgetContents = QWidget(self)
        self.scrollAreaWidgetContents.setObjectName('scrollAreaWidgetContends')

        # grid Layout
        self.gridLayout = QGridLayout(self.scrollAreaWidgetContents)
        self.scroll_ares_images.setWidget(self.scrollAreaWidgetContents)
        self.scroll_ares_images.setGeometry(20, 20, self.width, int(self.height * 0.7))
        self.vertocall = QVBoxLayout()

        # initialize button 1
        self.open_file_pushbutton = QPushButton(self)
        self.open_file_pushbutton.setGeometry(150, int(35 + self.height * 0.7), 100, 30)
        self.open_file_pushbutton.setText('Choose Image Directory To Search')
        self.open_file_pushbutton.adjustSize()
        self.open_file_pushbutton.setObjectName('open_pushbutton')

        self.open_file_pushbutton.clicked.connect(self.choose_img_directory)

        self.initial_output_path = 'Output'
        self.initial_input_path = 'Input'

        # set image original size
        self.displayed_image_size = 150
        self.col = 0
        self.row = 0

        # initialize button 2
        self.start_file_pushbutton = QPushButton(self)
        self.start_file_pushbutton.setGeometry(600, 35 + int(self.height * 0.7), 100, 30)
        self.start_file_pushbutton.setObjectName('start_pushbutton')
        self.start_file_pushbutton.setText('Search Image in Default Path')
        self.start_file_pushbutton.adjustSize()
        self.start_file_pushbutton.clicked.connect(self.get_similar_default)

        # initialize progress bar - invisible
        self.progressbar = QProgressBar(self)
        self.progressbar.setObjectName('progress_bar')
        self.progressbar.setGeometry(int(self.width / 2 - 100), int(self.height * 0.7 / 2), 200, 25)
        self.progressbar.setMaximum(100)
        self.step = 0
        self.progressbar.setVisible(False)

        self.vertocall.addWidget(self.scroll_ares_images)
        self.show()

        # self.initial_path = None

    def get_similar_image(self, test_image_root):

        clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

        # set test output path
        out_path = "Output"

        out_path2 = "Output2"

        out_path3 = "Output3"

        # set test image root
        # test_image_root = "Input"
        # set test image path
        test_image_paths = os.listdir(test_image_root)

        # set input class manually
        test_class = "pizza"

        # des_list - List where all the descriptors are stored
        des_list = []

        # Get the testing image path store them in a list
        # get descriptors for all testing images
        if len(test_image_paths) != 0:
            for test_image_path in test_image_paths:
                file_path = test_image_root + "/" + test_image_path
                img_test = cv2.imread(file_path)
                if img_test is None:
                    print("No such file {}\nCheck if the file exists".format(test_image_path))
                    exit()
                sift = cv2.xfeatures2d.SIFT_create()
                key_points, des = sift.detectAndCompute(img_test, None)
                des_list.append((test_image_paths, des))
        else:
            print("No Input image in ./Input folder!")
        # Stack all the descriptors vertically in a numpy array
        descriptors = des_list[0][1]

        for test_image_path, descriptor in des_list[0:]:
            descriptors = np.vstack((descriptors, descriptor))

        # Apply Dimension reduction to local features
        # test_features = np.zeros((len(test_image_paths), k), "float32")
        # for i in xrange(len(test_image_paths)):
        #     words, distance = vq(des_list[i][1], voc)
        #     for w in words:
        #         test_features[i][w] += 1
        test_img_features = vector_quantization(test_image_paths, k, des_list, voc)

        # Perform Tf-Idf vectorization
        nbr_occurrences = np.sum((test_img_features > 0) * 1, axis=0)
        idf = np.array(np.log((1.0 * len(test_image_paths) + 1) / (1.0 * nbr_occurrences + 1)), 'float32')

        # Scale the features
        test_img_features = stdSlr.transform(test_img_features)

        # Perform the predictions
        predictions = [classes_names[i] for i in clf.predict(test_img_features)]

        with open('vocabulary.pkl', 'rb') as f:
            vocabulary = pickle.load(f)

        distance = []

        for word in vocabulary:
            distance.append([word[0], np.linalg.norm(word[1] - test_img_features[0])])

        # sort by distance
        distance.sort(key=lambda t: t[1])

        # num_of_similar_img - number of outputs
        num_of_similar_img = 55
        true_pos = 0
        for test_image_path, prediction in zip(test_image_paths, predictions):
            image = cv2.imread(test_image_path)
            print(prediction)

        classifier = []

        for i in range(len(distance)):
            if i > num_of_similar_img - 1:
                print(distance[i][1])
                break
            elif distance[i][1] > 5:
                break

            class_of_img = Utils.get_class(distance[i][0])

            if class_of_img == test_class:
                true_pos = true_pos + 1
                classifier.append(distance[i][0])

            path = distance[i][0]
            image = cv2.imread(path)
            cv2.imwrite(os.path.join(out_path, 'rank_{0}.jpg'.format(i)), image)

        if true_pos >= num_of_similar_img / 2:
            for c in range(len(classifier)):
                path = str(classifier[c])
                image = cv2.imread(path)
                cv2.imwrite(os.path.join(out_path3, 'rank_{0}.jpg'.format(c)), image)
            self.initial_output_path = out_path3

        recall = true_pos / num_of_similar_img

        print('recall = ' + str(recall))

        self.start_img_viewer()

    def choose_img_directory(self):
        file_path = QFileDialog.getExistingDirectory(self, '选择文文件夹', '/')
        if file_path == None:
            QMessageBox.information(self, '提示', '文件为空，请重新操作')
        else:
            self.initial_input_path = file_path
            self.get_similar_image(self.initial_input_path)

    def get_similar_default(self):
        self.get_similar_image(self.initial_input_path)

    def start_img_viewer(self):
        if self.initial_output_path:
            file_path = self.initial_output_path
            print('file_path为{}'.format(file_path))
            print(file_path)
            img_type = 'jpg'
            if file_path and img_type:
                png_list = list(i for i in os.listdir(file_path) if str(i).endswith('.{}'.format(img_type)))
                print(png_list)
                num = len(png_list)
                name_list = self.get_output_file_name(num)
                if num != 0:
                    for i in range(num):
                        image_id = str(file_path + '/' + name_list[i])
                        print(image_id)
                        pixmap = QPixmap(image_id)
                        self.addImage(i, pixmap, image_id, name_list[i])
                        print(pixmap)
                        QApplication.processEvents()
                else:
                    QMessageBox.warning(self, 'error', 'no .jpg file in directory')
                    self.event(exit())
            else:
                QMessageBox.warning(self, 'error', 'no .jpg file in director, please waite')
        else:

            QMessageBox.warning(self, 'error', 'no .jpg file in director, please waite')

    def get_output_file_name(self, len):
        name_list = []
        for i in range(len):
            name_list.append('rank_{0}.jpg'.format(i))
        return name_list

    # progress bar control
    def set_progress_bar_visibility(self, bool):
        self.progressbar.setVisible(bool)

    def handle_progressbar(self):
        self.progressbar.setValue(self.step)

    def update_step(self, step_current):
        self.step = step_current

    def loc_fil(self, stre):
        print('Path{}'.format(stre))
        self.initial_output_path = stre

    def addImage(self, add_term, pixmap, image_id, image_name):
        # 图像法列数
        nr_of_columns = 2
        # 这个布局内的数量
        nr_of_widgets = self.gridLayout.count()
        self.max_columns = nr_of_columns
        if self.col < self.max_columns:
            self.col = self.col + 1
        else:
            self.col = 0
            self.row += 1

        if add_term == 0:
            self.col = 0
        clickable_image = QClickableImage(self.displayed_image_size, self.displayed_image_size, pixmap, image_name)
        clickable_image.clicked.connect(self.on_left_clicked)
        clickable_image.rightClicked.connect(self.on_right_clicked)
        self.gridLayout.addWidget(clickable_image, self.row, self.col)

    def setDisplayedImageSize(self, image_size):
        self.displayed_image_size = image_size

    def on_left_clicked(self, image_id):
        print('left clicked - image id = ' + image_id)

    def on_right_clicked(self, image_id):
        print('right clicked - image id = ' + image_id)

class QClickableImage(QWidget):
    image_id = ''

    def __init__(self, width=0, height=0, pixmap=None, image_name=''):
        QWidget.__init__(self)

        self.layout = QVBoxLayout(self)
        self.label1 = QLabel()
        self.label1.setObjectName('label1')
        self.lable2 = QLabel()
        self.lable2.setObjectName('label2')
        self.width = width
        self.height = height
        self.pixmap = pixmap

        if self.width and self.height:
            self.resize(self.width, self.height)
        if self.pixmap:
            pixmap = self.pixmap.scaled(QSize(self.width, self.height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label1.setPixmap(pixmap)
            self.label1.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label1)
        if image_name:
            self.image_id = image_name
            self.lable2.setText(image_name)
            self.lable2.setAlignment(Qt.AlignCenter)
            # 让文字自适应大小
            self.lable2.adjustSize()
            self.layout.addWidget(self.lable2)
        self.setLayout(self.layout)

    clicked = pyqtSignal(object)
    rightClicked = pyqtSignal(object)

    def imageId(self):
        return self.image_id


def run_user_interface():
    app = QApplication(sys.argv)
    window = img_viewed()
    window.show()
    sys.exit(app.exec_())


run_user_interface()
