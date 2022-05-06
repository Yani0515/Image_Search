# # -*- coding: utf-8 -*-
# import os
# import tkinter
# from tkinter import *
# import tkinter.filedialog
# from PIL import Image, ImageTk
# import cv2
#
#
# def Showimage(imgCV_in, canva, layout="null"):
#     """
#     Showimage()是一个用于在tkinter的canvas控件中显示OpenCV图像的函数。
#     使用前需要先导入库
#     import cv2 as cv
#     from PIL import Image,ImageTktkinter
#     并注意由于响应函数的需要，本函数定义了一个全局变量 imgTK，请不要在其他地方使用这个变量名!
#     参数：
#     imgCV_in：待显示的OpenCV图像变量
#     canva：用于显示的tkinter canvas画布变量
#     layout：显示的格式。可选项为：
#         "fill"：图像自动适应画布大小，并完全填充，可能会造成画面拉伸
#         "fit"：根据画布大小，在不拉伸图像的情况下最大程度显示图像，可能会造成边缘空白
#         给定其他参数或者不给参数将按原图像大小显示，可能会显示不全或者留空
#     """
#     canvawidth = int(canva.winfo_reqwidth())
#     canvaheight = int(canva.winfo_reqheight())
#     sp = imgCV_in.shape
#     cvheight = sp[0]  # height(rows) of image
#     cvwidth = sp[1]  # width(colums) of image
#     if (layout == "fill"):
#         imgCV = cv2.resize(imgCV_in, (canvawidth, canvaheight), interpolation=cv2.INTER_AREA)
#     elif (layout == "fit"):
#         if (float(cvwidth / cvheight) > float(canvawidth / canvaheight)):
#             imgCV = cv2.resize(imgCV_in, (canvawidth, int(canvawidth * cvheight / cvwidth)),
#                                interpolation=cv2.INTER_AREA)
#         else:
#             imgCV = cv2.resize(imgCV_in, (int(canvaheight * cvwidth / cvheight), canvaheight),
#                                interpolation=cv2.INTER_AREA)
#     else:
#         imgCV = imgCV_in
#     imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
#     current_image = Image.fromarray(imgCV2)  # 将图像转换成Image对象
#     imgTK = ImageTk.PhotoImage(image=current_image)  # 将image对象转换为imageTK对象
#     canva.create_image(0, 0, anchor=NW, image=imgTK)
#
#
# # initialize
# root = Tk()
#
# # set window title
# root.title('User Interface')
#
# # set window size
# root.geometry('600x700')
#
# # set size
# window_height = 700
# window_width = 600
#
# # fix window size
# root.minsize(window_width, window_height)
# root.maxsize(window_width, window_height)
#
# # ------------------------------set button to open directory
# filename = ''
#
#
# def xz():
#     filenames = tkinter.filedialog.askopenfilename()
#     if len(filenames) != 0:
#         string_filename = ""
#         for i in range(0, len(filenames)):
#             string_filename += str(filenames[i])
#         lb.config(text="The directory you choose is: " + string_filename)
#     else:
#         lb.config(text="You haven't choose any directory")
#
#
# btn_open_directory = Button(root, text="Choose Directory", command=xz)
# btn_open_directory.pack(fill=tkinter.X, side=tkinter.BOTTOM)
# lb = Label(root, text='')
# lb.pack(fill=tkinter.X, side=tkinter.BOTTOM)
#
# # ================
# img = cv2.imread("image_0001.jpg",1)
#
# # ----------- image display -----------------
# # set canvas size
# canvas_height = 550
# canvas_width = window_width * 0.955
#
# # set base frame
# base_frame = Frame(root, relief=GROOVE, width=window_width, height=canvas_height)
# base_frame.place(relx=0, rely=0)
#
# # set canvas - used to display images
# # TODO：根据图片数量高度； 图片排版
# canvas = Canvas(base_frame, bg='black', height=canvas_height, width=canvas_width, confine=False)
# canvas.place(relx=0, rely=0)
# Showimage(img, canvas, layout='fit')
# # # set frame on canvas to locate scrollbar
# # can_frame = Frame(canvas, width=canvas_width, height=canvas_height/2, bg='black')
# # canvas.create_window((0, 0), window=can_frame, anchor='nw')
#
# # set a scrollbar
# # TODO: 闲置滚动条滚动范围；滚动条样式；滚动条总长（？）
# vbar = Scrollbar(base_frame, orient=VERTICAL, takefocus=0.5)
# vbar.place(relwidth=0.04, relheight=1, relx=0.955, rely=0)
# # config scrollbar to the canvas
# vbar.config(command=canvas.yview)
# canvas.config(yscrollcommand=vbar.set)
# # range = 1000
# canvas.config(scrollregion=(0, 0, canvas_width, 1000))
#
# root.mainloop()
import this
import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import sys


# class BackendThread(QProgressBar):
#     # 通过类成员对象定义信号
#     update_step = pyqtSignal(str)
#
#     # 处理业务逻辑
#     def run(self):
#         while 1:
#             # 刷新1-10
#             for i in range(1, 11):
#                 QProgressBar.setValue()
#                 time.sleep(1)


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
        self.open_file_pushbutton.setGeometry(150, int(35 + self.height * 0.7), 150, 30)
        self.open_file_pushbutton.adjustSize()
        self.open_file_pushbutton.setObjectName('open_pushbutton')
        self.open_file_pushbutton.setText('Choose Image Directory')

        self.open_file_pushbutton.clicked.connect(self.open)

        # initialize button 2
        self.start_file_pushbutton = QPushButton(self)
        self.start_file_pushbutton.setGeometry(750, 35 + int(self.height * 0.7), 100, 30)
        self.start_file_pushbutton.setObjectName('start_pushbutton')
        self.start_file_pushbutton.setText('开始')
        self.start_file_pushbutton.clicked.connect(self.start_img_viewer)

        # initialize progress bar - invisible
        self.progressbar = QProgressBar(self)
        self.progressbar.setObjectName('progress_bar')
        self.progressbar.setGeometry(int(self.width / 2 - 100), int(self.height * 0.7 / 2), 200, 25)
        self.progressbar.setMaximum(100)
        self.step = 0
        self.progressbar.setVisible(False)

        self.vertocall.addWidget(self.scroll_ares_images)
        self.show()

        # set image original size
        self.displayed_image_size = 100
        self.col = 0
        self.row = 0

        # self.initial_path = None
        self.initial_path = 'Output'

    def open(self):
        file_path = QFileDialog.getExistingDirectory(self, '选择文文件夹', '/')
        if file_path == None:
            QMessageBox.information(self, '提示', '文件为空，请重新操作')
        else:
            self.initial_path = file_path

    def start_img_viewer(self):
        if self.initial_path:
            file_path = self.initial_path
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

    # progress bar control
    def set_progress_bar_visibility(self, bool):
        self.progressbar.setVisible(bool)

    # def progress_bar_thread(self):
    #     # make thread
    #     self.thread = QThread()
    #     self.backend = BackendThread(self.progressbar)
    #
    #     # 连接信号
    #     self.backend.update_date.connect(self.handle_progressbar())
    #     self.backend.moveToThread(self.thread)
    #
    #     # 开始线程
    #     self.thread.started.connect(self.backend.run)
    #     self.thread.start()

    def handle_progressbar(self):
        self.progressbar.setValue(self.step)

    def update_step(self, step_current):
        self.step = step_current

    def get_output_file_name(self, len):
        name_list = []
        for i in range(len):
            name_list.append('rank_{0}.jpg'.format(i))
        return name_list

    def loc_fil(self, stre):
        print('Path{}'.format(stre))
        self.initial_path = stre

    def geng_path(self, loc):
        print('path: {}'.format(loc))

    def gen_type(self, type):
        print('type: {}'.format(type))

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

    def on_left_clicked(self, image_id):
        print('left clicked - image id = ' + image_id)

    def on_right_clicked(self, image_id):
        print('right clicked - image id = ' + image_id)

    def setDisplayedImageSize(self, image_size):
        self.displayed_image_size = image_size


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

    def mouseressevent(self, ev):
        print('55555555555555555')
        if ev.button() == Qt.RightButton:
            print('dasdasd')
            # 鼠标右击
            self.rightClicked.emit(self.image_id)
        else:
            self.clicked.emit(self.image_id)

    def imageId(self):
        return self.image_id


def run_user_interface():
    app = QApplication(sys.argv)
    window = img_viewed()
    window.show()
    sys.exit(app.exec_())



