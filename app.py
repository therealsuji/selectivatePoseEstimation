
from PyQt5 import QtCore, QtGui, QtWidgets
from tools.tracking import Tracking
from utils.video_utils import (
    video_to_images, gen_masked_video, images_to_video)
import cv2
import numpy as np
from tools.infer_video_d2 import generate2dKeypoints
from tools.prepare_data_2d_custom import process2D_keypoints
from tools.inference_3d import generate_3d_inference


class Application(object):

    def __init__(self):
        self.tracking_complete = False
        self.tracker = Tracking()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(837, 679)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.placeholder = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.placeholder.setFont(font)
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder.setObjectName("placeholder")
        self.verticalLayout_3.addWidget(self.placeholder)
        self.horizontalLayout.addWidget(self.frame)
        self.frame1 = QtWidgets.QFrame(self.centralwidget)
        self.frame1.setMaximumSize(QtCore.QSize(200, 16777215))
        self.frame1.setObjectName("frame1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame1)
        self.verticalLayout_2.setSizeConstraint(
            QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(-1, -1, -1, 9)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.load_btn = QtWidgets.QPushButton(self.frame1)
        self.load_btn.setObjectName("load_btn")
        self.verticalLayout_2.addWidget(self.load_btn)
        self.label = QtWidgets.QLabel(self.frame1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setStyleSheet("margin-top:12")
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.selection_input = QtWidgets.QLineEdit(self.frame1)
        self.selection_input.setObjectName("selection_input")
        self.verticalLayout_2.addWidget(self.selection_input)
        self.generate_btn = QtWidgets.QPushButton(self.frame1)
        self.generate_btn.setStyleSheet("")
        self.generate_btn.setObjectName("generate_btn")
        self.verticalLayout_2.addWidget(self.generate_btn)
        self.load_btn.raise_()
        self.generate_btn.raise_()
        self.label.raise_()
        self.selection_input.raise_()
        self.horizontalLayout.addWidget(self.frame1, 0, QtCore.Qt.AlignTop)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Selective Pose Estimation"))
        self.placeholder.setText(_translate(
            "MainWindow", "Load a video to get started"))
        self.load_btn.setText(_translate("MainWindow", "Load Video"))
        self.label.setText(_translate("MainWindow", "Human Selection"))
        self.generate_btn.setText(_translate("MainWindow", "Generate"))

        self.load_btn.clicked.connect(self.openFileBrowser)
        self.generate_btn.clicked.connect(self.generate_pose)

    def generate_pose(self):
        if(self.tracking_complete):
            self.result = self.tracker.track_complete_file()
            gen_masked_video(self.image_folder, self.result,
                             self.selection_input.text())
            output_file_name = images_to_video(self.image_folder, 'masked_output.mp4')
            keypoints_file = generate2dKeypoints(output_file_name)
            process2D_keypoints(keypoints_file)
            generate_3d_inference('masked_output.mp4', True)
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText("Error")
            msg.setInformativeText('Please load a video first')
            msg.setWindowTitle("Error")
            msg.exec_()

    def openFileBrowser(self):
        filepath = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Video", "~", "Video Files (*.mp4 *.avi )")[0]
        if(filepath):
            self.image_folder = video_to_images(filepath)
            self.result = self.tracker.track_first_frame()
            
            if(len(self.result) != 0):
                img_path = self.tracker.get_tracked_image(self.result[0])
                self.pixmap = QtGui.QPixmap(img_path)
                self.placeholder.setPixmap(self.pixmap)
                self.tracking_complete = True
            

 


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Application()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
