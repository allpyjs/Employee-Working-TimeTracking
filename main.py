import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem, QTableWidget
from PyQt5.QtCore import QTimer, QUrl, Qt
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QBrush, QColor
from ui.ui_main import Ui_MainWindow  # Import the generated UI class

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np
import logging

import utils
import settings

logging.getLogger("yolo").setLevel(logging.ERROR)

BRUSH_GREEN = QBrush(QColor(0, 200, 0))
BRUSH_GRAY = QBrush(QColor(100, 100, 100))


class Station:
    def __init__(self, rect):
        self.rect = tuple(rect)
        self.work = 0
        self.sleep = 0
        self.empty = 0
        self.last_work_frame = -100
        self.last_sleep_frame = -100


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Create an instance of the UI class
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.btn_loadVideo.clicked.connect(self.onLoadVideoClicked)
        self.ui.btn_play_pause.clicked.connect(self.togglePlayPause)
        self.ui.btn_addStation.clicked.connect(self.onAddStation)
        self.ui.btn_removeStation.clicked.connect(self.onRemoveStation)
        self.ui.cb_showDetections.stateChanged.connect(self.refreshFrame)
        self.ui.cb_showStations.stateChanged.connect(self.refreshFrame)
        self.playing = False
        self.ui.label.mousePressEvent = self.onCanvasMouseDown
        self.ui.label.mouseMoveEvent = self.onCanvasMouseMove
        self.ui.label.mouseReleaseEvent = self.onCanvasMouseUp

        # Table Customization
        self.ui.tableWidget.setColumnWidth(0, 70)
        self.ui.tableWidget.setColumnWidth(1, 70)
        self.ui.tableWidget.setColumnWidth(2, 70)
        self.ui.tableWidget.setSelectionMode(QTableWidget.SingleSelection)
        self.ui.tableWidget.setSelectionBehavior(QTableWidget.SelectRows)

        # Init model
        self.model = YOLO(settings.MODEL_PATH)

        # Init class members
        self.cap = None
        self.init()
        self.timer = None

    def init(self):
        self.current_rect = [-1, -1, 0, 0]
        self.stations = []
        self.frameCounter = 0
        while self.ui.tableWidget.rowCount() > 0:
            self.ui.tableWidget.removeRow(0)

    def readStationsFromFile(self, fname):
        try:
            with open(fname, "r") as file:
                for line in file:
                    line = line.strip()
                    parts = line.split(" ")
                    if len(parts) == 4:
                        self.current_rect = [int(i) for i in parts]
                        self.onAddStation()
        except Exception as e:
            print("Error reading file", e)

    def writeStationsToFile(self, fname):
        try:
            with open(fname, "w") as file:
                for station in self.stations:
                    file.write(" ".join(str(i) for i in station.rect) + "\n")
                file.close()
        except Exception as e:
            print("Error writing to file", e)

    def getStationsFileName(self):
        return self.selected_file + ".stations.txt"

    def onAddStation(self):
        n = len(self.stations)
        self.ui.tableWidget.insertRow(n)
        self.ui.tableWidget.setVerticalHeaderItem(n, QTableWidgetItem(str(n + 1)))
        for i in range(3):
            cell = QTableWidgetItem("00:00")
            cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
            cell.setTextAlignment(Qt.AlignCenter)
            self.ui.tableWidget.setItem(n, i, cell)

        self.ui.btn_addStation.setEnabled(False)
        self.stations.append(Station(self.current_rect))
        self.current_rect = [-1, -1, 0, 0]
        self.ui.btn_removeStation.setEnabled(True)
        self.refreshFrame()
        self.writeStationsToFile(self.getStationsFileName())

    def onRemoveStation(self):
        selected = self.ui.tableWidget.selectedRanges()
        if len(selected) > 0:
            selected = selected[0].topRow()
            self.ui.tableWidget.removeRow(selected)
            self.stations.pop(selected)
            self.ui.tableWidget.setVerticalHeaderLabels(
                [str(i + 1) for i in range(0, len(self.stations))]
            )
            if len(self.stations) == 0:
                self.ui.btn_removeStation.setEnabled(False)
            self.refreshFrame()
            self.writeStationsToFile(self.getStationsFileName())

    def togglePlayPause(self):
        self.playing = not self.playing
        self.ui.btn_play_pause.setText("Pause" if self.playing else "Play")

    def onLoadVideoClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        selected_file, _ = QFileDialog.getOpenFileUrl(
            self,
            "Select Video File",
            QUrl(),
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
            options=options,
        )
        print("selected_file", selected_file.toLocalFile())

        if selected_file.toLocalFile():
            self.selected_file = file_url = selected_file.toLocalFile()

            self.ui.btn_play_pause.setEnabled(True)
            self.ui.label_video_url.setText(file_url)

            self.init()
            self.cap = cv2.VideoCapture(file_url)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            self.process_frame()
            self.readStationsFromFile(self.getStationsFileName())

            # timer
            if self.timer:
                self.timer.stop()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self.onTimer)
            self.timer.start(int(1))
        # else:
        #     self.ui.btn_play_pause.setEnabled(False)

    def process_frame(self, move2next=True):
        if not self.cap:
            return

        if move2next:
            success, img = self.cap.read()
            if not success:
                self.cache_img = None
                self.cap = None
                self.togglePlayPause()
                self.ui.btn_play_pause.setEnabled(False)
                return
            self.cache_img = img.copy()
            self.frameCounter += 1
            if self.frameCounter % self.fps == 0:
                self.refreshTable()
        else:
            if type(self.cache_img) == "<class 'NoneType'>":
                return
            img = self.cache_img.copy()

        results = self.model(img, stream=True)

        # iterate over the detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentclass = settings.CLASSES[cls]
                if (
                    currentclass in ["work", "sleep"]
                    and conf > settings.DETECTION_THRESHOLD
                ):
                    if self.ui.cb_showDetections.isChecked():
                        cvzone.cornerRect(img, (x1, y1, w, h), l=15)
                        cvzone.putTextRect(
                            img,
                            f"{settings.CLASSES[cls]}",
                            (max(0, x1), max(35, y1)),
                            scale=2,
                            thickness=2,
                            offset=3,
                        )
                    if move2next:
                        # check if detection belongs to station
                        for station in self.stations:
                            intersection = utils.get_rect_intersection(
                                station.rect, (x1, y1, w, h)
                            )
                            if (
                                utils.get_rect_area(intersection)
                                / utils.get_rect_area(station.rect)
                                > settings.INTERSECTION_THRESHOLD
                            ):
                                if currentclass == "work":
                                    station.last_work_frame = self.frameCounter
                                elif currentclass == "sleep":
                                    station.last_sleep_frame = self.frameCounter

        for i, station in enumerate(self.stations):
            rt = station.rect
            # check for the human state of each station
            if move2next:
                if (
                    station.last_work_frame
                    >= self.frameCounter - settings.PADDING_MS * self.fps / 1000
                ):
                    station.work += 1
                elif (
                    station.last_sleep_frame
                    >= self.frameCounter - settings.PADDING_MS * self.fps / 1000
                ):
                    station.sleep += 1
                else:
                    station.empty += 1

            # draw stations
            if self.ui.cb_showStations.isChecked():
                cvzone.cornerRect(img, rt, l=8, colorR=(0, 255, 0), colorC=(0, 255, 0))
                cvzone.putTextRect(
                    img,
                    f"Station #{i+1}",
                    (max(0, rt[0]), max(35, rt[1])),
                    scale=2,
                    thickness=2,
                    offset=3,
                    colorR=(0, 120, 0),
                )

        if not self.playing:
            self.draw_current_rect(img)
        self.show_image(img)

    def draw_current_rect(self, img):
        cv2.rectangle(
            img,
            utils.normalizeRect(self.current_rect),
            (0, 255, 0),
            thickness=2,
        )

    def show_image(self, img_np):
        self.image_size = [self.ui.label.width() - 2, self.ui.label.height() - 2]

        img_np = cv2.resize(img_np, self.image_size)
        bytesPerLine = 3 * self.image_size[0]
        frame = QImage(
            img_np.data,
            self.image_size[0],
            self.image_size[1],
            bytesPerLine,
            QImage.Format_RGB888,
        ).rgbSwapped()
        pix = QPixmap.fromImage(frame)
        self.ui.label.setPixmap(pix)

    def onDetailImageClick(self, event, idx):
        print(idx)

    def onTimer(self):
        if self.playing:
            self.process_frame()

    def refreshFrame(self):
        self.process_frame(False)

    def onCanvasMouseDown(self, event: QMouseEvent):
        if not self.cap or self.playing:
            return

        self.setMouseTracking(True)
        self.current_rect[0], self.current_rect[1] = self.canvasPtToImagePt(
            (event.x(), event.y())
        )

    def onCanvasMouseMove(self, event: QMouseEvent):
        if self.hasMouseTracking():
            ex, ey = self.canvasPtToImagePt((event.x(), event.y()))
            self.current_rect[2], self.current_rect[3] = (
                ex - self.current_rect[0],
                ey - self.current_rect[1],
            )
            self.refreshFrame()

    def onCanvasMouseUp(self, event: QMouseEvent):
        self.current_rect = utils.normalizeRect(self.current_rect)
        self.setMouseTracking(False)
        self.ui.btn_addStation.setEnabled(True)

    def canvasPtToImagePt(self, point):
        return (
            point[0] * self.cache_img.shape[1] // self.ui.label.width(),
            point[1] * self.cache_img.shape[0] // self.ui.label.height(),
        )

    def refreshTable(self):
        assert self.ui.tableWidget.rowCount() == len(self.stations)

        for r, station in enumerate(self.stations):
            for i, v in enumerate([station.work, station.sleep, station.empty]):
                cell = self.ui.tableWidget.item(r, i)
                newtext = utils.formatTime(v // self.fps)
                if newtext != cell.text():
                    cell.setText(newtext)
                    cell.setForeground(BRUSH_GREEN)
                else:
                    cell.setForeground(BRUSH_GRAY)


# Main function to start the application
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and show the main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
