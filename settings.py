PADDING_MS = 500  # Duration in milliseconds for which a single frame detection influences the overall detection interval
INTERSECTION_THRESHOLD = 0.6  # Percentage of intersection area to the station area above which a person is considered to be in that station

# yolo model settings
MODEL_PATH = "./model/stationsBest.pt"
CLASSES = ["sleep", "stand", "work", "empty", "monitor"]
DETECTION_THRESHOLD = (
    0.3  # Probability threshold above which a detection is considered valid
)
