import cv2
import numpy as np
import random

class Button:
    def __init__(self, label, polygon):
        self.label = label
        self.polygon = polygon
        self.is_clicked = False

    def draw(self, frame):
        if self.is_clicked:
            cv2.fillPoly(frame, [self.polygon], (0, 200, 0))
        else:
            cv2.fillPoly(frame, [self.polygon], (0, 0, 200))
        cv2.putText(frame, self.label, (self.polygon[0][0][0] + 10, self.polygon[0][0][1] + 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    def check_click(self, x, y):
        is_inside = cv2.pointPolygonTest(self.polygon, (x, y), False)
        if is_inside > 0:
            self.is_clicked = not self.is_clicked

net = cv2.dnn.readNet('dnn/yolov4-tiny.weights', 'dnn/yolov4-tiny.cfg')

# Read class names from file
classes = []
with open('dnn/coco.names', 'r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Generate random class names for buttons
random_classes = random.sample(classes, 5)

# Initialize the buttons
buttons = []
button_spacing = 20
button_height = 70

for i, class_name in enumerate(random_classes):
    button_x = button_spacing
    button_y = button_spacing + (button_height + button_spacing) * i
    button_width, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
    button_polygon = np.array([[
        (button_x, button_y),
        (button_x + button_width + 20, button_y),
        (button_x + button_width + 20, button_y + button_height),
        (button_x, button_y + button_height)
    ]])
    buttons.append(Button(class_name, button_polygon))

button_states = {class_name: False for class_name in random_classes}

def click_button(event, x, y, flags, params):
    global button_states, model
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        for button in buttons:
            button.check_click(x, y)
            if button.is_clicked:
                button_states[button.label] = True
            else:
                button_states[button.label] = False

        print('Button states:', button_states)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(320, 320), scale=1 / 255)

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_button)

while True:
    ret, frame = cap.read()

    for button in buttons:
        button.draw(frame)

    selected_class = None
    for button in buttons:
        if button_states[button.label]:
            selected_class = button.label

    if selected_class is not None:
        detected_classes, scores, bboxes = model.detect(frame, confThreshold=0.3)
        if detected_classes is not None:
            for class_id, score, bbox in zip(detected_classes, scores, bboxes):
                if button_states[selected_class]:
                    if classes[class_id.item(0)] == selected_class:
                        x, y, w, h = bbox
                        class_name = classes[class_id.item(0)]
                        cv2.putText(frame, str(class_name), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
                        text_width, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
                        rect_width = max(text_width + 40, w)
                        cv2.rectangle(frame, (x, y), (x + rect_width, y + h), (200, 0, 50), 3)
    else:
        # No button clicked, display all detected class names
        detected_classes, scores, bboxes = model.detect(frame, confThreshold=0.3)
        if detected_classes is not None:
            for class_id, score, bbox in zip(detected_classes, scores, bboxes):
                class_name = classes[class_id.item(0)]
                cv2.putText(frame, str(class_name), (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 2)
                text_width, _ = cv2.getTextSize(str(class_name), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
                rect_width = max(text_width + 40, bbox[2])
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + rect_width, bbox[1] + bbox[3]), (200, 0, 50), 3)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
