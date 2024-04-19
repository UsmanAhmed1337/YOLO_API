from ultralytics import YOLO
import cv2
from math import ceil
import numpy as np

class_names = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

def load_model(model_path):
    model = YOLO(model_path)
    return model

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer

def inference_image(model, image_path):
    results = model.predict(image_path)
    #bounding_boxes = []
    for result in results:
        #for i in range(len(result.boxes)):
        #    bounding_box = {
        #        "class":class_names[int(result.boxes.cls[i].numpy())],
        #        "coordinates":result.boxes.xyxy[i].numpy().tolist()
        #    }
        #    bounding_boxes.append(bounding_box)
        result.save('img.jpg')
    return 



def inference_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    writer = create_video_writer(cap, "ConstructionSiteSafetyOutput.mp4")

    my_color = (0, 0, 255)
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                current_class = class_names[cls]
                print(current_class)
                if conf > 0.5:
                    if current_class =='NO-Hardhat' or current_class =='NO-Safety Vest' or current_class == "NO-Mask":
                        my_color = (0, 0,255)
                    elif current_class =='Hardhat' or current_class =='Safety Vest' or current_class == "Mask":
                        my_color = (0,255,0)
                    else:
                        my_color = (255, 0, 0)


                    image = cv2.putText(img, f'{class_names[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(img, (x1, y1), (x2, y2), my_color, 3)

        cv2.imshow("Image", img)
        writer.write(img)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()