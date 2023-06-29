import cv2
import torch
import pytesseract
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='best.pt')  # Replace with the path to your YOLOv5 weights file

# Set device (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Load video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Run inference
    results = model(img)

    # Process predictions
    for detection in results.pred[0]:
        class_name = model.names[int(detection[-1])]
        confidence = detection[4]
        bbox = detection[:4]

        # Draw bounding box on the frame
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {confidence:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Extract the number plate region
        number_plate_img = frame[y1:y2, x1:x2]

        # Preprocess number plate image
        gray = cv2.cvtColor(number_plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Perform OCR on the preprocessed number plate image
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        number_plate_text = pytesseract.image_to_string(threshold, config=custom_config)

        # Apply additional text processing to improve OCR accuracy
        number_plate_text = number_plate_text.strip().replace('\n', '').replace('\r', '')

        # Display the detected number plate text
        print("Detected number plate:", number_plate_text)

    # Display the frame
    cv2.imshow('YOLOv5 Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):  # Press q to exit
        cv2.destroyAllWindows()
        break

# Clean up
cap.release()
