import tensorflow
import numpy as np
import cv2
import os
import time

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model("keras_model.h5")

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

output_directory = "captured_images/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

image_captured = False

capture_delay = 10
last_capture_time = time.time() - capture_delay

while True:
    success, img = cap.read()

    faces = faceCascade.detectMultiScale(img, 1.1, 4)

    for x, y, w, h in faces:
        crop_img = img[y : y + h, x : x + w]
        crop_img = cv2.resize(crop_img, (224, 224))

        normalized_image_array = (crop_img.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        print(prediction)

        if prediction[0][0] > prediction[0][1]:
            label = "Masker"
            accuracy = prediction[0][0] * 100
            color = (0, 255, 0)
            image_captured = False
        else:
            label = "Tanpa Masker"
            accuracy = prediction[0][1] * 100
            color = (0, 0, 255)

            current_time = time.time()
            if not image_captured and current_time - last_capture_time >= capture_delay:
                filename = f"captured_images/captured_{time.time()}.jpg"
                cv2.imwrite(filename, crop_img)
                print(f"Gambar {filename} telah disimpan.")

                image_captured = True
                last_capture_time = current_time
                time.sleep(capture_delay)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img,
            f"{label}: {accuracy:.2f}%",
            (x, y - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            color,
            1,
        )
        cv2.imshow("Hasil", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
