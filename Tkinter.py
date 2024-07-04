import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# 모델 로드
model_path = 'C:/Users/AISW-203-109/tkinter_0628/images/saved_model.h5'
model = tf.keras.models.load_model(model_path)

# 모델에 필요한 이미지 크기 (224x224로 변경)
IMG_SIZE = (224, 224)

# 클래스 이름 설정
class_names = ["Lilly", "Lotus", "Orchid", "Sunflower", "Tulip"]

# 입력 전처리 함수
def preprocess_input(image_array):
    return tf.keras.applications.efficientnet.preprocess_input(image_array)

def show_image():
    global filename, imported_image
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),
                                          title="이미지 파일 선택",
                                          filetypes=(("PNG 파일", "*.png"),
                                                     ("JPG 파일", "*.jpg"),
                                                     ("JPEG 파일", "*.jpeg"),
                                                     ("모든 파일", "*.*")))
    if filename:
        img = Image.open(filename)
        img = img.resize((380, 320), Image.LANCZOS)  # 프레임에 맞게 이미지 크기 조정
        imported_image = ImageTk.PhotoImage(img)
        lbl.configure(image=imported_image)
        lbl.image = imported_image
        predict_image(filename)

def predict_image(file_path):
    try:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # 배치 축 생성
        img_array = preprocess_input(img_array)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index]

        result_label.config(text=f"Prediction: {predicted_class}  (Confidence: {confidence:.2f})")
    except Exception as e:
        messagebox.showerror("Error", f"예측 중 오류가 발생했습니다: {e}")

# Tkinter 창 생성
window = tk.Tk()
window.title("5가지 꽃 분류 예측")
window.geometry("900x500+100+100")
window.configure(bg="white")

# 아이콘 설정
img_icon = Image.open("C:/Users/AISW-203-109/tkinter_0628/images/tulips.jpg")
img_icon = img_icon.resize((70, 100), Image.LANCZOS)
photo_img = ImageTk.PhotoImage(img_icon)
window.iconphoto(False, photo_img)

# 제목 레이블
tk.Label(window, image=photo_img, bg="#fff").place(x=10, y=10)
tk.Label(window, text="5가지 꽃 분류 예측", font="arial 25 bold", fg="yellow", bg="black").place(x=90, y=50)

# 이미지 선택 프레임
selectimage = tk.Frame(window, width=400, height=400, bg="#d6dee5")
selectimage.place(x=10, y=120)
f = tk.Frame(selectimage, bg="black", width=380, height=320)
f.place(x=10, y=10)
lbl = tk.Label(f, bg="black")
lbl.place(x=0, y=0)
tk.Button(selectimage, text="이미지 선택", width=12, height=2, font="arial 14 bold", command=show_image).place(x=10, y=340)

# 결과 표시 레이블
result_label = tk.Label(window, text="Prediction: ", font="arial 20 bold", bg="white")
result_label.place(x=450, y=200)

# Tkinter 창 실행
window.mainloop()

