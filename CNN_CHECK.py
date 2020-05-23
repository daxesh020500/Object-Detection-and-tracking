import cv2
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
CATAGORIES = ['LEFT_MARG','RIGHT_MARG','SLANT_ASC','SLANT-DESC']

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath)
    plt.imshow(img_array)
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = keras.models.load_model('MY_CNN.h5')


prediction = model.predict([prepare(r'C:\Users\daxesh\Downloads\Hiren.jpg')])

print(CATAGORIES[(prediction[0])])
plt.title(CATAGORIES[(prediction[0])])

lbl = str(CATAGORIES[int(prediction)])
if lbl == 'LEFT_MARG':
      print('\n > Courageous :')
      print('\n > Insecure and devotes oneself completely')
elif lbl == 'RIGHT_MARG':
      print('\n > Avoids future and a reserved person :\t')
elif lbl == 'SLANT_ASC':
      print('\n > Optimistic :')
elif lbl == 'SLANT_DESC':
      print('\n > Pessimistic :')

plt.show()