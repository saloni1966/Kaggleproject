import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential

labels = pd.read_csv('labels.csv')
print(labels.head())

#preprocessing the data
def preprocess_image(img_path,target_size = (224,224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img , target_size)
    img = img /255.0 #naromalizing pixel values 
    return img

#ex 
img_path = '/Users/salonisharma/Documents/syon/blob.jpg'
img = preprocess_image(img_path)
plt.imshow(img)
plt.show()

#split the data

train_df , val_df = train_test_split(labels, test_size = 0.2, stratify = labels['breed'])

train_datagen = ImageDataGenerator(rotation_range = 20, width_shoft_range = 0.2 , height_shift_range = 0.2 , shear_range = 0.2 , zoom_range = 0.2 , horizontal_flip = True, fill_mode = 'nearest')

val_datagen = ImageDataGenerator()



train_generator = train_datagen.flow_from_dataframe(train_df, directory = 'train' , x_col = 'id', y_col = 'breed' , target_size = (224,224), class_mode = 'categorical')
val_generator = val_datagen.flow_from_dataframe(val_df,directory = 'train', x_col = 'id', y_col = 'breed' , target_size = (224,224), class_mode = 'categorical')
#load the VGG16 model

base_model = VGG16(weights = 'imagenet', include_top = False , input_shape = (224,224,3))
for layer in base_model.layers :
    layer.trainable = False


from tensorflow.keras.layers import Flatten,Dense,Dropout

x = base_model.output
x = Flatten()(x)
x = Dense(512 , sctivation = 'relu')(x)
x = Dropout(0.5)(x)
preditions = Dense(120 , activation = 'softmax')(x) # 120 dog breeds

from tensorflow.keras.models import Model

model = Model(input = base_model.input , output = preditions )

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics= ['accuracy'])
history = model.fit(train_generator,epochs = 20,validation_data = val_generator,step_per_epochs = train_generator.samples // train_generator.batch_size,validation_steps = val_generator.samples//val_generator.batch_size)

#evaluate the model

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label = 'train_accuracy')
plt.plot(history.history['val_accuracy'],label = 'val_accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

val_loss , val_accuracy = model.evaluate(val_generator)
print(f'Validation Accuracy : {val_accuracy * 100:.2f}')

#Save and export the model
model.save('dog_breed_recognition_model.h5')

#make predictions on new data
