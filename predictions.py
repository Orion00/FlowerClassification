# %%
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# %%
images_dir = "training/"
labels = pd.read_csv("training_labels.csv")

# add the directory to the filename
labels['ID'] = labels['ID'].apply(lambda x: os.path.join(images_dir, x))

# Initialize the ImageDataGenerator
# You can change the size of the validation split (0.25 is 25% of data used as validation set)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

# Create the training and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # You can change the size of the image
    batch_size=32, # You can change the batch_size
    class_mode='categorical',  
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # Should match training size
    batch_size=32, # Should match training
    class_mode='categorical',  
    subset='validation'
)

# %%