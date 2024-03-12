# %%
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from keras.applications import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.models
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from sklearn.model_selection import train_test_split

### READ IN DATA
# Neural Net Data
# %%
images_dir = "Data/training/training/"
labels = pd.read_csv("Data/training_labels.csv")

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


### PLOT FIRST FEW IMAGES
# %%
## Plot a few of the images
import matplotlib.pyplot as plt

# Fetch a batch of images and their labels
images, labels = next(train_generator)

# Number of images to show
num_images = 8

plt.figure(figsize=(20, 10))
for i in range(num_images):
    ax = plt.subplot(2, 4, i + 1)
    plt.imshow(images[i])
    # The label for current image
    label_index = labels[i].argmax()  # Convert one-hot encoding to index
    label = list(train_generator.class_indices.keys())[label_index]  # Get label name from index
    plt.title(label)
    plt.axis('off')
plt.show()



### NN Train Classifier
# %%
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Optionally, unfreeze the top N layers
N = 5
for layer in base_model.layers[-N:]:
    layer.trainable = True

# Add custom layers on top
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model_transfer = Model(inputs=base_model.input, outputs=predictions)

# Compile with a smaller learning rate
model_transfer.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# %%
model_transfer.fit(train_generator, validation_data=validation_generator, epochs=1)

### SAVE/LOAD THE MODEL
# %%
# model_transfer.save('imageNet2Epoch.keras')
# model_transfer = keras.models.load_model("imageNet2Epoch.keras")

### Prediction
# %%
preds = model_transfer.predict(validation_generator)

### Validation
# %%
accuracy_score(labels,preds)

# %%
preds = model_transfer.predict_generator(validation_generator, steps=len(validation_generator))


# Get the class index with the highest probability for each image
predicted_classes = np.argmax(preds, axis=1)

# Now you have an array of predicted class indices for each image
print(predicted_classes)



### Random Forest Classifier
# RandomForest, Max Depth 8
# Took 10(?) minutes to fit
# Train Accuracy 0.9348534201954397
# Test (Validation) Accuracy 0.487698986975398

# Max Depth 6
# Train Accuracy 0.7864639884183858
# Test (Validation) Accuracy 0.5036179450072359

# Flat Data
# %%
image_dir = 'Data/training/training'
test_image_dir = 'Data/testing/testing'

# Load labels
flabels_csv = 'training_labels.csv'
flabels_df = pd.read_csv('Data/'+flabels_csv)
flabels = flabels_df['target'].values  

# Preprocess images
fimage_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
# You can resize the image to different dimensions
fimages = np.array([np.array(Image.open(img).resize((128, 128))).flatten() for img in fimage_paths])

# Preprocess test images
test_fimage_paths = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
test_fimages = np.array([np.array(Image.open(img).resize((128, 128))).flatten() for img in test_fimage_paths])

# Ensure images and labels are aligned, assuming filenames and labels are in the same order
assert len(fimages) == len(flabels), "The number of images and labels do not match."

# Split the dataset into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(fimages, labels, test_size=0.2, random_state=42)


# %%
rf = RandomForestClassifier(max_depth=6)
rf.fit(X_train, Y_train)

# %%
preds_train = rf.predict(X_train)
preds_test = rf.predict(X_test)

# %%

# %%
def acc(true,predicted):
    return sum(predicted[:,1] == true['target'])/len(true['target'])

# %%
print("Train Accuracy",acc(Y_train,preds_train))
print("Test (Validation) Accuracy",acc(Y_test,preds_test))

# Submission Code
# %% 
preds_final = rf.predict(test_fimages)

# %%
sub = pd.DataFrame(preds_final, columns=["ID","Prediction"])
#sub['ID'] = [i[23:] for i in sub['ID']]
sub.to_csv("RandomForest6Depth.csv", index=False)
