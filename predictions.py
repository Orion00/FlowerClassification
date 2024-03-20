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
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping


### Random Forest Classifier
# RandomForest, Max Depth 8
# Took 2ish minutes to fit
# Train Accuracy 0.9348534201954397
# Test (Validation) Accuracy 0.487698986975398

# Max Depth 6
# Train Accuracy 0.7958740499457112
# Test (Validation) Accuracy 0.47756874095513746

# Max Depth 7, Min Samples Leaf 5
# Train Accuracy 0.8613825551936302
# Test (Validation) Accuracy 0.5007235890014472

# Max Depth 8, Min Samples Leaf 10
# Train Accuracy 0.8802026782482808
# Test (Validation) Accuracy 0.5094066570188133

# max Depth 15, Min Samples Leaf 10
# Train Accuracy 0.9601882012305465
# Test (Validation) Accuracy 0.5065123010130246

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
X_train, X_test, Y_train, Y_test = train_test_split(fimages, flabels, test_size=0.2, random_state=42)


# %%
rf = RandomForestClassifier(max_depth=15, min_samples_leaf=10,n_jobs=-1,random_state=101)
rf.fit(X_train, Y_train)

# %%
preds_train = rf.predict(X_train)
preds_test = rf.predict(X_test)

# %%
def acc(true,predicted):
    return sum(true == predicted)/len(true)
 #sum(predicted[:,1] == true['target'])/len(true['target'])

# %%
print("Train Accuracy",acc(Y_train,preds_train))
print("Test (Validation) Accuracy",acc(Y_test,preds_test))

# %%

# Submission Code
# %% 
preds_final = rf.predict(test_fimages)

# %%
sub = pd.DataFrame(columns=["ID","Prediction"])
sub['ID'] = test_fimage_paths
sub['ID'] = [i[21:] for i in sub['ID']]
sub['Prediction'] = preds_final
sub.to_csv("RandomForest10Depth.csv", index=False)







#### Neural Net Data 
# 1 Epoch
# Train Accuracy 0.20725588575839443
# Test (Validation) Accuracy 0.20278099652375434

# 5 Epochs (30 minutes, .7 accuracy)

# 10 Epochs, Data augmentation (1+ hour)
# Train Accuracy 0.8324971053647241
# Test (Validation) Accuracy 0.7914252607184241

# %%
images_dir = "Data/training/training/"
labels = pd.read_csv("Data/training_labels.csv")

# add the directory to the filename
labels['ID'] = labels['ID'].apply(lambda x: os.path.join(images_dir, x))

# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.25)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create the training and validation generators
train_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # You can change the size of the image
    batch_size=32, # You can change the batch_size
    class_mode='categorical',  
    subset='training',
    shuffle=False
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory=None,  
    x_col='ID',
    y_col='target',
    target_size=(224, 224), # Should match training size
    batch_size=32, # Should match training
    class_mode='categorical',  
    subset='validation',
    shuffle=False
)

# %%
num_classes=5

my_new_model = Sequential()
#my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(VGG16(include_top=False, pooling='avg', weights='imagenet',input_shape=(224, 224, 3)))
my_new_model.add(Dense(units=num_classes, activation='softmax'))


my_new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
my_new_model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator,
        workers=3,
        validation_steps=1)
# %%
# my_new_model.save('imageNet5EpochTake2.keras')
#my_new_model = keras.models.load_model("imageNet5Epoch.keras")


# %%
train_preds = my_new_model.predict(train_generator)
valid_preds = my_new_model.predict(validation_generator)

### Validation
# %%
train_labels = train_generator.classes
valid_labels = validation_generator.classes
test_preds_labs = np.argmax(train_preds, axis=1)
valid_preds_labs = np.argmax(valid_preds, axis=1)
print("Train Accuracy",accuracy_score(train_labels,test_preds_labs))
print("Test (Validation) Accuracy",accuracy_score(valid_labels,valid_preds_labs))

# %%
train_eval = my_new_model.evaluate(train_generator)
validation_eval = my_new_model.evaluate(validation_generator)
# %%
### Predictions
test_images_dir = "Data/testing/testing/"
test_image_paths = [f for f in os.listdir(test_images_dir)]
sub = pd.DataFrame(columns=["ID","Prediction"])
sub['ID'] = test_image_paths
sub['ID'] = sub['ID'].apply(lambda x: os.path.join(test_images_dir, x))

# Initialize the ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the training and validation generators
test_generator = test_datagen.flow_from_dataframe(
    dataframe=sub,
    directory=None,  
    x_col='ID',
    target_size=(224, 224),
    class_mode=None,  
    validate_filenames=False,
    shuffle=False
)

# %%
test_preds = my_new_model.predict(test_generator)

# %%
class_names = list(train_generator.class_indices.keys())
sub['Prediction'] = np.argmax(test_preds, axis=-1)
sub['Prediction'] = [class_names[label] for label in sub['Prediction']]
sub['ID'] = [i[21:] for i in sub['ID']]

sub.to_csv("VG116Epoch10Try2.csv", index=False)





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




# Orion's 2 Hour experiment

# %%
num_classes=5

my_new_model = Sequential()
#my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(VGG16(include_top=False, pooling='avg', weights='imagenet',input_shape=(224, 224, 3)))
my_new_model.add(Flatten())
my_new_model.add(Dense(512,activation='relu'))
my_new_model.add(Dense(256,activation='relu'))
my_new_model.add(Dense(units=num_classes, activation='softmax')) #Prediction Layer

# %%
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %%
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=2,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)
# %%

my_new_model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        workers=3,
        validation_steps=1,
        callbacks=[early_stopping_callback])

# %%
# my_new_model.save('imageNet10EpochTake2.keras')
my_new_model = keras.models.load_model("imageNet10EpochTake2.keras")

 # %%