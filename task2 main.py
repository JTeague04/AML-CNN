# TEMP
import pandas as pd
import math

# For displaying the results visually
import matplotlib.pyplot as plt
print("matplotlib imported")

# Image and file manipulation
import numpy as np
import cv2
print("numpy+cv2 imported")

# CNN resources
import tensorflow as tf
from tensorflow.keras import layers, models
print("tensorflow imported")

# Functions from worksheet
def visualise_pts(img, pts):
    plt.imshow(img)
    for point in pts:
        plt.plot(point[0] *128, point[1] *128, '+r')
    plt.show()
def euclid_dist(pred_pts, gt_pts):
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts-gt_pts), axis=-1))
def save_as_csv(points, location = "."):
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==5*2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
    np.savetxt(location + '/results_task2.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

# Turn a list of even length into its coordinate pairs
def cardinalise(pts):
    pts =pts[0] # Unpack prediction
    newlist = []
    for i in range(len(pts)//2):
        newlist.append([])
        for j in range(2):
            newlist[-1].append(pts[i*2 +j])
    return newlist

# Preprocessing of training data
def preprocess(images):
    print("")
    update_interval = len(images) //10
    processed_images = []
    
    for counter, image_array in enumerate(images):
        # Apply convolution filter for contrast extraction
        convolution_kernel = np.array([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]])
        img = cv2.filter2D(image_array, -1, convolution_kernel)
        
        # Lower the resolution, to go from 65536 operations to 16384
        img = cv2.resize(img, (128, 128))
        
        # Conform to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Cast image to an appropriate type
        img = img.astype('float32') /255
        
        processed_images.append(img)
        if counter % update_interval == 0:
            print(f"preprocessing {round(counter*100 /len(images), 1)}% complete ")

    # Expand the dimensions by 1 to fit the Tensorflow requirements
    return np.expand_dims( np.array(processed_images), axis =-1)

# Callback method to get loss function at individual points in the training
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losslogs = []
    def on_epoch_end(self, epoch, logs=None):
        self.losslogs.append(logs['loss'])

# Load training and test data
train_data = np.load('face_alignment_training_images.npz', allow_pickle=True)
test_data = np.load('face_alignment_test_images.npz', allow_pickle=True)
print("Data loaded")

# Obtain information from data
i_train_images = train_data['images']
i_train_points = train_data['points']
i_test_images = test_data['images']
print(f"Information gathered, shapes: {i_train_images.shape}, {i_train_points.shape}, {i_test_images.shape}")

# Training variables
epoch_count = 20
batch_amount = 32
validation = 0.1

# Preprocessing, reformatting and storage
train_images = preprocess(i_train_images)
test_images = preprocess(i_test_images)

# Normalising the training points: Set coordinate domain to 0-1, and define as target
im_y = train_images[0].shape[0]
im_x = train_images[0].shape[1]
print(f"\nImage dimensions: {im_x}, {im_y}")

points = i_train_points.astype('float32')
points[..., 0] /= im_x
points[..., 1] /= im_y
output_targets = points.reshape(-1, 10)

print(f"Information processed, shapes: {train_images.shape}, {output_targets.shape}, {test_images.shape}")


loss_callback = CustomCallback()

# Training variables
batch_amount = 64
epoch_count = 50
validation = 0.1

# Define the CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),

    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2, padding='same'),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # '32' meaning 4x 'focus' (smaller features)
    layers.MaxPooling2D(2, 2, padding='same'), # Downsampling, to generalise further
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # '64' to identify shapes and loose organisation
    layers.MaxPooling2D(2, 2, padding='same'), # Further downsampling
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'), # For whole image analysis

    layers.Flatten(),                       # Comparison between layers
    layers.Dense(256, activation='relu'),   # Image component analysis
    layers.Dense(10)                        # 10 outputs: 5 points, (x, y)
    ])

# Compile and train the CNN
model.compile(optimizer='adam', loss='mse')
model.fit(train_images, output_targets,
          epochs=epoch_count, batch_size=batch_amount, validation_split=validation,
          callbacks=[loss_callback])

data = [[loss_callback.losslogs[epoch], -math.log(loss_callback.losslogs[epoch])] for epoch in range(epoch_count)]
df = pd.DataFrame(data, columns=["loss", "-log(loss)"])
df.to_excel(f"{batch_amount} lossdata100.xlsx")

print(f"Final results: {loss_callback.losslogs}")

temp_image = np.expand_dims(test_images[0], axis=0)
print(temp_image.shape)


    

