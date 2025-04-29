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

# Training variables
epoch_count = 20
batch_amount = 32
validation = 0.1

# Functions from worksheet
def visualise_pts(img, pts):
    plt.imshow(img)
    for point in pts:
        plt.plot(point[0] *128, point[1] *128, '+r')
    plt.show()

def euclid_dist(pred_pts, gt_pts):
    pred_pts = np.reshape(pred_pds, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts-gt_pts), axis=-1))

def save_as_csv(points, location = "."):
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==5*2, 'wrong number of points provided. There should be 5 points with 2 values (x,y) per point'
    np.savetxt(location + '/results_task2.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

# Load training and test data
train_data = np.load('face_alignment_training_images.npz', allow_pickle=True)
test_data = np.load('face_alignment_test_images.npz', allow_pickle=True)
print("Data loaded")

# Obtain information from data
itrain_images = train_data['images']
itrain_points = train_data['points']
itest_images = test_data['images']
print(f"Information gathered, shapes: {itrain_images.shape}, {itrain_points.shape}, {itest_images.shape}")

# Preprocessing of training data
def preprocess(images):
    print("")
    update_interval = len(images) //10
    processed_images = []
    
    for counter, image_array in enumerate(images):
        # Lower the resolution, to go from 65536 operations to 16384
        img = cv2.resize(image_array, (128, 128))
        
        # Cast image to an appropriate type
        img = img.astype('float32') /255

        # Conform to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply convolution filter for contrast extraction
        convolution_kernel = np.array([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]])
        gray = cv2.filter2D(gray, -1, convolution_kernel)
        
        processed_images.append(gray)
        if counter % update_interval == 0:
            print(f"preprocessing {round(counter*100 /len(images), 1)}% complete ")
        
    return processed_images

# Preprocessing, reformatting and storage
p_train_images = preprocess(itrain_images)
train_images = np.expand_dims(np.array(p_train_images), axis =-1)

p_test_images = preprocess(itest_images)
test_images = np.expand_dims(np.array(p_test_images), axis =-1)

# Normalising the training points: Set coordinate domain to 0-1, and define as target
im_y = train_images[0].shape[0]
im_x = train_images[0].shape[1]
print(f"\nImage dimensions: {im_x}, {im_y}")

points = itrain_points.astype('float32')
points[..., 0] /= im_x
points[..., 1] /= im_y
output_targets = points.reshape(-1, 10)

print(f"Information processed, shapes: {train_images.shape}, {output_targets.shape}, {test_images.shape}")

# Define the CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 1)),
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
model.fit(train_images, output_targets, epochs=epoch_count, batch_size=batch_amount, validation_split=validation)

print(f"Train image dimensions: {train_images.shape}")
print(f"Test image dimensions:  {test_images.shape}")

# Turn a list of even length into its coordinate pairs
def cardinalise(pts):
    pts =pts[0] # Unpack prediction
    newlist = []
    for i in range(len(pts)//2):
        newlist.append([])
        for j in range(2):
            newlist[-1].append(pts[i*2 +j])
    return newlist

temp_image = np.expand_dims(test_images[0], axis=0)

print(f"shape of test images: {test_images.shape} shape of image: {temp_image.shape}")
visualise_pts(itest_images[0], cardinalise(model.predict(temp_image)) )
input("\n\n\n The program has now completed, press ENTER to end.")

    

