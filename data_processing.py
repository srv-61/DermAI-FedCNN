from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training data generator (with augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to [0, 1]
    rotation_range=20,        # Randomly rotate images by 20 degrees
    width_shift_range=0.2,    # Randomly shift images horizontally
    height_shift_range=0.2,   # Randomly shift images vertically
    shear_range=0.2,          # Randomly shear images
    zoom_range=0.2,           # Randomly zoom in on images
    horizontal_flip=True,     # Randomly flip images horizontally
    fill_mode='nearest'       # Fill missing pixels after transformations
)

train_data = train_datagen.flow_from_directory(
    'path_to_train_data',    # Path to your training dataset
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,           # Batch size for training
    class_mode='categorical' # Use categorical labels (one-hot encoded)
)

# Validation data generator (without augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    'path_to_test_data',     # Path to your testing dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
