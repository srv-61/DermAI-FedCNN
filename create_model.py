from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define the model creation function
def create_model(train_data):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the layers of the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(train_data.class_indices), activation='softmax')  # Number of classes
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare the dataset generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    r'D:\SOHAN\7TH SEM\Capstone Project Phase 1\7th SEM\POC\Datasets\Train Data',  # Replace with the actual path to your train data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    r'D:\SOHAN\7TH SEM\Capstone Project Phase 1\7th SEM\POC\Datasets\Train Data',  # Replace with the actual path to your test data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Create the model
model = create_model(train_data)

# Train the model
history = model.fit(
    train_data,            # Training data
    epochs=10,             # Number of epochs to train
    validation_data=test_data,  # Validation data (for evaluation)
    verbose=1              # Verbosity level to show training progress
)

# Save the trained model
model.save('skin_disease_model.h5')  # Save the model to a file

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
