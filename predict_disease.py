from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = models.load_model('skin_disease_model.h5')

def predict_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = list(train_data.class_indices.keys())  # Get the class labels
    predicted_label = class_labels[predicted_class[0]]
    
    return f"Predicted Disease: {predicted_label}, Confidence: {np.max(prediction) * 100:.2f}%"
