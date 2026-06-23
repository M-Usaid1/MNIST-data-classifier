# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model

# class MNISTPredictor:
#     def __init__(self, model_path="mnist.pkl"):
#         self.model = self.load_model(model_path)
    
#     def load_model(self, model_path):
#         """Load the trained model from pickle file"""
#         try:
#             from tensorflow.keras.models import load_model

#             model = load_model(mnist_ann.keras)
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             raise
    
#     def preprocess_image(self, image_array):
#         """Preprocess the image for prediction"""
#         # Reshape to (1, 784) and normalize
#         image_array = np.array(image_array).reshape(1, 784)
#         image_array = image_array.astype("float32") / 255.0
#         return image_array
    
#     def predict(self, image_array):
#         """Make prediction on preprocessed image"""
#         processed = self.preprocess_image(image_array)
#         predictions = self.model.predict(processed, verbose=0)
#         predicted_class = np.argmax(predictions[0])
#         confidence = float(np.max(predictions[0]))
#         return {
#             "prediction": int(predicted_class),
#             "confidence": confidence,
#             "all_probabilities": predictions[0].tolist()
#         }

# # Create a singleton instance
# predictor = MNISTPredictor()
# import numpy as np
# from tensorflow.keras.models import load_model


# class MNISTPredictor:

#     def __init__(self, model_path="mnist_ann.h5"):

#         self.model = self.load_model(model_path)

#     def load_model(self, model_path):

#         try:

#             model = load_model(model_path)

#             print("Model loaded successfully")

#             return model

#         except Exception as e:

#             print(f"Error loading model: {e}")

#             raise

#     def preprocess_image(self, image_array):

#         image_array = np.array(image_array)

#         image_array = image_array.reshape(1,784)

#         image_array = image_array.astype("float32")

#         image_array = image_array/255.0

#         return image_array

#     def predict(self,image_array):

#         processed = self.preprocess_image(image_array)

#         predictions = self.model.predict(processed, verbose=0)

#         predicted_class = np.argmax(predictions[0])

#         confidence = float(np.max(predictions[0]))

#         return {

#             "prediction": int(predicted_class),

#             "confidence": confidence,

#             "all_probabilities": predictions[0].tolist()

#         }


# predictor = MNISTPredictor()
import numpy as np
from tensorflow.keras.models import load_model

class MNISTPredictor:
    def __init__(self, model_path="mnist_model.h5"):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            model = load_model(model_path)
            print("✅ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_array):
        image_array = np.array(image_array)
        image_array = image_array.reshape(1, 784)
        image_array = image_array.astype("float32") / 255.0
        return image_array
    
    def predict(self, image_array):
        processed = self.preprocess_image(image_array)
        predictions = self.model.predict(processed, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        return {
            "prediction": int(predicted_class),
            "confidence": confidence,
            "all_probabilities": predictions[0].tolist()
        }

predictor = MNISTPredictor()