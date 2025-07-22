# -IMAGE-CLASSIFICATION-MODEL

COMPANY: CODETECH IT SOLUTIONS

NAME: VADLAMUDI PRASANNA

INTERN ID: CT08DF213

DOMAIN: MACHINE LEARNING

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH

##This project demonstrates a simple yet powerful image classification pipeline using a pre-trained EfficientNetB0 model from TensorFlow Keras. The objective is to classify images into one of the 1,000 ImageNet categories using deep learning, without requiring any custom training. The entire process runs in a Google Colab environment and utilizes popular Python libraries such as TensorFlow, NumPy, OpenCV, PIL (Python Imaging Library), and Matplotlib for visualizing the input image and results.

The program begins by installing necessary dependencies: TensorFlow for deep learning, NumPy for numerical operations, and OpenCV for potential image processing needs. The EfficientNetB0 model is loaded directly from tensorflow.keras.applications, pre-trained on the large-scale ImageNet dataset. EfficientNet is a family of convolutional neural networks developed by Google, known for balancing performance and computational efficiency. The B0 version is the baseline model and provides a good trade-off between speed and accuracy.

After loading the model, the user is prompted to upload an image using the files.upload() feature in Google Colab. The uploaded image is read and opened using the PIL.Image library and converted to RGB to ensure compatibility. It is then resized to 224×224 pixels—the input size expected by EfficientNetB0. The resized image is displayed using Matplotlib to confirm the upload visually.

The image is then preprocessed using preprocess_input, which normalizes pixel values in a way that aligns with the model’s training configuration. The image array is also expanded into a batch of one using np.expand_dims, as TensorFlow models require inputs with batch dimensions. Once prepared, the image batch is passed to the model's predict method, which outputs a probability distribution across 1,000 classes.

The model’s raw output is decoded using the decode_predictions utility, which maps the highest probability indices to human-readable class labels and associated confidence scores. The top 5 predictions are printed, showing the most likely categories the image belongs to, along with the model's confidence in each prediction.

In summary, this program provides an easy-to-use pipeline for image classification using a state-of-the-art convolutional neural network. It eliminates the need for custom data or training and leverages the power of transfer learning by using a pre-trained EfficientNet model. This makes it an ideal starting point for exploring deep learning in image classification tasks.

OUTPUT: 
<img width="972" height="757" alt="Image" src="https://github.com/user-attachments/assets/1955f94d-01b9-4b47-96bb-1e81f3a9a2b2" />
