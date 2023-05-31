# Diabetic_Retinopathy_Detection

The provided code is an image classification script implemented in Python using TensorFlow. It uses various pre-trained models to classify images into different categories.

The script begins by importing the necessary libraries such as numpy, pandas, pathlib, os.path, matplotlib, IPython.display, seaborn, and tensorflow. These libraries provide functions and classes for data manipulation, file handling, visualization, and machine learning.

Next, the code defines a utility function called printmd(), which allows printing text with Markdown formatting. This function is used to display formatted text in the output.

The script then sets the path to the image directory and retrieves the filepaths and labels of the images using the Path and os.path modules. The filepaths and labels are stored in pandas Series objects, and a DataFrame called image_df is created by concatenating these Series.

To ensure randomness in the dataset, the DataFrame is shuffled using the sample() function and the index is reset. The resulting shuffled DataFrame is displayed to verify the changes.

Using matplotlib, the code creates a subplot grid of size 3x4 and displays a sample of 12 images from the dataset. Each image is shown along with its corresponding label.

The code then calculates and visualizes the number of pictures for each category using seaborn. A bar plot is created to represent the counts of different categories in the dataset.

The script defines a function called create_gen() that uses the ImageDataGenerator class from tensorflow.keras.preprocessing.image module to load the images with data augmentation. It creates generator objects for training, validation, and testing images, and applies preprocessing functions to the images.

A function named get_model() is defined to load a pre-trained model. It takes the model architecture as a parameter and returns a compiled model. In this case, the model architecture is specified using functions from tensorflow.keras.applications module.

The script splits the dataset into training and testing sets using the train_test_split() function from sklearn.model_selection module.

A dictionary named models is defined to store different pre-trained models along with their performance metrics. Each model is initialized and compiled using the get_model() function.

The script then trains each model for one epoch and measures the training time, validation accuracy, and training accuracy. The results are stored in the models dictionary.

The performance metrics of the models are organized into a DataFrame called df_results, which is sorted based on validation accuracy in descending order.

Using seaborn, a bar plot is created to visualize the training accuracy of each model after one epoch.

Next, a specific pre-trained model, InceptionResNetV2, is selected, and additional layers are added on top. The model is compiled and trained for multiple epochs using the training and validation data.

The training and validation accuracy and loss are plotted using matplotlib to visualize the model's performance.

Finally, the model is evaluated on the test data, and the test loss, accuracy, and classification report are displayed. Additionally, a confusion matrix is generated using seaborn to visualize the performance of the model in classifying different categories.

The script also includes an alternative approach where the dataset is reduced to two categories, 'No_DR' and 'DR', and a binary classification model is trained using the InceptionResNetV2 pre-trained model. The training and evaluation process for the binary classification model is similar to the multi-class classification model.

In conclusion, the provided code demonstrates the process of image classification using pre-trained models in TensorFlow. It involves data preprocessing, model creation and compilation, training, evaluation, and performance visualization.
