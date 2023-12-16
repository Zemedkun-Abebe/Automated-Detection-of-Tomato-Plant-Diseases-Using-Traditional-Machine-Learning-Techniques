import streamlit as st
import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import matplotlib.pyplot as plt

# Load the trained Random Forest model
model_filename = 'random_forest_model.joblib'
random_forest = joblib.load(model_filename)

# Function to Create a new dataframe
def create_empty_df():
    df = pd.DataFrame()
    df['area'] = None
    df['perimeter'] = None
    df['red_mean'] = None
    df['green_mean'] = None
    df['blue_mean'] = None
    df['f1'] = None
    df['f2'] = None
    df['red_std'] = None
    df['green_std'] = None
    df['blue_std'] = None
    df['f4'] = None
    df['f5'] = None
    df['f6'] = None
    df['f7'] = None
    df['f8'] = None
    df['label'] = None
    return df

# Function to extract the features
def feature_extractor(image):
    '''
    input params: 
    image : NumPy array representing the image

    Output params:
    l : Feature vector
    '''

    try:
        main_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        return "Invalid"

    # Preprocessing
    gs = cv2.cvtColor(main_img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gs, (25, 25), 0)
    ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((25, 25), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    # Shape features
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    if area == 0:
        return "Invalid"
    perimeter = cv2.arcLength(cnt, True)

    current_frame = main_img
    filtered_image = closing / 255

    # Elementwise Multiplication of range bounded filtered_image with current_frame
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0] = np.multiply(
        current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 0], filtered_image
    )  # B channel
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1] = np.multiply(
        current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 1], filtered_image
    )  # G channel
    current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2] = np.multiply(
        current_frame[0:current_frame.shape[0], 0:current_frame.shape[1], 2], filtered_image
    )  # R channel

    img = current_frame

    # Color features
    red_channel = img[:, :, 0]
    green_channel = img[:, :, 1]
    blue_channel = img[:, :, 2]

    red_mean = np.mean(red_channel)
    green_mean = np.mean(green_channel)
    blue_mean = np.mean(blue_channel)

    # standard deviation for color feature from the image.
    red_std = np.std(red_channel)
    green_std = np.std(green_channel)
    blue_std = np.std(blue_channel)

    # amt. of green color in the image
    gr = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    boundaries = [([30, 0, 0], [70, 255, 255])]
    for (lower, upper) in boundaries:
        mask = cv2.inRange(gr, (36, 0, 0), (70, 255, 255))
        ratio_green = cv2.countNonZero(mask) / (img.size / 3)
        f1 = np.round(ratio_green, 2)
    # amt. of non-green part of the image
    f2 = 1 - f1

    # Texture features using grey level co-occurrence matrix
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = greycomatrix(img, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    # with the help of glcm find the contrast
    contrast = greycoprops(g, 'contrast')
    f4 = contrast[0][0] + contrast[0][1] + contrast[0][2] + contrast[0][3]

    # with the help of glcm find the dissimilarity
    dissimilarity = greycoprops(g, prop='dissimilarity')
    f5 = dissimilarity[0][0] + dissimilarity[0][1] + dissimilarity[0][2] + dissimilarity[0][3]

    # with the help of glcm find the homogeneity
    homogeneity = greycoprops(g, prop='homogeneity')
    f6 = homogeneity[0][0] + homogeneity[0][1] + homogeneity[0][2] + homogeneity[0][3]

    energy = greycoprops(g, prop='energy')
    f7 = energy[0][0] + energy[0][1] + energy[0][2] + energy[0][3]

    correlation = greycoprops(g, prop='correlation')
    f8 = correlation[0][0] + correlation[0][1] + correlation[0][2] + correlation[0][3]

    l = [area, perimeter, red_mean, green_mean, blue_mean,
         f1, f2, red_std, green_std, blue_std,
         f4, f5, f6, f7, f8]
    return l

# Function to visualize the provided Image
def preprocess_and_visualize(image_resized):
    # Convert the image to grayscale
    gs = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gs, (5, 5), 0)

    # Edge detection using Canny
    edges_canny = cv2.Canny(blurred, 50, 150)

    # Edge detection using Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    edges_laplacian = np.uint8(np.absolute(laplacian))

    # Otsu's thresholding
    ret_otsu, im_bw_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological closing to remove holes
    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    # Find contours on the binary image
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the result
    mask = np.zeros_like(gs)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Extract features from the foreground based on the mask
    foreground_features = cv2.bitwise_and(image_resized, image_resized, mask=mask)

    # Display images at different stages with contours
    st.subheader('Image Processing Steps')
    
    # Display the original image
    st.image(image_resized, caption='Original Image', use_column_width=True)

    # Display other processed images
    st.image(blurred, caption='Gaussian Blurred Image', use_column_width=True)
    st.image(edges_canny, caption='Canny Edge Detection', use_column_width=True)
    st.image(edges_laplacian, caption='Laplacian Edge Detection', use_column_width=True)
    st.image(im_bw_otsu, caption="Otsu's Thresholding", use_column_width=True)
    st.image(closing, caption='Morphological Closing', use_column_width=True)
    st.image(mask, caption='Contour Mask', use_column_width=True)
    st.image(foreground_features, caption='Foreground Features', use_column_width=True)
    
    # Draw contours on the original image
    img_with_contours = image_resized.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    # Display the original image with contours
    st.subheader('Original Image with Contours')
    st.image(img_with_contours, caption='Original Image with Contours', use_column_width=True)

# Function to make prediction on new features
def make_prediction(features):
    # Reshape the features into a 2D array
    features_reshaped = np.array(features).reshape(1, -1)

    # Make predictions on the new features
    prediction = random_forest.predict(features_reshaped)
    return prediction[0]

# Streamlit app
def main():
    st.title('Tomato Disease Prediction App')

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a tomato leaf image...", type="jpg")

    if uploaded_file is not None:
        # Read image from the uploaded file
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Resize the image for better display
        image_resized = cv2.resize(image, (256, 256))

        # Display the resized image
        st.image(image_resized, caption='Uploaded Image', use_column_width=True)

        # Preprocess and visualize steps
        features = feature_extractor(image_resized)
        if features == 'Invalid':
            st.warning('Invalid image. Please upload a valid image.')
        else:
            preprocess_and_visualize(image_resized)

            # Make prediction
            prediction = make_prediction(features)
            st.success(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
