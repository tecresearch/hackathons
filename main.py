import streamlit as st
import tensorflow as tf
import numpy as np

# Function to get remedy for a disease
# def get_remedy_for_disease(disease):
#     # Define the disease-remedy mapping
#     disease_remedies = {
#         'Apple___Apple_scab': 'Apply fungicide X every 7 days.',

#         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "For Corn (maize) affected by Cercospora leaf spot or Gray leaf spot, one common remedy is to apply fungicides specifically formulated to target fungal infections. Additionally, cultural practices such as crop rotation, proper irrigation management, and maintaining good air circulation can help reduce the spread of the disease. It's also advisable to remove and destroy infected plant debris to prevent further contamination. Consulting with local agricultural extension services or experts can provide tailored recommendations based on the specific conditions and severity of the infection in your area.",

#         'Strawberry___Leaf_scorch':'Since this fungal pathogen overwinters on the fallen leaves of infected plants, proper garden sanitation is key. This includes the removal of infected garden debris from the strawberry patch, as well as the frequent establishment of new strawberry transplants',
#         # Add more disease-remedy mappings as needed
#     }
#     # Return the remedy if disease exists in mapping, otherwise return a default message
#     return disease_remedies.get(disease, " Don't worry consult with an agricultural expert for further advice")


# Function to get remedy for a disease
def get_remedy_for_disease(disease):
    # Define the disease-remedy mapping
    disease_remedies = {
        'Apple___Apple_scab': 'Apply fungicide X every 7 days.',
        'Apple___Black_rot': 'Prune affected leaves and apply fungicide Y.',
        'Apple___Cedar_apple_rust': 'Apply fungicide Z to control Cedar apple rust.',
        'Apple___healthy': 'No specific remedy needed for healthy apple trees.',
        'Blueberry___healthy': 'No specific remedy needed for healthy blueberry plants.',
        'Cherry_(including_sour)___healthy': 'No specific remedy needed for healthy cherry trees.',
        'Cherry_(including_sour)___Powdery_mildew': 'Apply fungicide M to control powdery mildew on cherry trees.',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "For Corn (maize) affected by Cercospora leaf spot or Gray leaf spot, one common remedy is to apply fungicides specifically formulated to target fungal infections. Additionally, cultural practices such as crop rotation, proper irrigation management, and maintaining good air circulation can help reduce the spread of the disease. It's also advisable to remove and destroy infected plant debris to prevent further contamination. Consulting with local agricultural extension services or experts can provide tailored recommendations based on the specific conditions and severity of the infection in your area.",
        'Corn_(maize)___Common_rust_': 'Apply fungicide N to control common rust in corn (maize).',
        'Corn_(maize)___healthy': 'No specific remedy needed for healthy corn plants.',
        'Corn_(maize)___Northern_Leaf_Blight': 'Apply fungicide O to control Northern leaf blight in corn (maize).',
        'Grape___Black_rot': 'Apply fungicide P to control black rot in grapes.',
        'Grape___Esca_(Black_Measles)': 'Apply fungicide Q to control Esca (Black Measles) in grapes.',
        'Grape___healthy': 'No specific remedy needed for healthy grapevines.',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Apply fungicide R to control leaf blight (Isariopsis Leaf Spot) in grapes.',
        'Orange___Haunglongbing_(Citrus_greening)': 'Apply antibiotic S to control Huanglongbing (Citrus greening) in oranges.',
        'Peach___Bacterial_spot': 'Apply copper-based fungicide T to control bacterial spot in peaches.',
        'Peach___healthy': 'No specific remedy needed for healthy peach trees.',
        'Pepper,_bell___Bacterial_spot': 'Apply copper-based fungicide U to control bacterial spot in bell peppers.',
        'Pepper,_bell___healthy': 'No specific remedy needed for healthy bell pepper plants.',
        'Potato___Early_blight': 'Apply fungicide V to control early blight in potatoes.',
        'Potato___healthy': 'No specific remedy needed for healthy potato plants.',
        'Potato___Late_blight': 'Apply fungicide W to control late blight in potatoes.',
        'Raspberry___healthy': 'No specific remedy needed for healthy raspberry plants.',
        'Soybean___healthy': 'No specific remedy needed for healthy soybean plants.',
        'Squash___Powdery_mildew': 'Apply fungicide X to control powdery mildew in squash.',
        'Strawberry___healthy': 'No specific remedy needed for healthy strawberry plants.',
        'Strawberry___Leaf_scorch': 'Since this fungal pathogen overwinters on the fallen leaves of infected plants, proper garden sanitation is key. This includes the removal of infected garden debris from the strawberry patch, as well as the frequent establishment of new strawberry transplants.',
        'Tomato___Bacterial_spot': 'Apply copper-based fungicide Y to control bacterial spot in tomatoes.',
        'Tomato___Early_blight': 'Apply fungicide Z to control early blight in tomatoes.',
        'Tomato___healthy': 'No specific remedy needed for healthy tomato plants.',
        'Tomato___Late_blight': 'Apply fungicide AA to control late blight in tomatoes.',
        'Tomato___Leaf_Mold': 'Apply fungicide BB to control leaf mold in tomatoes.',
        'Tomato___Septoria_leaf_spot': 'Apply fungicide CC to control Septoria leaf spot in tomatoes.',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply insecticidal soap DD to control spider mites in tomatoes.',
        'Tomato___Target_Spot': 'Apply fungicide EE to control target spot in tomatoes.',
        'Tomato___Tomato_mosaic_virus': 'Practice good sanitation and control measures to prevent the spread of Tomato mosaic virus.',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Apply insecticide FF to control Tomato Yellow Leaf Curl Virus in tomatoes.'
        # Add more disease-remedy mappings as needed
    }
    # Return the remedy if disease exists in mapping, otherwise return a positive message
    remedy = disease_remedies.get(disease)
    return (remedy if remedy else "No specific remedy found for this disease. Consult with agricultural experts for further advice.")

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=400, use_column_width=True)
    #Predict button
    if st.button("Predict"):
        st.write("Prediction Result")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        prediction = class_name[result_index]
        remedy = get_remedy_for_disease(prediction)
        st.success(f"Prediction: {prediction}")
        st.success(f"Remedy: {remedy}")
