import streamlit as st
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
import os
from test_model import testModel, predChar

unique_chars = {
                0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "A",
                11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K",
                21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30: "U",
                31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: "a", 37: "b", 38: "c", 39: "d", 40: "e",
                41: "f", 42: "g", 43: "h", 44: "i", 45: "j", 46: "k", 47: "l", 48: "m", 49: "n", 50: "o",
                51: "p", 52: "q", 53: "r", 54: "s", 55: "t", 56: "u", 57: "v", 58: "w", 59: "x", 60: "y",
                61: "z"}

st.write ("""
    ## Handwritten Text Character Recognition
""")

st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/jpeg;base64,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");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
def save_uploadedfile(uploadedfile):
    #save image
    with open(os.path.join("uploads/Img", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    
    #write image path to CSV
    with open('uploads/test.csv', 'a+') as wr:
        wr.write(f'Img/{uploadedfile.name},"X"\n')
    # return success message
    return st.success("Saved File:{} to uploads/Img".format(uploadedfile.name))

def writeOutput(res, test_set):
    maxIndex = list(res.idxmax(axis=1))
    for i in range(len(maxIndex)):
        #get image using the path in csv
        img = cv.imread('uploads/' + test_set.at[i, 'image'])
        #show the predicted character and the probability
        st.image(img)

        # write output to file
        with open('output/output.txt', 'w') as rf:
            rf.write(unique_chars.get(maxIndex[i]))
        
        #Display output on streamlit
        st.write(f'Prediction: {unique_chars.get(maxIndex[i], "error")}')
        st.write(f'Probability: {round(max(res[maxIndex[i]]), 4)}')

def show_im(doc_image):
    if doc_image is not None:
        st.write("File uploaded successfully")
        img = Image.open(doc_image)
        st.image(img,width=250)
        save_uploadedfile(doc_image)
        if st.button('Convert Image To Doc'):
            res, pred_set = predChar()
            writeOutput(res, pred_set)
            download_file()

def download_file():
    with open('output/output.txt', "rb") as file:
        btn = st.download_button(
                label="Download Output File",
                data=file,
                file_name="output.txt"
            )

option = st.selectbox(
    'Choose the option for loading data',
    ('Load Image', 'Take Photo'))
if option == "Take Photo":
    doc_image = st.camera_input("Allow this app to take pictures")
else:
    doc_image = st.file_uploader("Upload Document Image", type=["png","jpg","jpeg"], accept_multiple_files=False)

show_im(doc_image)

if st.button("Test the Model"):
    pred_results, test_set = testModel()
    writeOutput(pred_results, test_set)
    
