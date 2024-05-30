import streamlit as st
import replicate as r
from openai import OpenAI
import os

# API set up
# Image
os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']
stabilityAi = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
fofr = "fofr/become-image:8d0b076a2aff3904dfcec3253c778e0310a68f78483c4699c7fd800f3051d2b3"

# GPT
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = "gpt-3.5-turbo"

# UI
st.title("AI Filter")

# Image uploader
imageUploaded = False
with st.expander("Image uploader") :
    images = st.file_uploader("Upload Images", accept_multiple_files=True)
    imageNames = list()
    imageName = ""
    
    # Choose Image
    for image in images :
        imageNames.append(image.name)
        
    if imageNames :  
            imageName = st.selectbox("Pick a File to Display", options=imageNames)
  
# Display Image
for i in range(len(images)) :
    if imageNames[i] == imageName :
        currImage = images[i]
        st.image(currImage)
        imageUploaded = True
        
# User input
input = st.chat_input(placeholder="Enter a Theme!")

if imageUploaded :
    if input:
        with st.chat_message(name="user") :
            st.write(input)
            
        with st.chat_message(name="ai") : 
            text = "Loading your image"
            with st.spinner(text=text) :
                modelRole = f""" Create a prompt used ot generate an image based on the user input.
                RULES:
                There should be only one person in the headshot image
                Always specify to create a headshot image
                Specify the style
                Follow the example output
                Specify any background elements related to the theme
                Specify any elements related to the theme
                """
                query = client.chat.completions.create(
                    model=model,
                    messages = [
                        {"role": "system", "content": modelRole},
                        {"role": "user", "content" : input}
                    ]
                )
                
                styleImage = r.run(
                    stabilityAi,
                    input={
                        "prompt" : query.choices[0].message.content
                    }
                )
                
                st.write(styleImage)
                
                imgDir = "a person. Include any masks or facial accessories if needed"
                
                final = r.run(
                    fofr,
                    input={
                        "image": currImage,
                        "prompt": imgDir,
                        "image_to_become": styleImage[0],
                        "negative_prompt": "",
                        "prompt_strength": 2,
                        "number_of_images": 1,
                        "denoising_strength": 1,
                        "instant_id_strength": 1,
                        "image_to_become_noise": 0.3,
                        "control_depth_strength": 0.8,
                        "image_to_become_strength": 0.75
                    }
                )
                col1, col2 = st.columns(2)
                if final :
                    with col1 :
                        st.image(currImage)
                        
                    with col2 : 
                        st.image(final)
      
# No image uploaded      
if imageUploaded == False and input:
    with st.chat_message(name="user") :
        st.write(input)
        
    with st.chat_message(name="ai") :
        st.write("Please upload an image!")