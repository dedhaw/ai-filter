import streamlit as st
import replicate as r
from openai import OpenAI
import os

# API set up
# Image
os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']
imageModel = "zsxkib/instant-id:7fb5a4f0e61867205adb37a2a8680e2e3e307d0387a06e85ea363cf8ad6fb37d"

# GPT
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
model = "gpt-3.5-turbo"

# UI
st.title("AI images")

# Image uploader
imageUploaded = False
with st.expander("Image uploader") :
    images = st.file_uploader("Upload Images", type=("png"), accept_multiple_files=True)
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
input = st.chat_input(placeholder="Ask me anything")

if imageUploaded :
    if input:
        with st.chat_message(name="user") :
            st.write(input)
            
        with st.chat_message(name="ai") :
            modelRole = f"""TAKE THE USER INPUT AND TRANSLATE IT INTO SOMETHING USEFUL FOR THE IMAGE GENERATOR TO USE
            ALWAYS DEFINE:
            # Repeat the theme
            # What type of Color pallete should be used or stick to natural colors
            # Lighting
            # Include whether lighting is realistic or not
            # Filters
            # Realistic or not
            # Any clothing elements
            # Styling of the image overall
            # Quality of image (HIGH UNLESS SPECIFIED OTHERWISE)
            # Specify removing a slight amount of FAT under the chin (UNLESS SPECIFIED OTHERWISE)
            # How much blur
            # Should the image have realistic features based on the theme
            # ADD ADITIONAL ELEMENTS IF NECESSARY
            
            EXAMPLE OUTPUT:
            Professional Headshot. Natural Color Pallete. Soft lighting to create a flattering effect. Realistic lighting: Yes. No filters. Business attire. Clean and minimalistic High image quality. Remove slight amount of fat under the chin. No Blur. Realistic Features: YES. Additional elements: None
            
            EXAMPLE OUTPUT:
            Disney Character. Bright and colorful lighting to capture the whimsical and magical essence of a Disney character. Realistic lighting: No, enhanced and exaggerated lighting for a fantasy feel. Cartoon-style filters to emulate the animated look of a Disney character. Disney character costume or outfit. Playful and animated, incorporating elements from a specific Disney character. High Quality. Remove slight amount of fat under the chin. No Blur. Realistic Features: NO. Additional elements: Add iconic props or accessories related to the chosen Disney character for more authenticity
            
            Start here: {input}
            """
            query = client.chat.completions.create(
                model=model,
                messages=[{"role" : "system", "content" : modelRole},
                          {"role" : "user", "content" : input}]
            )
            
            print(input + " " + query.choices[0].message.content + "\n")
            
            output = r.run(
                imageModel,
                input={
                    "image": currImage,
                    "prompt": input + " " + query.choices[0].message.content,
                    "scheduler": "EulerDiscreteScheduler",
                    "enable_lcm": False,
                    "output_format": "webp",
                    "pose_strength": 0.4,
                    "canny_strength": 0.3,
                    "depth_strength": 0.5,
                    "guidance_scale": 5,
                    "output_quality": 80,
                    "negative_prompt": "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured",
                    "ip_adapter_scale": 0.8,
                    "lcm_guidance_scale": 1.5,
                    "num_inference_steps": 30,
                    "enable_pose_controlnet": True,
                    "enhance_nonface_region": True,
                    "enable_canny_controlnet": True,
                    "enable_depth_controlnet": False,
                    "lcm_num_inference_steps": 5,
                    "controlnet_conditioning_scale": 0.8
                }
            )
            st.image(output)
            
            
      
# No image uploaded      
if imageUploaded == False and input:
    with st.chat_message(name="user") :
        st.write(input)
        
    with st.chat_message(name="ai") :
        st.write("Please upload an image!")