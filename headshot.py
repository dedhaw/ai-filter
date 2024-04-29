import streamlit as st
import replicate as r
from openai import OpenAI
import os

# API set up
# Image
os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']
stabilityAi = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
instantId = "zsxkib/instant-id:7fb5a4f0e61867205adb37a2a8680e2e3e307d0387a06e85ea363cf8ad6fb37d"
controlNet = "jagilley/controlnet-depth2img:922c7bb67b87ec32cbc2fd11b1d5f94f0ba4f5519c4dbd02856376444127cc60"

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
            query = input + " MAKE SURE HAIR, FACIAL STRUCTURE, AND FACIAL FEATURES STAY THE SAME"
            output = r.run(
                "zsxkib/instant-id:7fb5a4f0e61867205adb37a2a8680e2e3e307d0387a06e85ea363cf8ad6fb37d",
                input={
                    "image": currImage,
                    "prompt": input,
                    "scheduler": "EulerDiscreteScheduler",
                    "enable_lcm": False,
                    "pose_image": "https://replicate.delivery/pbxt/KJmFdQRQVDXGDVdVXftLvFrrvgOPXXRXbzIVEyExPYYOFPyF/80048a6e6586759dbcb529e74a9042ca.jpeg",
                    "sdxl_weights": "protovision-xl-high-fidel",
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
                    "enable_canny_controlnet": False,
                    "enable_depth_controlnet": False,
                    "lcm_num_inference_steps": 5,
                    "controlnet_conditioning_scale": 0.8
                }
            )
            print(output)
            
        st.image(output)
            
      
# No image uploaded      
if imageUploaded == False and input:
    with st.chat_message(name="user") :
        st.write(input)
        
    with st.chat_message(name="ai") :
        st.write("Please upload an image!")