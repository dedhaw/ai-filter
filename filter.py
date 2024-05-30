import replicate as r
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load enviroment
load_dotenv()

# API set up
# Image
os.environ['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN')
stabilityAi = "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b"
fofr = "fofr/become-image:8d0b076a2aff3904dfcec3253c778e0310a68f78483c4699c7fd800f3051d2b3"

# GPT
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-3.5-turbo"

def generateCustomImage(image, theme) :
    query = generatePrompt(theme)
    baseImage = createBaseImage(query)
    final = createAIImage(image, baseImage)
    return final

def createAIImage(image, baseImage) :
    imgDir = "a person. Include any masks or facial accessories if needed"
    img = r.run(
        fofr,
        input={
            "image": image,
            "prompt": imgDir,
            "image_to_become": baseImage,
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
    
    return img

def createBaseImage(query) :
    baseImage = r.run(
        stabilityAi,
        input={
            "prompt" : query
        }
    )
    
    return baseImage[0]

def generatePrompt(theme) :
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
            {"role": "user", "content" : theme}
        ]
    )
    
    return query.choices[0].message.content