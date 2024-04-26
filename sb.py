import streamlit as st
import replicate as r
import os

os.environ['REPLICATE_API_TOKEN'] = st.secrets['REPLICATE_API_TOKEN']

output = r.run(
  "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
  input={"prompt": "an iguana on the beach, pointillism"}
)

st.image(output)
# ['https://replicate.delivery/pbxt/VJyWBjIYgqqCCBEhpkCqdevTgAJbl4fg62aO4o9A0x85CgNSA/out-0.png']