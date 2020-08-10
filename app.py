# External Libraries
import streamlit as st

# Load the model
from gttp_lightning import PointerGenerator

@st.cache(allow_output_mutation=True)
def load_model_for_app():
    pretrained_model = PointerGenerator.load_from_checkpoint(checkpoint_path='epoch=7.ckpt')
    pretrained_model.eval()
    pretrained_model.freeze()
    return pretrained_model

pretrained_GTTP = load_model_for_app()

# Streamlit App Starts Here

default_source_text = 'There was a celebration in Dubai for the New Year and it had in attendance some of the biggest ' \
                      'names in Hollywood. World leaders also gathered and promised to work towards a peaceful tomo.'
default_dest_len = 10

st.title('Get To The Point')

st.header("Demo")
# Get Text
source_text = st.text_area('Enter Text Here', default_source_text)
req_summ_len = st.slider('Summary Length', 1, 100, default_dest_len)

prjtns, _, i2v = pretrained_GTTP(orig_text=source_text, summ_len=req_summ_len)
prjtns_argmax = prjtns.argmax(dim=1)
pred_summary = ' '.join([i2v[i.item()] for i in prjtns_argmax])

st.header("Predicted Summary")
st.write(pred_summary)
