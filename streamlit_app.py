from pathlib import Path

import streamlit as st
import torch

from interpretable_nlp.models import (initialize_model,
                                      embeddings_mapper,
                                      MODEL_TYPE_MAPPING)
from interpretable_nlp.predict_utils import (predict_sentiment,
                                             attribution_fun,
                                             attribution_to_html,
                                             compose_annotation_file)

from transformers import AutoTokenizer


option = st.selectbox(
    'Выберите модель для автоматического аннотирования',
     ('Distil RuBert Cased Conversational от DeepPavlov',
      'Cointegrated RuBert'))


SENTIMENT_MODEL = MODEL_TYPE_MAPPING[option]


def get_tokenizer():
    tokenizer_ = AutoTokenizer.from_pretrained(SENTIMENT_MODEL.value)
    return tokenizer_


@st.cache
def get_model():
    sentiment_model_ = initialize_model(SENTIMENT_MODEL)
    sentiment_model_.eval()
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_model_.to(device_)
    return sentiment_model_, device_


sentiment_model, device = get_model()
embedding = embeddings_mapper(SENTIMENT_MODEL)

tokenizer = get_tokenizer()
special_tokens = {'PAD_IDX': tokenizer.pad_token_id,
                  'SEP_IDX': tokenizer.sep_token_id,
                  'CLS_IDX': tokenizer.cls_token_id}


vocab_size = tokenizer.vocab_size
max_len = 512

markdown = st.sidebar.markdown(
    "<h2>Автоматическое нанесение эмотивной разметки</h2>"
    "<h4>Категории эмоций:</h4>"
    "<ul>"
    "<li><span style='background-color:rgba(255, 255, 128, .8)'>радость</span></li>"
    "<li><span style='background-color:rgba(30,129,176, .7)'>печаль</span></li>"
    "<li><span style='background-color:rgba(228,52,52, .7)'>злость</span></li>"
    "<li><span style='background-color:rgba(200, 114, 226, .7)'>удивление/страх</span></li>"
    "<li>нейтральный</span></li>"
    "</ul>",
    unsafe_allow_html=True,
)

txt = st.text_area(
    "Введите аннотируемый текст:",
    "Какая замечательная погода!!))",
)

tokenized = tokenizer(txt)
sentiment_output = predict_sentiment(tokenized,
                                     model=sentiment_model,
                                     device=device)
color = {
         "радость": "rgba(255, 255, 128, .8)",
         "печаль": "rgba(30,129,176, .7)",
         "злость": "rgba(228,52,52, .7)",
         "неопред": "rgba(200, 114, 226, .7)",
         "нейтр": "white",
         }[sentiment_output]

emoji = {
    "радость": "😀",
    "печаль": "😭",
    "злость": "😡",
    "неопред": "🤔",
    "нейтр": "😐"}[sentiment_output]

st.write(
    "Эксплицируемая эмоция:",
    f"{emoji} <span style='background-color:{color}'>{sentiment_output}</span>",
    unsafe_allow_html=True,
)

tokens, attributions = attribution_fun(tokenized=tokenized,
                                       model=sentiment_model,
                                       embedding=embedding,
                                       device=device,
                                       special_tokens=special_tokens)
tokens = tokenizer.convert_ids_to_tokens(tokens.input_ids)[1:-1]
html = attribution_to_html(tokens=tokens, attributions=attributions, sentiment=sentiment_output)

st.write(
    "Визуальная аннотация:\n\n",
    html,
    unsafe_allow_html=True,
)

annotation = compose_annotation_file(txt, tokens, attributions)
st.json(annotation)
st.download_button(
    label="Скачать аннотацию",
    file_name="annotation.json",
    mime="application/json",
    data=annotation,
)

