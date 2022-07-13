# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(
    page_title="KoQuillBot", layout="wide", initial_sidebar_state="expanded"
)

@st.cache
def load_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

tokenizer = AutoTokenizer.from_pretrained("QuoQA-NLP/KE-T5-Ko2En-Base")
ko2en_model = load_model("QuoQA-NLP/KE-T5-Ko2En-Base")
en2ko_model = load_model("QuoQA-NLP/KE-T5-En2Ko-Base")


st.title("🤖 KoQuillBot")


default_value = "이건 한국어 문장 변환기 QuillBot입니다."
src_text = st.text_area(
    "바꾸고 싶은 문장을 입력하세요:",
    default_value,
    height=50,
    max_chars=200,
)
print(src_text)



if src_text == "":
    st.warning("Please **enter text** for translation")

# translate into english sentence
english_translation = ko2en_model.generate(
    **tokenizer(
        src_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    ),
    max_length=64,
    num_beams=5,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3,
    num_return_sequences=1,
)
english_translation = tokenizer.decode(
    english_translation[0],
    clean_up_tokenization_spaces=True,
    skip_special_tokens=True,
)

# translate back to korean
korean_translation = en2ko_model.generate(
    **tokenizer(
        english_translation,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    ),
    max_length=64,
    num_beams=5,
    repetition_penalty=1.3,
    no_repeat_ngram_size=3,
    num_return_sequences=1,
)

korean_translation = tokenizer.decode(
    korean_translation[0],
    clean_up_tokenization_spaces=True,
    skip_special_tokens=True,
)
print(f"{src_text} -> {english_translation} -> {korean_translation}")

st.write(korean_translation)
print(korean_translation)
