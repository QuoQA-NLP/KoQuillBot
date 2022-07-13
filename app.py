# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(
    page_title="KoQuillBot", layout="wide", initial_sidebar_state="expanded"
)

tokenizer = AutoTokenizer.from_pretrained("QuoQA-NLP/KE-T5-Ko2En-Base")
ko2en_model = AutoModelForSeq2SeqLM.from_pretrained("QuoQA-NLP/KE-T5-Ko2En-Base")
en2ko_model = AutoModelForSeq2SeqLM.from_pretrained("QuoQA-NLP/KE-T5-En2Ko-Base")


st.title("🤖 KoQuillBot")


default_value = "한국어 문장 변환기 QuillBot입니다."
src_text = st.text_area(
    "바꾸고 싶은 문장을 입력하세요:",
    default_value,
    height=50,
    max_chars=200,
)
print(src_text)


def infer_sentence(model, src_text, tokenizer=tokenizer):
    encoded_prompt = tokenizer.encode(
        src_text,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        max_length=64,
    )
    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=64,
        num_beams=5,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
    )
    print(output_sequences)

    generated_sequence = output_sequences[0]
    print(generated_sequence)

    # Decode text
    text = tokenizer.decode(
        generated_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True
    )
    print(text)

    # Remove all text after the pad token
    stop_token = tokenizer.eos_token
    text = text[: text.find(stop_token) if stop_token else None]
    text = text.strip()
    return text


if st.button("문장 변환") or src_text == default_value:
    if src_text == "":
        st.warning("Please **enter text** for translation")

    else:
        st.success("Translating...")
        english_translation = infer_sentence(
            model=ko2en_model, src_text=src_text, tokenizer=tokenizer
        )

        korean_translation = en2ko_model.generate(
            **tokenizer(
                english_translation,
                return_tensors="pt",
                padding=True,
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
        st.success(f"{src_text} -> {english_translation} -> {korean_translation}")
else:
    pass


st.write(korean_translation)
print(korean_translation)
