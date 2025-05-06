import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

st.set_page_config(
    page_title="QA система",
    page_icon="❓",
    layout="wide"
)

MODEL_NAME = "konoval03/rubert_qa_model"


@st.cache_resource
def load_model_from_hub():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        return model, tokenizer, device
    except Exception as e:
        st.error(f"Ошибка при загрузке модели из Hugging Face Hub: {e}")
        st.stop()


def answer_question(question, context, model, tokenizer, device):
    """Получение ответа на вопрос из контекста"""
    if not context:
        context = question

    # Токенизация вопроса и контекста
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=384,
        truncation="only_second",
        padding="max_length"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Поиск наиболее вероятных позиций начала и конца ответа
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    # Корректировка позиций
    if answer_end < answer_start:
        answer_end = answer_start

    # Декодирование ответа
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer_tokens = tokens[answer_start:answer_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # Очистка от специальных токенов
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, str):
            answer = answer.replace(special_token, "")
        elif isinstance(special_token, list):
            for token in special_token:
                answer = answer.replace(token, "")

    answer = answer.strip()

    # Вычисление уверенности модели
    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)
    confidence = start_probs[0, answer_start].item() * end_probs[0, answer_end].item()

    return answer, confidence


# Основная часть приложения
try:
    st.sidebar.title("Информация о модели")
    st.sidebar.markdown(f"Модель: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})")
    st.sidebar.markdown("Задача: Ответы на вопросы (Question Answering)")

    with st.spinner("Загрузка модели с Hugging Face Hub..."):
        model, tokenizer, device = load_model_from_hub()

    st.title("🤖 Система ответов на вопросы на русском языке")
    st.write("""
    Введите вопрос и опционально добавьте текстовый отрывок (passage) для поиска ответа.
    """)

    # Основной раздел приложения
    with st.form("qa_form"):
        question = st.text_input("Введите ваш вопрос:")

        # Опциональный текстовый отрывок
        document = st.text_area("Текстовый отрывок (опционально):", height=200)

        # Кнопка для отправки формы
        submit_button = st.form_submit_button("Получить ответ")

    # Обработка запроса
    if submit_button and question:
        with st.spinner("Поиск ответа..."):
            answer, confidence = answer_question(question, document, model, tokenizer, device)

        # Отображение результатов
        st.write("### Результаты:")
        st.success(f"**Ответ:** {answer}")
        st.info(f"**Уверенность модели:** {confidence:.2%}")

        # Визуализация уверенности
        st.progress(min(confidence, 1.0))

        # Показать предупреждение, если документ не был предоставлен
        if not document:
            st.warning("Ответ получен без текстового отрывка. Его точность может быть ограничена.")

    # Информация о модели
    with st.expander("О модели"):
        st.write("""
        Эта система использует предобученную модель для задачи ответов на вопросы, загруженную с Hugging Face Hub.
        Модель определяет начало и конец фрагмента текста, который содержит ответ на заданный вопрос.

        **Особенности работы:**
        - Если предоставлен текстовый отрывок, модель ищет ответ в нём
        - Если текстовый отрывок не предоставлен, модель пытается найти ответ исходя из самого вопроса
        - Лучше всего работает с фактическими вопросами при наличии релевантного текстового контекста
        """)

except Exception as e:
    st.error(f"Произошла ошибка: {e}")
    st.write("Проверьте правильность указанного пути к модели на Hugging Face Hub.")