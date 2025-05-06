from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig
import torch
import argparse
import os
import logging

# Настройка логирования для отладки
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "konoval03/rubert_qa_model"


def load_model():
    """Загрузка модели с корректной обработкой размера словаря"""
    try:
        logger.info(f"Загрузка токенизатора из {MODEL_NAME}...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

        logger.info(f"Загрузка конфигурации модели из {MODEL_NAME}...")
        config = BertConfig.from_pretrained(MODEL_NAME)

        # Проверяем размер словаря в конфигурации и приводим его в соответствие с токенизатором
        vocab_size = len(tokenizer.vocab)
        if config.vocab_size != vocab_size:
            logger.info(f"Обновление размера словаря в конфигурации: {config.vocab_size} -> {vocab_size}")
            config.vocab_size = vocab_size

        logger.info(f"Загрузка модели из {MODEL_NAME} с обновленной конфигурацией...")
        model = BertForQuestionAnswering.from_pretrained(
            MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Перемещение модели на устройство: {device}")
        model = model.to(device)

        return model, tokenizer, device
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise


def answer_question(question, context, model, tokenizer, device):
    """Получение ответа на вопрос из контекста"""
    # Если контекст не предоставлен, используем сам вопрос
    if not context:
        context = question

    # Токенизация входных данных
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=384,
        truncation="only_second",
        padding="max_length"
    )

    # Перемещение данных на нужное устройство
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Получение предсказаний модели
    with torch.no_grad():
        outputs = model(**inputs)

    # Находим наиболее вероятные позиции начала и конца ответа
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    # Корректировка индексов при необходимости
    if answer_end < answer_start:
        answer_end = answer_start

    # Декодирование ответа из токенов
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

    # Вычисление уверенности
    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)
    confidence = start_probs[0, answer_start].item() * end_probs[0, answer_end].item()

    return answer, confidence


def main():
    """Основная функция для запуска через командную строку"""
    parser = argparse.ArgumentParser(description='QA система на русском языке')
    parser.add_argument('--question', type=str, required=True, help='Вопрос')
    parser.add_argument('--context', type=str, default='', help='Контекст (опционально)')

    args = parser.parse_args()

    # Загрузка модели
    try:
        print("Загрузка модели...")
        model, tokenizer, device = load_model()

        # Получение ответа
        answer, confidence = answer_question(args.question, args.context, model, tokenizer, device)

        # Вывод результатов
        print(f"\nВопрос: {args.question}")
        if args.context:
            print(f"Контекст: {args.context[:100]}...")
        else:
            print("Контекст: не предоставлен")

        print(f"\nОтвет: {answer}")
        print(f"Уверенность: {confidence:.2%}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()