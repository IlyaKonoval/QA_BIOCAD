from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import argparse

MODEL_NAME = "konoval03/rubert_qa_model"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, tokenizer, device


def answer_question(question, context, model, tokenizer, device):
    if not context:
        context = question

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

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits)

    if answer_end < answer_start:
        answer_end = answer_start

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    answer_tokens = tokens[answer_start:answer_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, str):
            answer = answer.replace(special_token, "")
        elif isinstance(special_token, list):
            for token in special_token:
                answer = answer.replace(token, "")

    answer = answer.strip()

    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)
    confidence = start_probs[0, answer_start].item() * end_probs[0, answer_end].item()

    return answer, confidence


def main():
    parser = argparse.ArgumentParser(description='QA система на русском языке')
    parser.add_argument('--question', type=str, required=True, help='Вопрос')
    parser.add_argument('--context', type=str, default='', help='Контекст (опционально)')

    args = parser.parse_args()

    print("Загрузка модели...")
    model, tokenizer, device = load_model()

    answer, confidence = answer_question(args.question, args.context, model, tokenizer, device)

    print(f"\nВопрос: {args.question}")
    if args.context:
        print(f"Контекст: {args.context[:100]}...")
    else:
        print("Контекст: не предоставлен")

    print(f"\nОтвет: {answer}")
    print(f"Уверенность: {confidence:.2%}")


if __name__ == "__main__":
    main()