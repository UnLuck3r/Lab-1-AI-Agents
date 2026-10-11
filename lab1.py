import os
import time
from dotenv import load_dotenv
from groq import Groq
from google import genai
from openai import OpenAI

load_dotenv()

PROMPT = "Объясни разницу между машинным обучением и LLM"

# Не чат-модели: речь, распознавание, модерация, эмбеддинги
NOT_CHAT = ("orpheus", "whisper", "guard", "tts", "embed", "playai")
PREFERRED = ("llama", "gpt-oss", "qwen")


def groq_candidates(client):
    """Список чат-моделей Groq: сначала GROQ_MODEL из .env, затем предпочтительные, затем остальные."""
    forced = os.getenv("GROQ_MODEL")
    if forced:
        return [forced]
    ids = [m.id for m in client.models.list().data]
    chat = [i for i in ids if not any(b in i.lower() for b in NOT_CHAT)]
    preferred = [i for i in chat if any(p in i.lower() for p in PREFERRED)]
    rest = [i for i in chat if i not in preferred]
    result = preferred + rest
    if not result:
        raise RuntimeError("Нет подходящих чат-моделей. Доступно: " + ", ".join(ids))
    return result


def test_groq():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    last_error = None
    for model in groq_candidates(client):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
            )
        except Exception as e:
            last_error = e
            print(f"  (Groq: модель {model} не сработала, пробую следующую)")
            continue
        elapsed = time.time() - start
        u = response.usage
        return f"Groq ({model})", elapsed, u.prompt_tokens, u.completion_tokens, u.total_tokens
    raise RuntimeError(f"Ни одна модель Groq не ответила. Последняя ошибка: {last_error}")


def test_gemini():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    start = time.time()
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=PROMPT,
    )
    elapsed = time.time() - start
    u = response.usage_metadata
    prompt = (u.prompt_token_count or 0) if u else 0
    completion = (u.candidates_token_count or 0) if u else 0
    total = (u.total_token_count or 0) if u else 0
    return "Google AI Studio (Gemini 3.8 Flash)", elapsed, prompt, completion, total


def test_openrouter():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    start = time.time()
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": PROMPT}],
    )
    elapsed = time.time() - start
    u = response.usage
    return "OpenRouter (DeepSeek)", elapsed, u.prompt_tokens, u.completion_tokens, u.total_tokens


if __name__ == "__main__":
    results = []
    print("Запуск тестов моделей...\n")

    for func in [test_groq, test_gemini, test_openrouter]:
        try:
            name, duration, p_tok, c_tok, t_tok = func()
            results.append((name, round(duration, 2), p_tok, c_tok, t_tok))
            print(f"[Успешно] {name}: {duration:.2f} сек, {t_tok} токенов")
        except Exception as e:
            print(f"[Ошибка] в {func.__name__}: {e}")

    print("\n" + "=" * 90)
    print(f"{'Модель':<45}{'Время, с':>10}{'Промпт':>10}{'Ответ':>10}{'Всего':>10}")
    print("-" * 90)
    for name, t, p, c, tot in results:
        print(f"{name:<45}{t:>10}{p:>10}{c:>10}{tot:>10}")
