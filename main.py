import os
import pandas as pd
from dotenv import load_dotenv
from agents import (
    SourceExtractorAgent,
    ClaimDecomposerAgent,
    FinalClaimVerifierAgent,
    CompletenessCheckerAgent,
    FinalBulkReportAgent,
    print_agent5_report
)

# Загружаем ключ из файла .env
load_dotenv()
GIGACHAT_KEY = os.getenv("GIGACHAT_KEY")

if not GIGACHAT_KEY:
    raise ValueError("Ошибка: Токен GIGACHAT_KEY не найден в файле .env!")


def run_benchmark():
    # Наш текстовый файл с тест-кейсами
    csv_path = "test_cases.txt"
    if not os.path.exists(csv_path):
        print(f"❌ Ошибка: Файл {csv_path} не найден в папке проекта!")
        return

    # Жестко прописываем кодировку utf-8 для корректной работы с русским языком на Linux
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"🚀 Загружено тест-кейсов для проверки: {len(df)}")

    # Инициализация всех 5 агентов в равной степени
    print("🤖 Инициализация агентского контура...")
    agent1 = SourceExtractorAgent(credentials=GIGACHAT_KEY)
    agent2 = ClaimDecomposerAgent(credentials=GIGACHAT_KEY)
    agent3 = FinalClaimVerifierAgent(credentials=GIGACHAT_KEY)
    agent4 = CompletenessCheckerAgent(credentials=GIGACHAT_KEY)
    agent5 = FinalBulkReportAgent(credentials=GIGACHAT_KEY)

    # Пробегаем по строкам бенчмарка
    for idx, row in df.iterrows():
        print(f"\n================ Проверка кейса №{idx + 1} ================")
        question = row["question"]
        answer = row["answer"]
        sources = row["sources"]

        try:
            print("1️⃣ Извлечение фактов из источника (Агент 1)...")
            facts = agent1.extract_facts(sources)

            print("2️⃣ Декомпозиция ответа на утверждения (Агент 2)...")
            claims = agent2.decompose_claims(answer)

            print("3️⃣ Пакетная проверка утверждений на истинность (Агент 3)...")
            agent3_results = agent3.batch_verify(claims, question, sources)

            print("4️⃣ Проверка полноты и упущенных фактов (Агент 4)...")
            agent4_results = agent4.check_completeness(facts, claims)

            print("5️⃣ Консолидация отчетов и вынесение вердикта (Агент 5)...")
            final_report = agent5.generate_final_verdict(
                question=question,
                answer=answer,
                agent3_bulk_output=agent3_results,
                agent4_structured_output=agent4_results
            )

            # Отрисовка красивой цветной панели
            print_agent5_report(final_report)

        except Exception as e:
            print(f"❌ Сбой при обработке кейса №{idx + 1}: {e}")


if __name__ == "__main__":
    run_benchmark()
