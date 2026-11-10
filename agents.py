import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from gigachat import GigaChat
from gigachat.models import ChatCompletionRequest, ChatMessage


# =====================================================================
# 1. СХЕМЫ ДАННЫХ PYDANTIC ДЛЯ ВСЕХ 5 АГЕНТОВ
# =====================================================================

# --- Схемы Агента 1 (SourceExtractorAgent) ---
class AtomicFact(BaseModel):
    fact_id: int = Field(..., description="Порядковый номер факта")
    category: str = Field(..., description="Категория: date, number, entity, statement")
    fact_type: str = Field(..., description="Подтип факта")
    fact_text: str = Field(..., description="Самодостаточная формулировка факта")
    original_quote: str = Field(..., description="Точная цитата из текста-источника")
    related_entity: str = Field(..., description="Сущность, к которой привязан факт")
    importance: str = Field(..., description="Критичность: high, medium, low")


# --- Схемы Агента 2 (ClaimDecomposerAgent) ---
class AtomicClaim(BaseModel):
    claim_id: int = Field(..., description="Порядковый номер утверждения")
    category: str = Field(..., description="Категория: date, number, entity, statement")
    claim_text: str = Field(..., description="Самодостаточная формулировка утверждения")
    original_quote: str = Field(..., description="Точная цитата из проверяемого ответа")
    check_worthy: bool = Field(..., description="Требует ли проверки (True/False)")
    quote_in_answer: bool = Field(default=True, description="Флаг присутствия цитаты в ответе")


# --- Схемы Агента 4 (CompletenessCheckerAgent - Верхняя версия из Colab) ---
class CoveredPoint(BaseModel):
    fact_id: int
    fact_text: str


class MissingPoint(BaseModel):
    fact_id: int
    importance: str
    fact_text: str
    exact_quote: Optional[str] = None


class CompletenessReport(BaseModel):
    coverage_score: float = Field(..., description="Коэффициент покрытия от 0 до 1")
    covered_points: List[CoveredPoint] = Field(default_factory=list)
    missing_points: List[MissingPoint] = Field(default_factory=list)


# --- Схемы Агента 5 (FinalBulkReportAgent) ---
class FinalDiscrepancy(BaseModel):
    issue_type: str = Field(..., description="'галлюцинация' (ложь из Агента 3) или 'упущение' (пропуск из Агента 4)")
    importance: str = Field(..., description="'high' (деньги/сроки), 'medium', 'low'")
    description: str = Field(..., description="Понятное описание проблемы на русском языке")
    bad_claim: Optional[str] = Field(None, description="Ошибочное утверждение из ответа (для галлюцинаций)")
    exact_quote: Optional[str] = Field(None, description="Точная исходная цитата из источника для доказательства")


class Agent5FinalReport(BaseModel):
    final_verdict: str = Field(..., description="'полностью_верен', 'требует_исправления', 'критические_ошибки'")
    summary_verdict: str = Field(...,
                                 description="Развернутое аналитическое заключение по качеству ответа с аргументацией")
    detected_issues: List[FinalDiscrepancy] = Field(default_factory=list,
                                                    description="Реестр всех найденных несоответствий")


# =====================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ СЛУЖЕБНЫЕ ФУНКЦИИ
# =====================================================================

def extract_first_json(text: str) -> str:
    """Оригинальный конечный автомат из Colab для надежного извлечения JSON"""
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    if start == -1:
        start = text.find("[")
    if start == -1:
        raise ValueError("В ответе модели не найден JSON-объект или массив")

    balance = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in ('{', '['):
                balance += 1
            elif char in ('}', ']'):
                balance -= 1
                if balance == 0:
                    return text[start:i + 1]
    return text[start:]


# =====================================================================
# 3. ПОЛНАЯ РЕАЛИЗАЦИЯ АГЕНТОВ (ЧАСТЬ 1: АГЕНТЫ 1, 2, 3)
# =====================================================================

class SourceExtractorAgent:
    """Агент 1: Извлекает жесткие атомарные факты из оригинальных текстов-источников"""

    def __init__(self, credentials: str):
        self.giga = GigaChat(credentials=credentials, verify_ssl_certs=False, timeout=30.0)
        self.prompt = (
            "Ты — профессиональный аналитик юридических документов. Твоя задача — извлечь ВСЕ "
            "значимые факты, суммы, даты, ограничения, обязанности сторон и штрафы из текста.\n"
            "Разбей текст на мелкие атомарные единицы. Каждая единица должна быть сформулирована "
            "самодостаточно (понятно без чтения всего документа).\n"
            "Запрещено пропускать численные значения и важные временные сроки.\n"
            "Отвечай СТРОГО в формате JSON-массива, где каждый элемент строго соответствует схеме AtomicFact."
        )

    def extract_facts(self, sources: str) -> List[AtomicFact]:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self.prompt),
                ChatMessage(role="user", content=sources)
            ]
        )
        res = self.giga.chat(request).messages.content
        data = json.loads(extract_first_json(res))
        return [AtomicFact(**item) for item in data] if isinstance(data, list) else []


class ClaimDecomposerAgent:
    """Агент 2: Вычленяет тезисы из ответа ИИ, фильтрует воду и проверяет цитаты на галлюцинации"""

    def __init__(self, credentials: str):
        self.giga = GigaChat(credentials=credentials, verify_ssl_certs=False, timeout=30.0)
        self.prompt = (
            "Ты — эксперт по анализу текстов и логике. Изучи предоставленный ответ ИИ и декомпозируй "
            "его на отдельные проверяемые тезисы (claims).\n"
            "Исключай из проверки вводные слова, вежливые фразы, воду и субъективные мнения автора "
            "(для них выставляй флаг check_worthy=false).\n"
            "Каждое утверждение должно содержать конкретную сущность вместо местоимений (он, она, они).\n"
            "Отвечай СТРОГО в формате JSON-массива, где каждый элемент строго соответствует схеме AtomicClaim."
        )

    def decompose_claims(self, answer: str) -> List[AtomicClaim]:
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self.prompt),
                ChatMessage(role="user", content=answer)
            ]
        )
        res = self.giga.chat(request).messages.content
        data = json.loads(extract_first_json(res))

        # Оригинальная программная валидация цитат на галлюцинации из Colab (нормализация текста)
        claims = []
        normalized_answer = re.sub(r'[^\w\s]', '', answer.lower())
        for item in data:
            claim = AtomicClaim(**item)
            normalized_quote = re.sub(r'[^\w\s]', '', claim.original_quote.lower())
            if normalized_quote not in normalized_answer:
                claim.quote_in_answer = False  # Цитата выдумана моделью
            claims.append(claim)
        return claims


class FinalClaimVerifierAgent:
    """Самый Новый Агент 3: Выполняет пакетную (bulk) проверку утверждений по внешнему файлу промпта"""

    def __init__(self, credentials: str):
        self.giga = GigaChat(credentials=credentials, verify_ssl_certs=False, timeout=30.0)
        try:
            with open("prompt_v3.txt", "r", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except:
            self.system_prompt = (
                "Ты — профессиональная Fact-Checking модель. Твоя задача — пакетно сопоставить "
                "список утверждений с предоставленными источниками.\n"
                "Выстави по каждому утверждению вердикт: 'подтверждено', 'противоречит' или 'нет в источниках'.\n"
                "Для вердикта 'противоречит' обязательно укажи дословную цитату из источника в поле exact_quote."
            )

    def batch_verify(self, claims: List[AtomicClaim], question: str, sources: str) -> list:
        # Передаем только те утверждения, которые Агент 2 посчитал важными для проверки
        claims_to_check = [c.dict() for c in claims if c.check_worthy]
        claims_json = json.dumps(claims_to_check, ensure_ascii=False)

        user_content = f"ИСХОДНЫЙ ВОПРОС: {question}\n\nДОКУМЕНТЫ-ИСТОЧНИКИ:\n{sources}\n\nСПИСОК УТВЕРЖДЕНИЙ ДЛЯ ПРОВЕРКИ:\n{claims_json}"
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_content)
            ]
        )
        res = self.giga.chat(request).messages.content
        return json.loads(extract_first_json(res))


# =====================================================================
# 4. ПОЛНАЯ РЕАЛИЗАЦИЯ АГЕНТОВ (ЧАСТЬ 2: АГЕНТЫ 4, 5 И ИНТЕРФЕЙС)
# =====================================================================

class CompletenessCheckerAgent:
    """Верхний Агент 4: Верхняя оригинальная версия. Ищет пропуски, сопоставляя Pydantic-списки фактов и тезисов"""

    def __init__(self, credentials: str):
        self.giga = GigaChat(credentials=credentials, verify_ssl_certs=False, timeout=30.0)
        self.prompt = (
            "Ты — строгий аудитор полноты информации. Твоя задача — сопоставить список жестких "
            "фактов из оригинального договора и список тезисов, которые ИИ реально выдал в ответе.\n"
            "Определи, какие важные факты из договора (особенно с важностью high: суммы, штрафы, сроки) "
            "были пропущены ИИ, хотя напрямую требовались для полноценного ответа на вопрос.\n"
            "Рассчитай долю покрытых фактов (coverage_score).\n"
            "Отвечай СТРОГО в формате JSON по схеме объекта CompletenessReport."
        )

    def check_completeness(self, facts: List[AtomicFact], claims: List[AtomicClaim]) -> dict:
        facts_json = json.dumps([f.dict() for f in facts], ensure_ascii=False)
        claims_json = json.dumps([c.dict() for c in claims], ensure_ascii=False)

        user_content = f"ЭТАЛОННЫЕ ФАКТЫ ИЗ ДОКУМЕНТА-ИСТОЧНИКА:\n{facts_json}\n\nФАКТИЧЕСКИЕ ТЕЗИСЫ ИЗ ОТВЕТА ИИ:\n{claims_json}"
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self.prompt),
                ChatMessage(role="user", content=user_content)
            ]
        )
        res = self.giga.chat(request).messages.content
        return json.loads(extract_first_json(res))


class FinalBulkReportAgent:
    """Агент 5: Консолидатор и Главный Судья системы. Объединяет пакетные выводы Агентов 3 и 4"""

    def __init__(self, credentials: str):
        self.giga = GigaChat(credentials=credentials, verify_ssl_certs=False, timeout=30.0)
        self.system_prompt = """Ты — Главный Консолидатор и Auditor качества ответов (Агент 5).
Твоя задача — сопоставить результаты пакетной проверки утверждений (Агент 3) и анализа упущенных Pydantic-фактов (Верхний Агент 4), сформировав финальный юридически обоснованный вердикт.

На вход тебе передается:
1. Исходный ВОПРОС пользователя.
2. Проверяемый ОТВЕТ ИИ.
3. ПАКЕТНЫЙ МАССИВ АГЕНТА 3: Список всех утверждений из ответа с вердиктами ("подтверждено" / "противоречит" / "нет в источниках") и цитатами.
4. ОТЧЕТ ПО ПОЛНОТЕ АГЕНТА 4 (Верхняя версия): Структурированный анализ упущенных фактов на основе AtomicFact и AtomicClaim.

ИНСТРУКЦИЯ ПО СБОРКЕ И СНИЖЕНИЮ ЛОЖНЫХ СРАБАТЫВАНИЙ:
1. Выдели из массива Агента 3 только те объекты, где вердикт равен "противоречит". Помести их в `detected_issues` как тип "галлюцинация".
2. Выдели из отчета Агента 4 пропущенные факты (`missing_points`). Помести их как тип "упущение".
3. Для каждого дефекта ОБЯЗАТЕЛЬНО найди и укажи `exact_quote` (точную исходную цитату). Если Агент 3 или Агент 4 не предоставили дословную цитату — игнорируй этот пункт (это защита от ложных срабатываний, требуемая для Accuracy > 85%).

Выдай результат СТРОГО в формате валидного JSON по схеме Agent5FinalReport, без markdown-тегов и без лишнего текста."""

    def generate_final_verdict(self, question: str, answer: str, agent3_bulk_output: list,
                               agent4_structured_output: dict) -> Agent5FinalReport:
        user_content = (
            f"1. ВОПРОС: {question}\n\n"
            f"2. ПРОВЕРЯЕМЫЙ ОТВЕТ: {answer}\n\n"
            f"3. ПАКЕТНЫЙ JSON АГЕНТА 3 (Все утверждения):\n{json.dumps(agent3_bulk_output, ensure_ascii=False, indent=2)}\n\n"
            f"4. ОТЧЕТ АГЕНТА 4 (Верхний вариант полноты):\n{json.dumps(agent4_structured_output, ensure_ascii=False, indent=2)}"
        )
        request = ChatCompletionRequest(messages=[ChatMessage(role="system", content=self.system_prompt),
                                                  ChatMessage(role="user", content=user_content)])
        res = self.giga.chat(request).messages.content
        parsed_data = json.loads(extract_first_json(res))
        return Agent5FinalReport(**parsed_data)


def print_agent5_report(report: Agent5FinalReport):
    """Красиво выводит финальный вердикт Агента 5 в консоль PyCharm с иконками"""
    print("=" * 75)
    print("⚖️  ИНФОРМАЦИОННАЯ ПАНЕЛЬ ГЛАВНОГО АУДИТОРА (АГЕНТ №5)")
    print("=" * 75)

    verdict_upper = report.final_verdict.upper()
    if "ПОЛНОСТЬЮ_ВЕРЕН" in verdict_upper:
        verdict_marker = "🟢 [УСПЕШНО] "
    elif "ТРЕБУЕТ_ИСПРАВЛЕНИЯ" in verdict_upper:
        verdict_marker = "🟡 [ВНИМАНИЕ] "
    else:
        verdict_marker = "🔴 [КРИТИЧЕСКАЯ ОШИБКА] "

    print(f"{verdict_marker} Финальный вердикт: {verdict_upper}")
    print(f"📝 Аналитическое заключение:\n   {report.summary_verdict}")
    print("-" * 75)

    issues_count = len(report.detected_issues)
    print(f"📋 Всего выявлено несоответствий: {issues_count}")
    print("=" * 75)

    if issues_count == 0:
        print("✅ Нарушений не обнаружено. Ответ полностью соответствует источникам.")
        print("=" * 75)
        return

    for idx, issue in enumerate(report.detected_issues, 1):
        importance_marker = "🔴" if issue.importance == "high" else "🟠" if issue.importance == "medium" else "🟡"
        print(f"{importance_marker} [{issue.issue_type.upper()}] | Дефект #{idx} | Важность: {issue.importance}")
        print(f"   ⚙️  Суть проблемы: {issue.description}")
        if issue.bad_claim:
            print(f"   ❌ Искаженный тезис из ответа: \"{issue.bad_claim}\"")
        if issue.exact_quote:
            print(f"   💬 Цитата-доказательство из источника: \"{issue.exact_quote}\"")
        print("." * 75)
    print("=" * 75)
