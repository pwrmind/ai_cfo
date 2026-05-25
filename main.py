import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# МОДУЛЬ 1: ПАРСЕР ФОРМАТА 1C (ГЛАЗА СИСТЕМЫ)
# ==========================================
class BankParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.my_account = None

    def parse(self):
        """Читает файл 1CClientBankExchange и возвращает DataFrame транзакций"""
        encodings = ['cp1251', 'utf-8', 'windows-1251'] # Стандартные кодировки 1С
        content = ""
        
        for enc in encodings:
            try:
                with open(self.filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if not content:
            raise ValueError("Не удалось прочитать файл. Проверьте кодировку.")

        lines = content.split('\n')
        transactions = []
        current_tx = {}
        in_doc = False

        # 1. Находим наш расчетный счет в заголовке
        for line in lines:
            if line.startswith('РасчСчет=') and not self.my_account:
                self.my_account = line.split('=')[1].strip()
                break

        # 2. Парсим документы
        for line in lines:
            line = line.strip()
            if line.startswith('СекцияДокумент='):
                in_doc = True
                current_tx = {}
                continue
            
            if line.startswith('КонецДокумента'):
                in_doc = False
                if 'Сумма' in current_tx:
                    # ЛОГИКА ЗНАКОВ:
                    # Если плательщик - МЫ, то это РАСХОД (минус)
                    # Если получатель - МЫ, то это ПРИХОД (плюс)
                    amount = float(current_tx.get('Сумма', 0))
                    payer_acc = current_tx.get('ПлательщикРасчСчет')
                    
                    if payer_acc == self.my_account:
                        amount = -abs(amount) # Принудительный минус
                        direction = "Rashod"
                    else:
                        amount = abs(amount)  # Принудительный плюс
                        direction = "Prihod"

                    transactions.append({
                        'Дата': current_tx.get('Дата'),
                        'Назначение': current_tx.get('НазначениеПлатежа', ''),
                        'Сумма': amount,
                        'Контрагент': current_tx.get('Получатель' if direction == "Rashod" else 'Плательщик'),
                        'Направление': direction
                    })
                continue

            if in_doc and '=' in line:
                key, value = line.split('=', 1)
                current_tx[key] = value

        return pd.DataFrame(transactions)

# ==========================================
# МОДУЛЬ 2: ИИ-КЛАССИФИКАТОР (МОЗГ СИСТЕМЫ)
# ==========================================
# ==========================================
# МОДУЛЬ 2: УСИЛЕННЫЙ ГИБРИДНЫЙ КЛАССИФИКАТОР
# ==========================================
import re

class AIClassifier:
    def __init__(self):
        print("\n[Система] Загрузка нейросети (LaBSE) и справочников MCC...")
        self.model = SentenceTransformer("sentence-transformers/LaBSE")
        
        # 1. СПРАВОЧНИК MCC КОДОВ (Жесткая логика)
        self.mcc_codes = {
            "5411": "OPERATING_EXPENSE", # Продукты/Супермаркеты (Офис)
            "5812": "OPERATING_EXPENSE", # Рестораны (Представительские)
            "5814": "OPERATING_EXPENSE", # Фастфуд
            "5999": "OPERATING_EXPENSE", # Разное (Магазины)
            "5732": "CAPEX",             # Электроника (Техника)
            "5942": "OPERATING_EXPENSE", # Книги/Канцелярия
            "4814": "FIXED_EXPENSE",     # Телеком/Связь
            "7372": "FIXED_EXPENSE",     # Программирование/IT услуги
            "6011": "FINANCIAL_FLOW",    # Снятие наличных
            "6538": "FINANCIAL_FLOW"     # Перевод на карту (Card2Card)
        }

        # 2. УЛУЧШЕННЫЕ ЭТАЛОНЫ (С добавлением специфики РФ)
        self.categories = {
            "OPERATING_INCOME": [
                "Оплата от клиента", "Поступление выручки", "Оплата по счету", 
                "Розничная выручка", "Зачисление средств от продаж"
            ],
            "OPERATING_EXPENSE": [
                "Закупка товара", "Материалы", "Логистика", "Комиссия банка", 
                "Хозтовары", "Канцелярия", "Представительские расходы", "Питание"
            ],
            "FIXED_EXPENSE": [
                "Аренда офиса", "Зарплата", "Бухгалтерия", "Интернет", 
                "Налоги и взносы", "Взносы в фонды", "Абонентская плата", "SMS информирование"
            ],
            "CAPEX": [
                "Покупка оборудования", "Компьютеры", "Мебель", "Автомобиль", "Основные средства"
            ],
            "FINANCIAL_FLOW": [
                "Взнос наличных", "Перевод собственных средств", "Пополнение счета", 
                "Выдача займа", "Возврат кредита", "Уставный капитал", "Вывод дивидендов"
            ]
        }
        
        self.anchors = {}
        for cat, texts in self.categories.items():
            self.anchors[cat] = np.mean(self.model.encode(texts), axis=0)

    def clean_text(self, text):
        """Удаляет технический мусор, чтобы ИИ видел суть"""
        # Удаляем "Расчеты через ТУ ... \RU\..."
        text = re.sub(r'Расчеты через ТУ\s+\d+', '', text)
        text = re.sub(r'\\RU\\[A-Za-z]+\\', ' ', text) 
        # Удаляем номера карт (маскированные)
        text = re.sub(r'\d{6}\+{4,}\d{4}', '', text)
        # Удаляем чеки и даты
        text = re.sub(r'по чеку\s+[\d\.,]+', '', text)
        text = re.sub(r'MP-\d+', '', text)
        # Удаляем лишние пробелы
        return " ".join(text.split())

    def get_mcc(self, text):
        """Ищет MCC код в строке"""
        match = re.search(r'MCC(\d{4})', text)
        return match.group(1) if match else None

    def classify(self, df):
        print("[Система] Запуск гибридной классификации (MCC + AI)...")
        
        results = []
        # Проходим по каждой транзакции отдельно
        for txt in df['Назначение']:
            # 1. ПРОВЕРКА ПО MCC
            mcc = self.get_mcc(txt)
            if mcc and mcc in self.mcc_codes:
                results.append(self.mcc_codes[mcc])
                continue 
            
            # 2. ЕСЛИ MCC НЕТ - ЧИСТИМ ТЕКСТ И СПРАШИВАЕМ ИИ
            cleaned_txt = self.clean_text(txt)
            
            # Если текст стал пустым (были одни тех данные), помечаем как расход
            if len(cleaned_txt) < 3:
                 results.append("OTHER")
                 continue

            vec = self.model.encode(cleaned_txt)
            best_cat = "OTHER"
            max_sim = -1
            
            for cat, anchor in self.anchors.items():
                sim = cosine_similarity(vec.reshape(1, -1), anchor.reshape(1, -1))[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_cat = cat
            
            # Порог уверенности (0.30 для LaBSE нормально)
            if max_sim < 0.30:
                best_cat = "OTHER"
                
            results.append(best_cat)
        
        df['Category'] = results
        return df

# ==========================================
# МОДУЛЬ 3: PINN-СИМУЛЯТОР (ФИЗИКА БИЗНЕСА)
# ==========================================
class PINN_Engine:
    def __init__(self, dataframe, tax_rate=0.06):
        self.df = dataframe
        self.tax_rate = tax_rate

    def run_simulation(self):
        print("\n" + "="*50)
        print(" ЗАПУСК PINN-СИМУЛЯЦИИ (Physics-Informed Neural Logic) ")
        print("="*50)

        # 1. СБОР ФАКТОВ (Linear Analysis)
        # Суммируем строго по категориям
        op_income = self.df[self.df['Category'] == 'OPERATING_INCOME']['Сумма'].sum()
        
        # Расходы в базе отрицательные, берем модуль для расчетов "затрат"
        op_expense_abs = abs(self.df[self.df['Category'] == 'OPERATING_EXPENSE']['Сумма'].sum())
        fixed_expense_abs = abs(self.df[self.df['Category'] == 'FIXED_EXPENSE']['Сумма'].sum())
        capex_abs = abs(self.df[self.df['Category'] == 'CAPEX']['Сумма'].sum())
        
        # Финансовый поток (кредиты/вливания) - это "допинг", он не является прибылью
        fin_flow = self.df[self.df['Category'] == 'FINANCIAL_FLOW']['Сумма'].sum()
        
        current_balance = self.df['Сумма'].sum()

        print(f"📊 ФАКТ (DATA):")
        print(f"   Выручка от клиентов:  {op_income:,.2f} руб.")
        print(f"   Переменные расходы:   {op_expense_abs:,.2f} руб.")
        print(f"   Постоянные расходы:   {fixed_expense_abs:,.2f} руб.")
        print(f"   Вложения (CAPEX):     {capex_abs:,.2f} руб.")
        print(f"   Финансовая подпитка:  {fin_flow:,.2f} руб.")
        print(f"   Остаток (Cash):       {current_balance:,.2f} руб.")

        # 2. ПРИМЕНЕНИЕ ЗАКОНОВ ФИЗИКИ (PINN Logic)
        
        # Закон 1: Скрытая амортизация
        # Если был CAPEX, он генерирует виртуальный ежемесячный расход на износ.
        # Допустим, срок жизни оборудования 24 месяца.
        hidden_amortization = capex_abs / 24
        
        # Закон 2: Отложенный налог (Tax Liability)
        # Налог считается с прихода ("кассовый метод"), но платится позже. Деньги на счете "грязные".
        hidden_tax_debt = op_income * self.tax_rate

        # Реальная операционная прибыль (Unit Economics Level)
        # Выручка - Переменные - Налог
        unit_margin = op_income - op_expense_abs - hidden_tax_debt

        print("\n⚖️  ПРОВЕРКА ЗАКОНОВ СОХРАНЕНИЯ:")
        if unit_margin < 0:
            print("   [!] ЮНИТ-ЭКОНОМИКА ОТРИЦАТЕЛЬНА. Каждая продажа генерирует убыток.")
        else:
            print("   [OK] Маржинальность положительная.")

        # 3. СТРЕСС-ТЕСТ МАСШТАБИРОВАНИЯ (Scale 5x)
        print("\n🚀 СИМУЛЯЦИЯ РОСТА В 5 РАЗ (Сценарий масштабирования):")
        
        scale = 5.0
        
        # Линейный рост выручки и переменных расходов
        sim_income = op_income * scale
        sim_op_expense = op_expense_abs * scale
        
        # НЕЛИНЕЙНЫЙ рост постоянных расходов (Step Cost Function)
        # При росте в 5 раз, управление дорожает не в 5, а примерно в 3.5 раза (эффект масштаба), 
        # НО если база маленькая, может потребоваться офис и штат (скачок).
        # Модель PINN предполагает скачкообразную функцию:
        sim_fixed_expense = fixed_expense_abs * 3.5 
        
        # Налог растет линейно
        sim_tax_debt = sim_income * self.tax_rate
        
        # Амортизация растет, так как для x5 нужно x5 оборудования
        sim_amortization = hidden_amortization * scale

        # ИТОГОВОЕ УРАВНЕНИЕ ПРИБЫЛЬНОСТИ (Net Profit Equation)
        sim_net_profit = sim_income - sim_op_expense - sim_fixed_expense - sim_tax_debt - sim_amortization

        # ИТОГОВОЕ УРАВНЕНИЕ КЭША (Cash Flow Equation)
        # Будет ли кассовый разрыв? 
        # Предположим, кредитов больше не дают (fin_flow = 0).
        # Нам нужно обеспечить оборотку.
        sim_cash_balance = sim_income - sim_op_expense - sim_fixed_expense
        
        print(f"   Прогноз выручки:       {sim_income:,.2f}")
        print(f"   Прогноз чистой прибыли:{sim_net_profit:,.2f} (с учетом налогов и износа)")
        print(f"   Скрытый налоговый долг:{sim_tax_debt:,.2f}")

        return sim_net_profit, sim_cash_balance, sim_tax_debt

    def get_verdict(self, net_profit, cash_balance, tax_debt):
        print("\n" + "="*50)
        print(" ВЕРДИКТ ИСКУССТВЕННОГО ИНТЕЛЛЕКТА ")
        print("="*50)
        
        if net_profit < 0:
            print("🛑 КАТЕГОРИЧЕСКИЙ ЗАПРЕТ МАСШТАБИРОВАНИЯ")
            print("Диагноз: Масштабирование убытков.")
            print("При росте оборота ваша структура расходов сожрет весь капитал.")
            print("Вы будете работать больше, а долгов станет больше.")
            print(f"Дыра в балансе составит: {net_profit:,.2f} руб.")
            
        elif cash_balance < tax_debt:
            print("⚠️ ОПАСНОСТЬ КАССОВОГО РАЗРЫВА (НАЛОГОВАЯ ЛОВУШКА)")
            print("Бизнес операционно прибылен, НО:")
            print("Денег на счете не хватит для уплаты налогов в конце периода.")
            print("Вы тратите государственные деньги на оборотку.")
            
        else:
            print("✅ ЗЕЛЕНЫЙ СВЕТ")
            print("Модель устойчива. Физика бизнеса соблюдена.")
            print("Прибыль покрывает нелинейный рост постоянных расходов и налоги.")

# ==========================================
# MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    print("Финансовый AI-Аналитик v1.0 (PINN Core)")
    
    # 1. Получение файла
    # Создадим демо-файл, если пользователь нажмет Enter
    user_path = input("Введите путь к файлу выгрузки (.txt) или Enter для демо-режима: ").strip()
    
    if not user_path:
        print(">> Активирован ДЕМО-РЕЖИМ (Создаем тестовый файл 1C...)")
        demo_content = """
1CClientBankExchange
ВерсияФормата=1.03
РасчСчет=40702810101770007143
СекцияРасчСчет
КонецРасчСчет
СекцияДокумент=Банковский ордер
Дата=20.05.2026
Сумма=500000.00
ПлательщикРасчСчет=40702810101770007143
ПолучательРасчСчет=12345
НазначениеПлатежа=Оплата за поставку материалов по договору
КонецДокумента
СекцияДокумент=Банковский ордер
Дата=21.05.2026
Сумма=600000.00
ПлательщикРасчСчет=55555
ПолучательРасчСчет=40702810101770007143
НазначениеПлатежа=Поступление оплаты от клиента за услуги разработки
КонецДокумента
СекцияДокумент=Банковский ордер
Дата=22.05.2026
Сумма=100000.00
ПлательщикРасчСчет=40702810101770007143
ПолучательРасчСчет=66666
НазначениеПлатежа=Аренда офиса за май
КонецДокумента
"""
        with open("demo_1c.txt", "w", encoding='utf-8') as f:
            f.write(demo_content.strip())
        user_path = "demo_1c.txt"

    try:
        # Шаг 1: Парсинг
        parser = BankParser(user_path)
        df = parser.parse()
        
        if df.empty:
            print("Файл пуст или не содержит транзакций.")
            exit()

        # Шаг 2: ИИ Классификация
        ai = AIClassifier()
        df_classified = ai.classify(df)
        
        # Вывод разметки для проверки пользователем
        print("\n[Проверка понимания ИИ]:")
        print(df_classified[['Назначение', 'Сумма', 'Category']].to_string(index=False, max_rows=10))

        # Шаг 3: Физическая симуляция
        engine = PINN_Engine(df_classified, tax_rate=0.06) # УСН 6%
        profit, cash, tax = engine.run_simulation()
        
        # Шаг 4: Вердикт
        engine.get_verdict(profit, cash, tax)
        
        # Очистка демо файла
        if user_path == "demo_1c.txt":
            os.remove("demo_1c.txt")

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("Убедитесь, что файл имеет формат 1CClientBankExchange")
