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
        encodings = ['cp1251', 'utf-8', 'windows-1251'] 
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

        # 1. Находим наш расчетный счет
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
                    amount = float(current_tx.get('Сумма', 0))
                    payer_acc = current_tx.get('ПлательщикРасчСчет')
                    
                    # Логика знаков: Плательщик мы = Расход (минус)
                    if payer_acc == self.my_account:
                        amount = -abs(amount)
                        direction = "Rashod"
                    else:
                        amount = abs(amount)
                        direction = "Prihod"

                    transactions.append({
                        'Дата': current_tx.get('Дата'),
                        'Назначение': current_tx.get('НазначениеПлатежа', ''),
                        'Сумма': amount,
                        'Контрагент': current_tx.get('Получатель' if direction == "Rashod" else 'Плательщик'),
                    })
                continue

            if in_doc and '=' in line:
                key, value = line.split('=', 1)
                current_tx[key] = value

        return pd.DataFrame(transactions)

# ==========================================
# МОДУЛЬ 2: ГИБРИДНЫЙ КЛАССИФИКАТОР (REGEX + MCC + AI)
# ==========================================
class AIClassifier:
    def __init__(self):
        print("\n[Система] Инициализация гибридного мозга (MCC + LaBSE)...")
        self.model = SentenceTransformer("sentence-transformers/LaBSE")
        
        # 1. СПРАВОЧНИК MCC (Hard Rules)
        self.mcc_codes = {
            "5411": "OPERATING_EXPENSE", # Продукты
            "5812": "OPERATING_EXPENSE", # Рестораны
            "5814": "OPERATING_EXPENSE", # Фастфуд
            "5999": "OPERATING_EXPENSE", # Разное
            "5732": "CAPEX",             # Электроника
            "5942": "OPERATING_EXPENSE", # Канцелярия
            "4814": "FIXED_EXPENSE",     # Телеком
            "7372": "FIXED_EXPENSE",     # IT услуги
            "6011": "FINANCIAL_FLOW",    # Наличные
            "6538": "FINANCIAL_FLOW"     # Переводы
        }

        # 2. СЕМАНТИЧЕСКИЕ ЯКОРЯ (Soft Rules)
        self.categories = {
            "OPERATING_INCOME": ["Оплата от клиента", "Поступление выручки", "Розничная выручка"],
            "OPERATING_EXPENSE": ["Закупка товара", "Логистика", "Комиссия банка", "Хозтовары", "Материалы"],
            "FIXED_EXPENSE": ["Аренда офиса", "Зарплата", "Бухгалтерия", "Интернет", "Налоги", "SMS информирование"],
            "CAPEX": ["Покупка оборудования", "Компьютеры", "Мебель", "Автомобиль"],
            "FINANCIAL_FLOW": ["Взнос наличных", "Перевод собственных средств", "Пополнение счета", "Уставный капитал"]
        }
        
        self.anchors = {}
        for cat, texts in self.categories.items():
            self.anchors[cat] = np.mean(self.model.encode(texts), axis=0)

    def clean_text(self, text):
        """Удаляет технический мусор для ИИ"""
        text = re.sub(r'Расчеты через ТУ\s+\d+', '', text)
        text = re.sub(r'\\RU\\[A-Za-z]+\\', ' ', text) 
        text = re.sub(r'\d{6}\+{4,}\d{4}', '', text) # Маски карт
        text = re.sub(r'MP-\d+', '', text)
        return " ".join(text.split())

    def get_mcc(self, text):
        match = re.search(r'MCC(\d{4})', text)
        return match.group(1) if match else None

    def classify(self, df):
        print("[Система] Классификация транзакций...")
        results = []
        
        for txt in df['Назначение']:
            # A. Проверка MCC
            mcc = self.get_mcc(txt)
            if mcc and mcc in self.mcc_codes:
                results.append(self.mcc_codes[mcc])
                continue 
            
            # B. Семантический анализ
            cleaned = self.clean_text(txt)
            if len(cleaned) < 3:
                 results.append("OTHER")
                 continue

            vec = self.model.encode(cleaned)
            best_cat = "OTHER"
            max_sim = -1
            
            for cat, anchor in self.anchors.items():
                sim = cosine_similarity(vec.reshape(1, -1), anchor.reshape(1, -1))[0][0]
                if sim > max_sim:
                    max_sim = sim
                    best_cat = cat
            
            results.append(best_cat if max_sim > 0.30 else "OTHER")
        
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
        print(" ЗАПУСК PINN-СИМУЛЯЦИИ (Physics-Informed Logic) ")
        print("="*50)

        # Агрегация фактов
        op_income = self.df[self.df['Category'] == 'OPERATING_INCOME']['Сумма'].sum()
        op_expense = abs(self.df[self.df['Category'] == 'OPERATING_EXPENSE']['Сумма'].sum())
        fixed_expense = abs(self.df[self.df['Category'] == 'FIXED_EXPENSE']['Сумма'].sum())
        capex = abs(self.df[self.df['Category'] == 'CAPEX']['Сумма'].sum())
        fin_flow = self.df[self.df['Category'] == 'FINANCIAL_FLOW']['Сумма'].sum()
        
        current_balance = self.df['Сумма'].sum()

        print(f"📊 ФАКТ (DATA):")
        print(f"   Выручка:              {op_income:,.2f} руб.")
        print(f"   Переменные расходы:   {op_expense:,.2f} руб.")
        print(f"   Постоянные расходы:   {fixed_expense:,.2f} руб.")
        print(f"   Финансовая подпитка:  {fin_flow:,.2f} руб.")
        print(f"   Текущий остаток:      {current_balance:,.2f} руб.")

        # Проверка Юнит-Экономики
        unit_margin = op_income - op_expense
        if unit_margin < 0:
            print("\n⚠️ ПРЕДУПРЕЖДЕНИЕ: Отрицательная маржинальность (Unit Economics < 0)")

        # Симуляция масштабирования x5
        print("\n🚀 ПРОГНОЗ РОСТА (x5):")
        scale = 5.0
        
        sim_income = op_income * scale
        sim_op_expense = op_expense * scale
        sim_fixed_expense = fixed_expense * 3.5 # Нелинейный рост (Step Function)
        sim_tax = sim_income * self.tax_rate    # Отложенный налог
        
        sim_net_profit = sim_income - sim_op_expense - sim_fixed_expense - sim_tax

        print(f"   Прогноз выручки:       {sim_income:,.2f}")
        print(f"   Скрытые налоги:        {sim_tax:,.2f}")
        print(f"   Прогноз чистой прибыли:{sim_net_profit:,.2f}")

        return sim_net_profit

    def get_verdict(self, net_profit):
        print("\n" + "="*50)
        print(" ВЕРДИКТ СИСТЕМЫ ")
        print("="*50)
        
        if net_profit < 0:
            print("🛑 КАТЕГОРИЧЕСКИЙ ЗАПРЕТ МАСШТАБИРОВАНИЯ")
            print(f"Дыра в балансе составит: {net_profit:,.2f} руб.")
            print("Причина: Структура расходов растет быстрее выручки.")
        else:
            print("✅ ЗЕЛЕНЫЙ СВЕТ. Модель масштабируема.")

# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    user_path = input("Путь к файлу 1C (.txt): ").strip()
    if user_path:
        try:
            parser = BankParser(user_path)
            df = parser.parse()
            
            classifier = AIClassifier()
            df_classified = classifier.classify(df)
            
            print("\n[Результат классификации]:")
            print(df_classified[['Назначение', 'Сумма', 'Category']].tail(5).to_string())

            engine = PINN_Engine(df_classified)
            profit = engine.run_simulation()
            engine.get_verdict(profit)
            
        except Exception as e:
            print(f"Ошибка: {e}")
