import os
import re
import sys
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
        self.real_end_balance = None  # Реальный остаток из шапки файла

    def parse(self):
        """Читает файл 1CClientBankExchange и возвращает DataFrame + Остаток"""
        # 1. Подбор кодировки (CP1251 стандарт для 1С, но бывает UTF-8)
        content = ""
        encodings = ['cp1251', 'utf-8', 'windows-1251', 'ibm866']
        
        for enc in encodings:
            try:
                with open(self.filepath, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if not content:
            raise ValueError("❌ Не удалось прочитать файл. Проверьте кодировку.")

        lines = content.split('\n')
        transactions = []
        current_tx = {}
        in_doc = False
        in_header = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # --- БЛОК 1: Поиск нашего счета и остатка ---
            if line.startswith('СекцияРасчСчет'):
                in_header = True
                continue
            if in_header and line.startswith('РасчСчет='):
                if not self.my_account:
                    self.my_account = line.split('=')[1].strip()
            if in_header and line.startswith('КонечныйОстаток='):
                try:
                    self.real_end_balance = float(line.split('=')[1])
                except ValueError:
                    self.real_end_balance = 0.0
            if in_header and line.startswith('КонецРасчСчет'):
                in_header = False

            # Если счет не найден в секции, ищем в глобальной шапке
            if line.startswith('РасчСчет=') and not self.my_account:
                self.my_account = line.split('=')[1].strip()

            # --- БЛОК 2: Парсинг транзакций ---
            if line.startswith('СекцияДокумент='):
                in_doc = True
                current_tx = {}
                continue
            
            if line.startswith('КонецДокумента'):
                in_doc = False
                if 'Сумма' in current_tx:
                    try:
                        amount = float(current_tx.get('Сумма', 0))
                        payer_acc = current_tx.get('ПлательщикРасчСчет')
                        
                        # Логика знаков: Плательщик МЫ = Расход (минус)
                        # Важно: Сравниваем с self.my_account
                        if self.my_account and payer_acc == self.my_account:
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
                    except ValueError:
                        continue # Пропуск битых строк
                continue

            if in_doc and '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    current_tx[parts[0]] = parts[1]

        if not self.my_account:
             print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Расчетный счет не найден в файле. Логика +/- может быть нарушена.")
        
        return pd.DataFrame(transactions), self.real_end_balance

# ==========================================
# МОДУЛЬ 2: ГИБРИДНЫЙ КЛАССИФИКАТОР (REGEX + MCC + AI)
# ==========================================
class AIClassifier:
    def __init__(self):
        print("\n[Система] Инициализация гибридного мозга (MCC + LaBSE)...")
        try:
            self.model = SentenceTransformer("sentence-transformers/LaBSE")
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            sys.exit(1)
        
        # 1. СПРАВОЧНИК MCC (Hard Rules)
        self.mcc_codes = {
            "5411": "OPERATING_EXPENSE", # Супермаркеты
            "5812": "OPERATING_EXPENSE", # Рестораны
            "5814": "OPERATING_EXPENSE", # Фастфуд
            "5999": "OPERATING_EXPENSE", # Разное
            "5732": "CAPEX",             # Электроника
            "5942": "OPERATING_EXPENSE", # Канцелярия
            "4814": "FIXED_EXPENSE",     # Телеком
            "7372": "FIXED_EXPENSE",     # IT услуги
            "6011": "FINANCIAL_FLOW",    # Наличные
            "6538": "FINANCIAL_FLOW",    # Переводы Card2Card
            "5499": "OPERATING_EXPENSE"  # Продукты разное
        }

        # 2. СЕМАНТИЧЕСКИЕ ЯКОРЯ (Soft Rules)
        self.categories = {
            "OPERATING_INCOME": ["Оплата от клиента", "Поступление выручки", "Розничная выручка", "Оплата по счету"],
            "OPERATING_EXPENSE": ["Закупка товара", "Логистика", "Комиссия банка", "Хозтовары", "Материалы", "ГСМ"],
            "FIXED_EXPENSE": ["Аренда офиса", "Зарплата", "Бухгалтерия", "Интернет", "Налоги", "SMS информирование"],
            "CAPEX": ["Покупка оборудования", "Компьютеры", "Мебель", "Автомобиль", "Основные средства"],
            "FINANCIAL_FLOW": ["Взнос наличных", "Перевод собственных средств", "Пополнение счета", "Уставный капитал", "Займ"]
        }
        
        self.anchors = {}
        print("[Система] Калибровка векторов...", end="\r")
        for cat, texts in self.categories.items():
            self.anchors[cat] = np.mean(self.model.encode(texts), axis=0)
        print("[Система] Калибровка завершена.   ")

    def clean_text(self, text):
        """Удаляет технический мусор для ИИ"""
        if not isinstance(text, str): return ""
        # Удаляем стандартные префиксы 1С / Банков
        text = re.sub(r'Расчеты через ТУ\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\\RU\\[A-Za-z0-9\s]+\\', ' ', text) 
        text = re.sub(r'по чеку\s+[\d\.,]+', '', text, flags=re.IGNORECASE)
        # Маски карт и даты
        text = re.sub(r'\d{6}\+{4,}\d{4}', '', text) 
        text = re.sub(r'\d{2}\.\d{2}\.\d{2,4}', '', text)
        # MP/MCC мусор
        text = re.sub(r'MP-?\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'MCC:?\s?\d+', '', text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def get_mcc(self, text):
        """Гибкий поиск MCC кода"""
        if not isinstance(text, str): return None
        # Ловит: MCC5411, MCC 5411, MCC: 5411
        match = re.search(r'MCC[:\s]*(\d{4})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def classify(self, df):
        print("[Система] Классификация транзакций...")
        results = []
        
        for txt in df['Назначение']:
            # A. Проверка MCC (Золотой стандарт)
            mcc = self.get_mcc(txt)
            if mcc and mcc in self.mcc_codes:
                results.append(self.mcc_codes[mcc])
                continue 
            
            # B. Семантический анализ
            cleaned = self.clean_text(txt)
            if len(cleaned) < 3:
                 # Если после очистки ничего не осталось, но сумма минус -> скорее всего расход (комиссия/покупка)
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
            
            # Порог уверенности
            results.append(best_cat if max_sim > 0.30 else "OTHER")
        
        df['Category'] = results
        return df

# ==========================================
# МОДУЛЬ 3: PINN-СИМУЛЯТОР (ФИЗИКА БИЗНЕСА)
# ==========================================
class PINN_Engine:
    def __init__(self, dataframe, end_balance, tax_rate=0.06):
        self.df = dataframe
        self.end_balance = end_balance if end_balance is not None else dataframe['Сумма'].sum()
        self.tax_rate = tax_rate

    def run_simulation(self):
        print("\n" + "="*50)
        print(" ЗАПУСК PINN-СИМУЛЯЦИИ (Physics-Informed Logic) ")
        print("="*50)

        # Агрегация фактов (Обработка пустых категорий через get)
        sums = self.df.groupby('Category')['Сумма'].sum()
        
        op_income = sums.get('OPERATING_INCOME', 0.0)
        # Расходы берем по модулю
        op_expense = abs(sums.get('OPERATING_EXPENSE', 0.0))
        fixed_expense = abs(sums.get('FIXED_EXPENSE', 0.0))
        capex = abs(sums.get('CAPEX', 0.0))
        fin_flow = sums.get('FINANCIAL_FLOW', 0.0)
        
        print(f"📊 ФАКТ (DATA):")
        print(f"   Выручка (Revenue):    {op_income:,.2f} руб.")
        print(f"   Переменные (COGS):    {op_expense:,.2f} руб.")
        print(f"   Постоянные (Fixed):   {fixed_expense:,.2f} руб.")
        print(f"   Инвестиции (Capex):   {capex:,.2f} руб.")
        print(f"   Финансы (Loans/Own):  {fin_flow:,.2f} руб.")
        print(f"   -----------------------------------")
        print(f"   💰 БАНКОВСКИЙ ОСТАТОК: {self.end_balance:,.2f} руб.")

        # Проверка Юнит-Экономики
        unit_profit = op_income - op_expense
        # Если выручка 0, то маржинальность считаем "0", но выводим предупреждение
        if op_income > 0:
            margin_percent = (unit_profit / op_income) * 100
        else:
            margin_percent = 0

        print(f"\n⚖️  ЮНИТ-ЭКОНОМИКА: {'✅ Положительная' if unit_profit > 0 else '❌ ОТРИЦАТЕЛЬНАЯ'}")
        if unit_profit < 0:
            print("   (!) Внимание: Прямые расходы превышают выручку.")

        # Симуляция масштабирования x5
        print("\n🚀 ПРОГНОЗ РОСТА (x5):")
        scale = 5.0
        
        sim_income = op_income * scale
        sim_op_expense = op_expense * scale
        
        # Нелинейный рост постоянных расходов (Step Function)
        # Если сейчас постоянные > 0, то они растут скачком. Если 0 - предполагаем появление
        sim_fixed_expense = fixed_expense * 3.5 
        if fixed_expense == 0: sim_fixed_expense = 150000 # Минимальный офис+сотрудник
            
        # Налог считается от "грязной" выручки
        sim_tax = sim_income * self.tax_rate
        
        # Амортизация Capex (размазываем на 12 мес для модели)
        sim_amortization = (capex * scale) / 12
        
        # Чистая прибыль (Net Profit)
        sim_net_profit = sim_income - sim_op_expense - sim_fixed_expense - sim_tax - sim_amortization

        print(f"   Прогноз выручки:       {sim_income:,.2f}")
        print(f"   Скрытые налоги:        {sim_tax:,.2f}")
        print(f"   Скрытая амортизация:   {sim_amortization:,.2f}")
        print(f"   -----------------------------------")
        print(f"   📉 ПРОГНОЗ ПРИБЫЛИ:    {sim_net_profit:,.2f} руб.")

        return sim_net_profit

    def get_verdict(self, net_profit):
        print("\n" + "="*50)
        print(" ВЕРДИКТ СИСТЕМЫ ")
        print("="*50)
        
        if net_profit < 0:
            print("🛑 КАТЕГОРИЧЕСКИЙ ЗАПРЕТ МАСШТАБИРОВАНИЯ")
            print(f"Дыра в балансе составит: {net_profit:,.2f} руб.")
            print("Причина: При текущей структуре расходов рост убивает прибыль.")
        else:
            print("✅ ЗЕЛЕНЫЙ СВЕТ. Модель масштабируема.")
            print("Прибыль перекрывает нелинейный рост издержек.")

# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("AI CFO v1.1 initialized.")
    user_path = input("Путь к файлу 1C (.txt): ").strip().strip('"') # Убираем кавычки, если пользователь скопировал путь как есть
    
    if user_path and os.path.exists(user_path):
        try:
            # 1. Парсинг
            parser = BankParser(user_path)
            df, balance = parser.parse()
            
            if df.empty:
                print("❌ Файл не содержит транзакций или имеет неверный формат.")
            else:
                print(f"Загружено {len(df)} транзакций.")
                
                # 2. Классификация
                classifier = AIClassifier()
                df_classified = classifier.classify(df)
                
                # Демонстрация (последние 5 строк)
                print("\n[Пример классификации]:")
                pd.set_option('display.max_colwidth', 60)
                print(df_classified[['Назначение', 'Сумма', 'Category']].tail(5).to_string(index=False))

                # 3. Анализ
                engine = PINN_Engine(df_classified, balance)
                profit = engine.run_simulation()
                engine.get_verdict(profit)
            
        except Exception as e:
            print(f"\n❌ Критическая ошибка выполнения: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Файл не найден. Проверьте путь.")
