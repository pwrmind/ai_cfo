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
        self.real_end_balance = None  # реальный остаток из шапки файла

    def parse(self):
        """
        Читает файл 1CClientBankExchange и возвращает (DataFrame транзакций, конечный остаток).
        При отсутствии расчётного счёта выбрасывает исключение.
        """
        content = ""
        # Приоритетные кодировки: utf-8-sig (для BOM), cp1251 (стандарт 1С), utf-8, ibm866 (DOS)
        encodings = ['utf-8-sig', 'cp1251', 'utf-8', 'ibm866']
        
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
            # Пропуск комментариев (строка начинается с //)
            if line.startswith('//'):
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

            # Резервный поиск счета в глобальной шапке
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
                        # Всегда работаем с модулем, знак определяем только по логике
                        abs_amount = abs(amount)
                        if self.my_account and payer_acc == self.my_account:
                            signed_amount = -abs_amount
                            direction = "Rashod"
                        else:
                            signed_amount = abs_amount
                            direction = "Prihod"

                        # Определение контрагента с учётом альтернативных названий полей
                        if direction == "Rashod":
                            counterparty = current_tx.get('Получатель') or current_tx.get('ПолучательНаим', '')
                        else:
                            counterparty = current_tx.get('Плательщик') or current_tx.get('ПлательщикНаим', '')

                        transactions.append({
                            'Дата': current_tx.get('Дата', ''),
                            'Назначение': current_tx.get('НазначениеПлатежа', ''),
                            'Сумма': signed_amount,
                            'Контрагент': counterparty,
                        })
                    except ValueError:
                        continue  # пропуск битых строк
                continue

            if in_doc and '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    current_tx[parts[0]] = parts[1]

        if not self.my_account:
            raise ValueError(
                "❌ Расчетный счет не найден в файле. Невозможно определить направление платежей (приход/расход)."
            )

        # Формируем DataFrame даже если транзакций нет
        df = pd.DataFrame(transactions, columns=['Дата', 'Назначение', 'Сумма', 'Контрагент'])
        return df, self.real_end_balance


# ==========================================
# МОДУЛЬ 2: ГИБРИДНЫЙ КЛАССИФИКАТОР (REGEX + MCC + AI + KEYWORDS)
# ==========================================
class AIClassifier:
    def __init__(self, model_name="cointegrated/rubert-tiny2"):
        print(f"\n[Система] Инициализация гибридного мозга (MCC + {model_name} + keywords)...")
        self.model = None
        self.use_ai = False
        try:
            self.model = SentenceTransformer(model_name)
            self.use_ai = True
            print("[Система] Модель загружена успешно.")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")
            print("[Система] Переключение в режим классификации только по MCC и ключевым словам.")
        
        # 1. СПРАВОЧНИК MCC (Hard Rules)
        self.mcc_codes = {
            "5411": "OPERATING_EXPENSE",
            "5812": "OPERATING_EXPENSE",
            "5814": "OPERATING_EXPENSE",
            "5999": "OPERATING_EXPENSE",
            "5732": "CAPEX",
            "5942": "OPERATING_EXPENSE",
            "4814": "FIXED_EXPENSE",
            "7372": "FIXED_EXPENSE",
            "6011": "FINANCIAL_FLOW",
            "6538": "FINANCIAL_FLOW",
            "5499": "OPERATING_EXPENSE"
        }

        # 2. СЕМАНТИЧЕСКИЕ ЯКОРЯ (только если модель доступна)
        self.categories = {
            "OPERATING_INCOME": ["Оплата от клиента", "Поступление выручки", "Розничная выручка", "Оплата по счету"],
            "OPERATING_EXPENSE": ["Закупка товара", "Логистика", "Комиссия банка", "Хозтовары", "Материалы", "ГСМ", "сырьё"],
            "FIXED_EXPENSE": [
                "Аренда офиса", "Зарплата", "Бухгалтерия", "Интернет", "Налоги",
                "SMS информирование", "SMS-оповещение", "Комиссия за обслуживание счета",
                "Банковская комиссия за ведение счета", "Плата за ведение счета"
            ],
            "CAPEX": ["Покупка оборудования", "Компьютеры", "Мебель", "Автомобиль", "Основные средства"],
            "FINANCIAL_FLOW": ["Взнос наличных", "Перевод собственных средств", "Пополнение счета", "Уставный капитал", "Займ"]
        }
        
        self.anchors = {}
        if self.use_ai:
            print("[Система] Калибровка векторов...", end="\r")
            for cat, texts in self.categories.items():
                self.anchors[cat] = np.mean(self.model.encode(texts), axis=0)
            print("[Система] Калибровка завершена.   ")
        
        # 3. Резервная карта ключевых слов (используется всегда, если AI не сработал или отсутствует)
        self.keyword_map = {
            "OPERATING_INCOME": ["выручк", "оплат", "поступлен", "эквайринг", "доход"],
            "OPERATING_EXPENSE": ["закупк", "товар", "логистик", "комисси", "хозтовар", "гсм", "материал", "канцеляр"],
            "FIXED_EXPENSE": ["аренд", "зарплат", "бухгалтер", "интернет", "налог", "sms", "связ", "коммунальн",
                              "обслуживание счета", "ведение счета"],
            "CAPEX": ["оборудован", "компьютер", "мебел", "автомобил", "основных средств"],
            "FINANCIAL_FLOW": ["взнос наличных", "перевод собственных средств", "пополнение счета",
                               "уставный капитал", "займ", "кредит", "card2card", "corpcards", "p2p"]
        }

    def clean_text(self, text):
        """Удаляет технический мусор, оставляя смысловую часть"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'Расчеты через ТУ\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\\RU\\[A-Za-z0-9\s]+\\', ' ', text)
        text = re.sub(r'по чеку\s+[\d\.,]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{6}\+{4,}\d{4}', '', text)          # маски карт
        text = re.sub(r'\d{2}\.\d{2}\.\d{2,4}', '', text)     # даты
        text = re.sub(r'MP-?\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'MCC:?\s?\d+', '', text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def get_mcc(self, text):
        """Гибкий поиск MCC-кода (MCC5411, MCC 5411, MCC: 5411 и т.п.)"""
        if not isinstance(text, str):
            return None
        match = re.search(r'MCC[:\s]*(\d{4})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _classify_by_keywords(self, cleaned_text):
        """Поиск категории по вхождению ключевых слов (fallback)."""
        for cat, keywords in self.keyword_map.items():
            if any(kw in cleaned_text.lower() for kw in keywords):
                return cat
        return None

    def classify(self, df):
        """Классифицирует транзакции, добавляет столбец 'Category'"""
        print("[Система] Классификация транзакций...")
        if df.empty:
            df['Category'] = []
            return df

        results = []
        for idx, row in df.iterrows():
            txt = row['Назначение']
            amount = row['Сумма']

            # 0. Явные паттерны (детерминированные правила)
            if re.search(r'CARD2CARD|CORPCARDS|P2P', txt, re.IGNORECASE):
                results.append("FINANCIAL_FLOW")
                continue

            # A. Проверка MCC (золотой стандарт)
            mcc = self.get_mcc(txt)
            if mcc and mcc in self.mcc_codes:
                results.append(self.mcc_codes[mcc])
                continue

            # B. Семантический анализ (ИИ или ключевые слова)
            cleaned = self.clean_text(txt)
            if len(cleaned) < 3:
                # Эвристика по сумме, если текст пустой
                if amount < 0:
                    results.append("OPERATING_EXPENSE")  # скорее всего комиссия или покупка
                else:
                    results.append("OTHER")
                continue

            category = None

            # B1. Основной путь: SentenceTransformer (если доступен)
            if self.use_ai:
                vec = self.model.encode(cleaned)
                best_cat = "OTHER"
                max_sim = -1
                for cat, anchor in self.anchors.items():
                    sim = cosine_similarity(vec.reshape(1, -1), anchor.reshape(1, -1))[0][0]
                    if sim > max_sim:
                        max_sim = sim
                        best_cat = cat
                if best_cat != "OTHER" and max_sim > 0.30:
                    category = best_cat

            # B2. Если AI не дал уверенного ответа (или отсутствует), пробуем ключевые слова
            if category is None:
                category = self._classify_by_keywords(cleaned)

            # B3. Если всё ещё не определили — OTHER
            if category is None:
                category = "OTHER"

            results.append(category)

        df['Category'] = results
        return df


# ==========================================
# МОДУЛЬ 3: СИМУЛЯТОР МАСШТАБИРОВАНИЯ БИЗНЕСА
# ==========================================
class ForecastEngine:
    def __init__(self, dataframe, end_balance=None, tax_regime='income', custom_tax_rate=None,
                 scale_factor=5.0, fixed_exp_growth=3.5):
        self.df = dataframe
        self.end_balance = end_balance
        self.tax_regime = tax_regime          # 'income', 'profit', 'custom'
        self.custom_tax_rate = custom_tax_rate  # используется при tax_regime='custom'
        self.scale_factor = scale_factor
        self.fixed_exp_growth = fixed_exp_growth

    def _calculate_tax(self, income, expense, fixed, amort):
        """Расчёт налога в зависимости от режима."""
        if self.tax_regime == 'income':
            # УСН Доходы 6%
            rate = 0.06
            tax = income * rate
        elif self.tax_regime == 'profit':
            # УСН Доходы минус расходы 15%
            rate = 0.15
            profit_before_tax = income - expense - fixed - amort
            tax = max(0, profit_before_tax) * rate
            # Минимальный налог 1% от дохода
            min_tax = income * 0.01
            tax = max(tax, min_tax)
        elif self.tax_regime == 'custom' and self.custom_tax_rate is not None:
            # Произвольная ставка от выручки
            tax = income * self.custom_tax_rate
        else:
            tax = 0.0
        return tax

    def run_simulation(self):
        print("\n" + "="*50)
        print(" СИМУЛЯЦИЯ МАСШТАБИРОВАНИЯ БИЗНЕСА ")
        print("="*50)

        # Агрегация данных
        sums = self.df.groupby('Category')['Сумма'].sum()
        
        op_income = sums.get('OPERATING_INCOME', 0.0)
        op_expense = abs(sums.get('OPERATING_EXPENSE', 0.0))
        fixed_expense = abs(sums.get('FIXED_EXPENSE', 0.0))
        capex = abs(sums.get('CAPEX', 0.0))
        
        # Разбивка финансовых потоков на приток и отток
        fin_flow_df = self.df[self.df['Category'] == 'FINANCIAL_FLOW']['Сумма']
        fin_inflow = fin_flow_df[fin_flow_df > 0].sum()
        fin_outflow = fin_flow_df[fin_flow_df < 0].sum()
        fin_net = fin_inflow + fin_outflow
        
        # Вывод фактических данных
        print(f"📊 ФАКТ (DATA):")
        print(f"   Выручка (Revenue):        {op_income:>12,.2f} руб.")
        print(f"   Переменные расходы (COGS):{op_expense:>12,.2f} руб.")
        print(f"   Постоянные расходы (Opex):{fixed_expense:>12,.2f} руб.")
        print(f"   Инвестиции (Capex):       {capex:>12,.2f} руб.")
        print(f"   Финансовые потоки:")
        print(f"      Приток (поступления):  {fin_inflow:>12,.2f} руб.")
        print(f"      Отток (переводы/изъятия):{fin_outflow:>12,.2f} руб.")
        print(f"      Нетто-поток:           {fin_net:>12,.2f} руб.")
        if self.end_balance is not None:
            print(f"   💰 Банковский остаток:    {self.end_balance:>12,.2f} руб.")
        else:
            print(f"   💰 Банковский остаток:    (неизвестен)")
        print(f"   -----------------------------------")

        # Юнит-экономика
        unit_profit = op_income - op_expense
        print(f"⚖️  ЮНИТ-ЭКОНОМИКА: {'✅ Положительная' if unit_profit > 0 else '❌ ОТРИЦАТЕЛЬНАЯ'}")
        if unit_profit < 0:
            print("   (!) Прямые расходы превышают выручку – базовый бизнес убыточен.")

        # Фактический налог (для информации)
        actual_tax = self._calculate_tax(op_income, op_expense, fixed_expense, capex/12)
        print(f"   💼 Расчётный налог за период: {actual_tax:,.2f} руб. (режим: {self.tax_regime})")

        # Прогноз при масштабировании
        print(f"\n🚀 ПРОГНОЗ РОСТА (x{self.scale_factor}):")
        
        sim_income = op_income * self.scale_factor
        sim_op_expense = op_expense * self.scale_factor
        
        # Нелинейный рост постоянных расходов
        if fixed_expense == 0:
            print("   ⚠️ Постоянные расходы в отчётном периоде отсутствуют. В реальности они могут появиться.")
            sim_fixed_expense = 0.0
        else:
            sim_fixed_expense = fixed_expense * self.fixed_exp_growth
        
        # Амортизация капитальных затрат (условно на 12 месяцев)
        sim_amortization = (capex * self.scale_factor) / 12
        
        # Налог по выбранному режиму
        sim_tax = self._calculate_tax(sim_income, sim_op_expense, sim_fixed_expense, sim_amortization)
        
        # Чистая прибыль (финансовые потоки не учитываем – это не операционная деятельность)
        sim_net_profit = sim_income - sim_op_expense - sim_fixed_expense - sim_tax - sim_amortization

        print(f"   Прогноз выручки:         {sim_income:>12,.2f}")
        print(f"   Переменные расходы:      {sim_op_expense:>12,.2f}")
        print(f"   Постоянные расходы:      {sim_fixed_expense:>12,.2f} (x{self.fixed_exp_growth} от исходных)")
        print(f"   Налоги:                  {sim_tax:>12,.2f} ({self.tax_regime})")
        print(f"   Амортизация Capex:       {sim_amortization:>12,.2f}")
        print(f"   -----------------------------------")
        print(f"   📉 ПРОГНОЗ ПРИБЫЛИ:      {sim_net_profit:>12,.2f} руб.")

        return sim_net_profit

    def get_verdict(self, net_profit):
        print("\n" + "="*50)
        print(" ВЕРДИКТ СИСТЕМЫ ")
        print("="*50)
        
        if net_profit < 0:
            print("🛑 ВЫСОКИЙ РИСК МАСШТАБИРОВАНИЯ")
            print(f"   Прогнозируемый убыток: {net_profit:,.2f} руб.")
            print("   Причина: при текущей структуре расходов рост приводит к отрицательной чистой прибыли.")
            print("   Рекомендация: пересмотреть ценовую политику, сократить издержки или найти источники финансирования.")
        else:
            print("✅ ЗЕЛЕНЫЙ СВЕТ. Бизнес-модель масштабируема.")
            print(f"   Прогнозируемая чистая прибыль при масштабировании: {net_profit:,.2f} руб.")


# ==========================================
# ИНСТРУМЕНТЫ ВЫВОДА
# ==========================================
def print_classification_summary(df):
    """Сводная статистика по классифицированным категориям."""
    if df.empty:
        print("Нет данных для отображения.")
        return
    summary = df.groupby('Category').agg(
        Количество=('Сумма', 'count'),
        Общая_сумма=('Сумма', 'sum')
    ).sort_values('Общая_сумма', ascending=False)
    
    print("\n📊 СВОДКА ПО КАТЕГОРИЯМ:")
    for cat, row in summary.iterrows():
        print(f"   {cat:<25} {int(row['Количество']):>4} шт.   {row['Общая_сумма']:>12,.2f} руб.")


def select_tax_regime():
    """Интерактивный выбор налогового режима."""
    print("\nВыберите налоговый режим:")
    print("1 - УСН Доходы (6%)")
    print("2 - УСН Доходы минус расходы (15%, но не менее 1% от выручки)")
    print("3 - Другая ставка (указать вручную)")
    choice = input("Введите номер режима (1-3, по умолчанию 1): ").strip()
    if choice == '2':
        return 'profit', None
    elif choice == '3':
        try:
            rate = float(input("Введите ставку налога от выручки (например, 0.06 для 6%): "))
            return 'custom', rate
        except ValueError:
            print("⚠️ Некорректная ставка. Будет использован режим УСН Доходы (6%).")
            return 'income', None
    else:
        return 'income', None


# ==========================================
# ТОЧКА ВХОДА
# ==========================================
def main():
    print("AI CFO v1.4 – Система анализа выписки 1С")
    user_path = input("Путь к файлу 1C (.txt): ").strip().strip('"')
    
    if not user_path or not os.path.exists(user_path):
        print("❌ Файл не найден. Проверьте путь.")
        return

    try:
        # 1. Парсинг
        parser = BankParser(user_path)
        df, balance = parser.parse()
        
        if df.empty:
            print("❌ Файл не содержит транзакций или имеет неверный формат.")
            return
        
        print(f"✅ Загружено {len(df)} транзакций.")
        
        # 2. Классификация
        classifier = AIClassifier()
        df_classified = classifier.classify(df)
        
        # Покажем последние 5 записей для беглого просмотра
        pd.set_option('display.max_colwidth', 60)
        print("\n[Пример последних транзакций]:")
        print(df_classified[['Назначение', 'Сумма', 'Category']].tail(5).to_string(index=False))
        
        # Сводная статистика
        print_classification_summary(df_classified)
        
        # 3. Выбор налогового режима
        tax_regime, custom_rate = select_tax_regime()
        
        # 4. Симуляция масштабирования
        engine = ForecastEngine(
            dataframe=df_classified,
            end_balance=balance,
            tax_regime=tax_regime,
            custom_tax_rate=custom_rate,
            scale_factor=5.0,
            fixed_exp_growth=3.5
        )
        profit = engine.run_simulation()
        engine.get_verdict(profit)
        
    except ValueError as ve:
        print(f"❌ Ошибка в данных: {ve}")
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n⏹️ Прервано пользователем.")


if __name__ == "__main__":
    main()