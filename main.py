import os
import re
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# МОДУЛЬ 1: ПАРСЕР ФОРМАТА 1C (ГЛАЗА СИСТЕМЫ)
# ==========================================
class BankParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.my_account = None
        self.real_end_balance = None

    def parse(self):
        content = ""
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
            if not line or line.startswith('//'):
                continue

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

            if line.startswith('РасчСчет=') and not self.my_account:
                self.my_account = line.split('=')[1].strip()

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
                        abs_amount = abs(amount)
                        if self.my_account and payer_acc == self.my_account:
                            signed_amount = -abs_amount
                            direction = "Rashod"
                        else:
                            signed_amount = abs_amount
                            direction = "Prihod"

                        counterparty = (
                            current_tx.get('Получатель') or current_tx.get('ПолучательНаим', '')
                            if direction == "Rashod"
                            else current_tx.get('Плательщик') or current_tx.get('ПлательщикНаим', '')
                        )
                        transactions.append({
                            'Дата': current_tx.get('Дата', ''),
                            'Назначение': current_tx.get('НазначениеПлатежа', ''),
                            'Сумма': signed_amount,
                            'Контрагент': counterparty,
                        })
                    except ValueError:
                        continue
                continue

            if in_doc and '=' in line:
                parts = line.split('=', 1)
                if len(parts) == 2:
                    current_tx[parts[0]] = parts[1]

        if not self.my_account:
            raise ValueError("❌ Расчетный счет не найден. Невозможно определить направления платежей.")

        df = pd.DataFrame(transactions, columns=['Дата', 'Назначение', 'Сумма', 'Контрагент'])
        # Преобразование даты
        try:
            df['Дата_dt'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
            if df['Дата_dt'].isna().any():
                print("⚠️ Некоторые даты не распознаны. Динамический прогноз может быть неточным.")
        except Exception:
            df['Дата_dt'] = pd.NaT
            print("⚠️ Ошибка преобразования дат. Динамический прогноз будет недоступен.")
        return df, self.real_end_balance


# ==========================================
# МОДУЛЬ 2: ГИБРИДНЫЙ КЛАССИФИКАТОР
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

        self.mcc_codes = {
            "5411": "OPERATING_EXPENSE", "5812": "OPERATING_EXPENSE",
            "5814": "OPERATING_EXPENSE", "5999": "OPERATING_EXPENSE",
            "5732": "CAPEX", "5942": "OPERATING_EXPENSE",
            "4814": "FIXED_EXPENSE", "7372": "FIXED_EXPENSE",
            "6011": "FINANCIAL_FLOW", "6538": "FINANCIAL_FLOW",
            "5499": "OPERATING_EXPENSE"
        }

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
        if not isinstance(text, str): return ""
        text = re.sub(r'Расчеты через ТУ\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\\RU\\[A-Za-z0-9\s]+\\', ' ', text)
        text = re.sub(r'по чеку\s+[\d\.,]+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d{6}\+{4,}\d{4}', '', text)
        text = re.sub(r'\d{2}\.\d{2}\.\d{2,4}', '', text)
        text = re.sub(r'MP-?\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'MCC:?\s?\d+', '', text, flags=re.IGNORECASE)
        return " ".join(text.split())

    def get_mcc(self, text):
        if not isinstance(text, str): return None
        match = re.search(r'MCC[:\s]*(\d{4})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _classify_by_keywords(self, cleaned_text):
        for cat, keywords in self.keyword_map.items():
            if any(kw in cleaned_text.lower() for kw in keywords):
                return cat
        return None

    def classify(self, df):
        print("[Система] Классификация транзакций...")
        if df.empty:
            df['Category'] = []
            return df
        results = []
        for idx, row in df.iterrows():
            txt = row['Назначение']
            amount = row['Сумма']
            if re.search(r'CARD2CARD|CORPCARDS|P2P', txt, re.IGNORECASE):
                results.append("FINANCIAL_FLOW")
                continue
            mcc = self.get_mcc(txt)
            if mcc and mcc in self.mcc_codes:
                results.append(self.mcc_codes[mcc])
                continue
            cleaned = self.clean_text(txt)
            if len(cleaned) < 3:
                results.append("OPERATING_EXPENSE" if amount < 0 else "OTHER")
                continue
            category = None
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
            if category is None:
                category = self._classify_by_keywords(cleaned)
            if category is None:
                category = "OTHER"
            results.append(category)
        df['Category'] = results
        return df


# ==========================================
# МОДУЛЬ 3: БАЗОВАЯ СИМУЛЯЦИЯ (упрощённая)
# ==========================================
class ForecastEngine:
    def __init__(self, dataframe, end_balance=None, tax_regime='income', custom_tax_rate=None,
                 scale_factor=5.0, fixed_exp_growth=3.5):
        self.df = dataframe
        self.end_balance = end_balance
        self.tax_regime = tax_regime
        self.custom_tax_rate = custom_tax_rate
        self.scale_factor = scale_factor
        self.fixed_exp_growth = fixed_exp_growth

    def _calculate_tax(self, income, expense, fixed, amort):
        if self.tax_regime == 'income':
            return income * 0.06
        elif self.tax_regime == 'profit':
            profit_before_tax = max(0, income - expense - fixed - amort)
            tax = profit_before_tax * 0.15
            min_tax = income * 0.01
            return max(tax, min_tax)
        elif self.tax_regime == 'custom' and self.custom_tax_rate is not None:
            return income * self.custom_tax_rate
        return 0.0

    def run_simulation(self):
        print("\n" + "="*50)
        print(" БАЗОВАЯ СИМУЛЯЦИЯ МАСШТАБИРОВАНИЯ ")
        print("="*50)
        sums = self.df.groupby('Category')['Сумма'].sum()
        op_income = sums.get('OPERATING_INCOME', 0.0)
        op_expense = abs(sums.get('OPERATING_EXPENSE', 0.0))
        fixed_expense = abs(sums.get('FIXED_EXPENSE', 0.0))
        capex = abs(sums.get('CAPEX', 0.0))
        fin_flow_df = self.df[self.df['Category'] == 'FINANCIAL_FLOW']['Сумма']
        fin_inflow = fin_flow_df[fin_flow_df > 0].sum()
        fin_outflow = fin_flow_df[fin_flow_df < 0].sum()
        fin_net = fin_inflow + fin_outflow

        print(f"📊 ФАКТ (DATA):")
        print(f"   Выручка:                {op_income:>12,.2f} руб.")
        print(f"   Переменные расходы:     {op_expense:>12,.2f} руб.")
        print(f"   Постоянные расходы:     {fixed_expense:>12,.2f} руб.")
        print(f"   Инвестиции (Capex):     {capex:>12,.2f} руб.")
        print(f"   Финансовые потоки:")
        print(f"      Приток:              {fin_inflow:>12,.2f} руб.")
        print(f"      Отток:               {fin_outflow:>12,.2f} руб.")
        print(f"      Нетто:               {fin_net:>12,.2f} руб.")
        if self.end_balance is not None:
            print(f"   💰 Банковский остаток:  {self.end_balance:>12,.2f} руб.")
        print(f"   -----------------------------------")
        unit_profit = op_income - op_expense
        print(f"⚖️  ЮНИТ-ЭКОНОМИКА: {'✅ Положительная' if unit_profit > 0 else '❌ ОТРИЦАТЕЛЬНАЯ'}")
        actual_tax = self._calculate_tax(op_income, op_expense, fixed_expense, capex/12)
        print(f"   💼 Расчётный налог:     {actual_tax:,.2f} руб. (режим: {self.tax_regime})")

        print(f"\n🚀 ПРОГНОЗ РОСТА (x{self.scale_factor}):")
        sim_income = op_income * self.scale_factor
        sim_op_expense = op_expense * self.scale_factor
        sim_fixed_expense = fixed_expense * self.fixed_exp_growth if fixed_expense > 0 else 0.0
        if fixed_expense == 0:
            print("   ⚠️ Постоянные расходы отсутствуют в отчётном периоде.")
        sim_amortization = (capex * self.scale_factor) / 12
        sim_tax = self._calculate_tax(sim_income, sim_op_expense, sim_fixed_expense, sim_amortization)
        sim_net_profit = sim_income - sim_op_expense - sim_fixed_expense - sim_tax - sim_amortization

        print(f"   Прогноз выручки:       {sim_income:>12,.2f}")
        print(f"   Переменные расходы:    {sim_op_expense:>12,.2f}")
        print(f"   Постоянные расходы:    {sim_fixed_expense:>12,.2f}")
        print(f"   Налоги:                {sim_tax:>12,.2f}")
        print(f"   Амортизация:           {sim_amortization:>12,.2f}")
        print(f"   -----------------------------------")
        print(f"   📉 ПРОГНОЗ ПРИБЫЛИ:    {sim_net_profit:>12,.2f} руб.")
        return sim_net_profit

    def get_verdict(self, net_profit):
        print("\n" + "="*50)
        print(" ВЕРДИКТ ")
        print("="*50)
        if net_profit < 0:
            print("🛑 ВЫСОКИЙ РИСК МАСШТАБИРОВАНИЯ")
            print(f"   Прогнозируемый убыток: {net_profit:,.2f} руб.")
            print("   Рекомендация: оптимизировать издержки.")
        else:
            print("✅ ЗЕЛЕНЫЙ СВЕТ. Модель масштабируема.")
            print(f"   Прогнозируемая прибыль: {net_profit:,.2f} руб.")


# ==========================================
# МОДУЛЬ 4: ДИНАМИЧЕСКИЙ ПРОГНОЗ (СЕЗОННОСТЬ, ДЕБИТОРКА, КАССОВЫЕ РАЗРЫВЫ)
# ==========================================
class DynamicForecastEngine:
    def __init__(self, df_classified, end_balance, config):
        self.df = df_classified
        self.end_balance = end_balance if end_balance is not None else 0.0
        self.config = config
        self.months = config.get('forecast_months', 12)
        self.seasonality = config.get('seasonality_profile', None)
        self.inflation_monthly = config.get('inflation_rate_monthly', 0.005)
        self.receivables_days = config.get('receivables_days', 0)
        self.payables_days = config.get('payables_days', 0)
        self.tax_schedule = config.get('tax_payment_schedule', 'quarterly')
        self.tax_regime = config.get('tax_regime', 'income')
        self.custom_tax_rate = config.get('custom_tax_rate', None)
        self.capex_monthly = config.get('capex_monthly', 0.0)

    def _calculate_tax(self, cumulative_income, cumulative_expense, cumulative_fixed, cumulative_amort):
        if self.tax_regime == 'income':
            return cumulative_income * 0.06
        elif self.tax_regime == 'profit':
            profit = max(0, cumulative_income - cumulative_expense - cumulative_fixed - cumulative_amort)
            tax = profit * 0.15
            min_tax = cumulative_income * 0.01
            return max(tax, min_tax)
        elif self.tax_regime == 'custom' and self.custom_tax_rate is not None:
            return cumulative_income * self.custom_tax_rate
        return 0.0

    def run(self):
        if self.df['Дата_dt'].isna().all():
            print("❌ Нет корректных дат для динамического прогноза.")
            return
        min_date = self.df['Дата_dt'].min()
        max_date = self.df['Дата_dt'].max()
        if pd.isna(min_date) or pd.isna(max_date):
            print("❌ Ошибка определения периода.")
            return

        # Группировка по месяцам
        self.df['Month'] = self.df['Дата_dt'].dt.to_period('M')
        monthly = self.df.groupby(['Month', 'Category'])['Сумма'].sum().unstack(fill_value=0)
        for cat in ['OPERATING_INCOME', 'OPERATING_EXPENSE', 'FIXED_EXPENSE', 'CAPEX', 'FINANCIAL_FLOW']:
            if cat not in monthly.columns:
                monthly[cat] = 0.0
        monthly = monthly.sort_index()

        num_hist_months = len(monthly)
        if num_hist_months == 0:
            print("❌ Нет данных для прогноза.")
            return

        avg_income = monthly['OPERATING_INCOME'].mean()
        if avg_income <= 0:
            print("⚠️ Среднемесячная выручка равна нулю или отрицательна. Динамический прогноз невозможен.")
            return

        avg_op_exp = abs(monthly['OPERATING_EXPENSE'].mean())
        avg_fixed = abs(monthly['FIXED_EXPENSE'].mean())
        hist_capex = abs(monthly['CAPEX'].mean())
        capex_base = max(self.capex_monthly, hist_capex)

        # --- Обработка сезонности ---
        if self.seasonality and len(self.seasonality) == 12:
            season_factors = self.seasonality
            print("ℹ️ Используются заданные сезонные коэффициенты.")
        else:
            # Подсчитываем количество месяцев с ненулевым доходом
            income_by_month = monthly['OPERATING_INCOME']
            positive_months = (income_by_month > 0).sum()
            if num_hist_months < 3 or positive_months < 3:
                print("⚠️ Недостаточно данных для расчёта сезонности (менее 3 месяцев с выручкой). Используются равномерные коэффициенты.")
                season_factors = [1.0] * 12
            else:
                monthly['month_num'] = monthly.index.month
                season_income = monthly.groupby('month_num')['OPERATING_INCOME'].mean()
                season_factors = []
                for m in range(1, 13):
                    val = season_income.get(m, avg_income)
                    # Защита от отрицательных или нулевых значений
                    if val <= 0:
                        val = avg_income
                    factor = val / avg_income
                    season_factors.append(factor)
                print("ℹ️ Сезонные коэффициенты рассчитаны по историческим данным.")

        start_month = max_date + relativedelta(months=1)
        balance = self.end_balance
        cum_income = 0.0
        cum_op_exp = 0.0
        cum_fixed = 0.0
        cum_amort = 0.0
        tax_paid_total = 0.0

        receivables_carry = 0.0
        payables_carry = 0.0
        immediate_income_ratio = max(0.0, 1.0 - self.receivables_days / 30.0)
        immediate_expense_ratio = max(0.0, 1.0 - self.payables_days / 30.0)

        rows = []
        warnings = []
        for i in range(self.months):
            current_date = start_month + relativedelta(months=i)
            month_idx = current_date.month - 1
            season = season_factors[month_idx]

            inflation_factor = (1 + self.inflation_monthly) ** i
            revenue = avg_income * season * inflation_factor
            op_expense = avg_op_exp * season * inflation_factor
            fixed_expense = avg_fixed * inflation_factor
            capex = capex_base * inflation_factor
            amort = capex / 12

            immediate_income = revenue * immediate_income_ratio
            cash_income = immediate_income + receivables_carry
            receivables_carry = revenue * (1 - immediate_income_ratio)

            immediate_expense_payment = op_expense * immediate_expense_ratio
            cash_op_expense = immediate_expense_payment + payables_carry
            payables_carry = op_expense * (1 - immediate_expense_ratio)

            cash_fixed = fixed_expense
            cash_capex = capex

            cum_income += revenue
            cum_op_exp += op_expense
            cum_fixed += fixed_expense
            cum_amort += amort

            total_tax_due = self._calculate_tax(cum_income, cum_op_exp, cum_fixed, cum_amort)
            tax_to_pay = total_tax_due - tax_paid_total
            if self.tax_schedule == 'monthly':
                tax_payment = tax_to_pay
                tax_paid_total += tax_payment
            elif self.tax_schedule == 'quarterly':
                if (i+1) % 3 == 0:
                    tax_payment = tax_to_pay
                    tax_paid_total += tax_payment
                else:
                    tax_payment = 0.0
            else:  # annual
                if i == self.months-1:
                    tax_payment = tax_to_pay
                    tax_paid_total += tax_payment
                else:
                    tax_payment = 0.0

            net_cash = cash_income - cash_op_expense - cash_fixed - cash_capex - tax_payment
            balance += net_cash
            if balance < 0:
                warnings.append(f"⚠️ Месяц {current_date.strftime('%Y-%m')}: кассовый разрыв (баланс = {balance:,.2f} руб.)")

            rows.append({
                'Месяц': current_date.strftime('%Y-%m'),
                'Выручка': revenue,
                'Поступления': cash_income,
                'Перем.расходы': op_expense,
                'Платежи перем.': cash_op_expense,
                'Пост.расходы': fixed_expense,
                'Capex': capex,
                'Налоги': tax_payment,
                'Чистый поток': net_cash,
                'Баланс': balance
            })

        forecast_df = pd.DataFrame(rows)
        print("\n" + "="*60)
        print(" ДИНАМИЧЕСКИЙ ПРОГНОЗ ДВИЖЕНИЯ ДЕНЕЖНЫХ СРЕДСТВ ")
        print("="*60)
        print(forecast_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        if warnings:
            print("\n⚠️ ПРЕДУПРЕЖДЕНИЯ О КАССОВЫХ РАЗРЫВАХ:")
            for w in warnings:
                print(w)
            print("Рекомендация: привлечь дополнительное финансирование или сократить отсрочки.")
        else:
            print("\n✅ За весь период кассовых разрывов не ожидается.")
        return forecast_df

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def load_config(filepath='config.yaml'):
    if not os.path.exists(filepath):
        print(f"⚠️ Файл конфигурации {filepath} не найден. Используются значения по умолчанию.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def select_tax_regime():
    print("\nВыберите налоговый режим:")
    print("1 - УСН Доходы (6%)")
    print("2 - УСН Доходы минус расходы (15%, но не менее 1% от выручки)")
    print("3 - Другая ставка (указать вручную)")
    choice = input("Введите номер (1-3, по умолчанию 1): ").strip()
    if choice == '2':
        return 'profit', None
    elif choice == '3':
        try:
            rate = float(input("Введите ставку налога от выручки (например, 0.06): "))
            return 'custom', rate
        except ValueError:
            print("⚠️ Некорректная ставка. Будет использован режим Доходы 6%.")
            return 'income', None
    else:
        return 'income', None

def print_classification_summary(df):
    if df.empty:
        return
    summary = df.groupby('Category').agg(
        Количество=('Сумма', 'count'),
        Общая_сумма=('Сумма', 'sum')
    ).sort_values('Общая_сумма', ascending=False)
    print("\n📊 СВОДКА ПО КАТЕГОРИЯМ:")
    for cat, row in summary.iterrows():
        print(f"   {cat:<25} {int(row['Количество']):>4} шт.   {row['Общая_сумма']:>12,.2f} руб.")


# ==========================================
# ТОЧКА ВХОДА
# ==========================================
def main():
    print("AI CFO v2.0 – Интеллектуальный финансовый прогноз")
    config = load_config()

    user_path = input("Путь к файлу 1C (.txt): ").strip().strip('"')
    if not user_path or not os.path.exists(user_path):
        print("❌ Файл не найден.")
        return

    try:
        # 1. Парсинг
        parser = BankParser(user_path)
        df, balance = parser.parse()
        if df.empty:
            print("❌ Файл не содержит транзакций.")
            return
        print(f"✅ Загружено {len(df)} транзакций.")

        # 2. Классификация
        classifier = AIClassifier()
        df_classified = classifier.classify(df)

        # Вывод примеров и сводки
        pd.set_option('display.max_colwidth', 60)
        print("\n[Пример последних транзакций]:")
        print(df_classified[['Назначение', 'Сумма', 'Category']].tail(5).to_string(index=False))
        print_classification_summary(df_classified)

        # 3. Выбор налогового режима
        tax_regime, custom_rate = select_tax_regime()

        # 4. Базовая симуляция (всегда)
        scale_factor = config.get('scale_factor', 5.0)
        fixed_growth = config.get('fixed_exp_growth', 3.5)
        engine_base = ForecastEngine(df_classified, balance, tax_regime, custom_rate,
                                     scale_factor, fixed_growth)
        base_profit = engine_base.run_simulation()
        engine_base.get_verdict(base_profit)

        # 5. Динамический прогноз (опционально)
        run_dynamic = input("\nЗапустить динамический прогноз с учётом сезонности, дебиторки и кассовых разрывов? (y/n): ").strip().lower()
        if run_dynamic == 'y':
            # Загружаем параметры из конфига, но можно переопределить
            dyn_config = config.get('dynamic_forecast', {})
            # Добавляем налоговый режим
            dyn_config['tax_regime'] = tax_regime
            dyn_config['custom_tax_rate'] = custom_rate

            # Опционально спрашиваем, не хочет ли пользователь изменить параметры
            print("\nТекущие параметры динамического прогноза (из config.yaml):")
            for k, v in dyn_config.items():
                print(f"  {k}: {v}")
            change = input("Изменить параметры? (y/n): ").strip().lower()
            if change == 'y':
                try:
                    dyn_config['forecast_months'] = int(input("Горизонт прогноза (мес.): ") or dyn_config.get('forecast_months', 12))
                    dyn_config['receivables_days'] = float(input("Отсрочка от клиентов (дней): ") or dyn_config.get('receivables_days', 0))
                    dyn_config['payables_days'] = float(input("Отсрочка поставщикам (дней): ") or dyn_config.get('payables_days', 0))
                    infl = input(f"Инфляция в месяц (по умолч. {dyn_config.get('inflation_rate_monthly', 0.005)}): ")
                    if infl:
                        dyn_config['inflation_rate_monthly'] = float(infl)
                    # сезонность оставим из конфига, либо спросим 12 чисел
                    print("Сезонность оставлена из config.yaml (или автоопределение, если не задана).")
                except ValueError:
                    print("⚠️ Ошибка ввода, используются значения по умолчанию.")

            dyn_engine = DynamicForecastEngine(df_classified, balance, dyn_config)
            dyn_engine.run()

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