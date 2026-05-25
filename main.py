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
import torch
import torch.nn as nn
import math

# ==========================================
# МОДУЛЬ 1: ПАРСЕР ФОРМАТА 1C
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
            if in_header and (line.startswith('РасчСчет=') or line.startswith('Счет=')):
                if not self.my_account:
                    self.my_account = line.split('=')[1].strip()
            if in_header and line.startswith('КонечныйОстаток='):
                try:
                    self.real_end_balance = float(line.split('=')[1])
                except ValueError:
                    self.real_end_balance = 0.0
            if in_header and line.startswith('КонецРасчСчет'):
                in_header = False

            if (line.startswith('РасчСчет=') or line.startswith('Счет=')) and not self.my_account:
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
                        payer_acc = (current_tx.get('ПлательщикРасчСчет') or
                                     current_tx.get('ПлательщикСчет') or '')
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
        try:
            df['Дата_dt'] = pd.to_datetime(df['Дата'], dayfirst=True, errors='coerce')
            if df['Дата_dt'].isna().any():
                print("⚠️ Некоторые даты не распознаны. Прогнозы могут быть неточными.")
        except Exception:
            df['Дата_dt'] = pd.NaT
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
            "5411": "OPERATING_EXPENSE", "5499": "OPERATING_EXPENSE",
            "5812": "OPERATING_EXPENSE", "5814": "OPERATING_EXPENSE",
            "5999": "OPERATING_EXPENSE", "5300": "OPERATING_EXPENSE",
            "4111": "OPERATING_EXPENSE", "3990": "OPERATING_EXPENSE",
            "4814": "FIXED_EXPENSE", "7372": "FIXED_EXPENSE",
            "7394": "OPERATING_EXPENSE", "8299": "OPERATING_EXPENSE",
            "5732": "CAPEX",
            "6011": "FINANCIAL_FLOW", "6538": "FINANCIAL_FLOW",
        }

        self.categories = {
            "OPERATING_INCOME": [
                "Оплата от клиента", "Поступление выручки", "Розничная выручка",
                "Оплата по счету", "Оплата по договору", "Поступление оплаты",
                "Выручка от реализации", "Эквайринг", "Терминал"
            ],
            "OPERATING_EXPENSE": [
                "Закупка товара", "Логистика", "Хозтовары", "Материалы",
                "ГСМ", "сырьё", "расходные материалы", "канцелярия",
                "транспортные услуги", "доставка"
            ],
            "FIXED_EXPENSE": [
                "Аренда офиса", "Зарплата", "Бухгалтерия", "Интернет",
                "Налоги", "SMS информирование", "SMS-оповещение",
                "Комиссия за обслуживание счета", "Банковская комиссия за ведение счета",
                "Плата за ведение счета", "Абонентская плата",
                "Информирование об операциях"
            ],
            "CAPEX": [
                "Покупка оборудования", "Компьютеры", "Мебель",
                "Автомобиль", "Основные средства", "Станок",
                "Приобретение ОС", "Капитальные вложения"
            ],
            "FINANCIAL_FLOW": [
                "Взнос наличных", "Перевод собственных средств",
                "Пополнение счета", "Уставный капитал", "Займ",
                "Кредит", "C2C", "Card2Card"
            ]
        }
        self.anchors = {}
        if self.use_ai:
            print("[Система] Калибровка векторов...", end="\r")
            for cat, texts in self.categories.items():
                self.anchors[cat] = np.mean(self.model.encode(texts), axis=0)
            print("[Система] Калибровка завершена.   ")

        self.keyword_map = {
            "OPERATING_INCOME": [
                "выручк", "оплат", "поступлен", "эквайринг", "доход",
                "реализац", "терминал", "за услуги", "по договору"
            ],
            "OPERATING_EXPENSE": [
                "закупк", "товар", "логистик", "хозтовар",
                "гсм", "материал", "канцеляр", "расход",
                "доставка", "транспорт", "услуги связи", "интернет"
            ],
            "FIXED_EXPENSE": [
                "аренд", "зарплат", "бухгалтер", "налог",
                "sms", "коммунальн", "обслуживание счета",
                "ведение счета", "ндфл", "алимент", "исполнительный лист",
                "страховые взносы", "пфр", "фсс", "штраф", "пеня",
                "абонентская плата", "информирование об операциях"
            ],
            "CAPEX": [
                "оборудован", "компьютер", "мебел", "автомобил",
                "основных средств", "станок", "техник", "покупка ОС",
                "приобретение", "капитальные вложения"
            ],
            "FINANCIAL_FLOW": [
                "взнос наличных", "перевод собственных средств",
                "пополнение счета", "уставный капитал", "займ",
                "кредит", "card2card", "corpcards", "p2p"
            ]
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

            # =====  АБСОЛЮТНЫЙ ПРИОРИТЕТ  =====
            # Эти правила выполняются первыми и не могут быть перекрыты.
            if re.search(r'уставный капитал', txt, re.IGNORECASE):
                results.append("FINANCIAL_FLOW")
                continue
            if re.search(r'CARD2CARD|CORPCARDS|P2P', txt, re.IGNORECASE):
                results.append("FINANCIAL_FLOW")
                continue
            if re.search(r'перевод собственных средств', txt, re.IGNORECASE):
                results.append("FINANCIAL_FLOW")
                continue
            if re.search(r'алимент|ндфл|страховые взносы|пфр|фсс|исполнительный лист|штраф|пеня|налог на имущество|транспортный налог',
                         txt, re.IGNORECASE):
                results.append("FIXED_EXPENSE")
                continue
            if re.search(r'обслуживание счета|ведение счета|информирование об операциях|абонентская плата',
                         txt, re.IGNORECASE):
                results.append("FIXED_EXPENSE")
                continue

            # Проверка MCC
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
                if best_cat == "OPERATING_INCOME" and amount < 0:
                    fallback = self._classify_by_keywords(cleaned)
                    if fallback and fallback != "OPERATING_INCOME":
                        category = fallback
                    else:
                        category = "OPERATING_EXPENSE"
                elif best_cat != "OTHER" and max_sim > 0.30:
                    category = best_cat

            if category is None:
                category = self._classify_by_keywords(cleaned)
            if category is None:
                category = "OTHER"

            if category == "OPERATING_INCOME" and amount < 0:
                category = "OPERATING_EXPENSE"

            results.append(category)

        df['Category'] = results
        return df


# ==========================================
# МОДУЛЬ 3: БАЗОВАЯ СИМУЛЯЦИЯ
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
# МОДУЛЬ 4: ДИНАМИЧЕСКИЙ ПРОГНОЗ
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

    def _calculate_tax_period(self, income, expense, fixed, amort):
        if self.tax_regime == 'income':
            return income * 0.06
        elif self.tax_regime == 'profit':
            profit = max(0, income - expense - fixed - amort)
            tax = profit * 0.15
            min_tax = income * 0.01
            return max(tax, min_tax)
        elif self.tax_regime == 'custom' and self.custom_tax_rate is not None:
            return income * self.custom_tax_rate
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

        self.df['Month'] = self.df['Дата_dt'].dt.to_period('M')
        monthly = self.df.groupby(['Month', 'Category'])['Сумма'].sum().unstack(fill_value=0)
        for cat in ['OPERATING_INCOME', 'OPERATING_EXPENSE', 'FIXED_EXPENSE', 'CAPEX', 'FINANCIAL_FLOW']:
            if cat not in monthly.columns:
                monthly[cat] = 0.0
        monthly = monthly.sort_index()

        if len(monthly) > 2:
            monthly_full = monthly.iloc[1:-1]
        else:
            monthly_full = monthly

        num_hist_months = len(monthly_full)
        if num_hist_months == 0:
            print("❌ Недостаточно полных месяцев для прогноза.")
            return

        avg_income = monthly_full['OPERATING_INCOME'].mean()
        if avg_income <= 0:
            print("⚠️ Среднемесячная выручка равна нулю или отрицательна.")
            return

        avg_op_exp = abs(monthly_full['OPERATING_EXPENSE'].mean())
        avg_fixed = abs(monthly_full['FIXED_EXPENSE'].mean())
        hist_capex = abs(monthly_full['CAPEX'].mean())
        capex_base = max(self.capex_monthly, hist_capex)

        if self.seasonality and len(self.seasonality) == 12:
            season_factors = self.seasonality
            print("ℹ️ Используются заданные сезонные коэффициенты.")
        else:
            positive_months = (monthly_full['OPERATING_INCOME'] > 0).sum()
            if num_hist_months < 3 or positive_months < 3:
                print("⚠️ Недостаточно данных для расчёта сезонности – коэффициенты равномерны.")
                season_factors = [1.0] * 12
            else:
                monthly_full = monthly_full.copy()
                monthly_full['month_num'] = monthly_full.index.month
                season_income = monthly_full.groupby('month_num')['OPERATING_INCOME'].median()
                season_factors = []
                for m in range(1, 13):
                    val = season_income.get(m, avg_income)
                    if val <= 0:
                        val = avg_income
                    season_factors.append(val / avg_income)
                print("ℹ️ Сезонные коэффициенты рассчитаны (медиана).")

        start_month = max_date + relativedelta(months=1)
        balance = self.end_balance

        q_income = q_expense = q_fixed = q_amort = 0.0
        period_counter = 0

        lag_recv = int(math.ceil(self.receivables_days / 30.0))
        lag_pay = int(math.ceil(self.payables_days / 30.0))
        revenue_buffer = []
        expense_buffer = []

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

            cash_income = revenue_buffer.pop(0) if revenue_buffer else 0.0
            if lag_recv == 0:
                cash_income += revenue
            else:
                if len(revenue_buffer) < lag_recv:
                    revenue_buffer.extend([0.0] * (lag_recv - len(revenue_buffer)))
                revenue_buffer[lag_recv - 1] += revenue

            cash_op_expense = expense_buffer.pop(0) if expense_buffer else 0.0
            if lag_pay == 0:
                cash_op_expense += op_expense
            else:
                if len(expense_buffer) < lag_pay:
                    expense_buffer.extend([0.0] * (lag_pay - len(expense_buffer)))
                expense_buffer[lag_pay - 1] += op_expense

            q_income += revenue
            q_expense += op_expense
            q_fixed += fixed_expense
            q_amort += amort
            period_counter += 1

            tax_payment = 0.0
            if self.tax_schedule == 'monthly':
                tax_payment = self._calculate_tax_period(revenue, op_expense, fixed_expense, amort)
            elif self.tax_schedule == 'quarterly' and period_counter == 3:
                tax_payment = self._calculate_tax_period(q_income, q_expense, q_fixed, q_amort)
                q_income = q_expense = q_fixed = q_amort = 0.0
                period_counter = 0
            elif self.tax_schedule == 'annual' and i == self.months - 1:
                total_income = sum(r['Выручка'] for r in rows) + revenue
                total_expense = sum(r['Перем.расходы'] for r in rows) + op_expense
                total_fixed = sum(r['Пост.расходы'] for r in rows) + fixed_expense
                total_amort = sum(r.get('Capex', 0)/12 for r in rows) + amort
                tax_payment = self._calculate_tax_period(total_income, total_expense, total_fixed, total_amort)

            net_cash = cash_income - cash_op_expense - fixed_expense - capex - tax_payment
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
        print(" ИСПРАВЛЕННЫЙ ДИНАМИЧЕСКИЙ ПРОГНОЗ ДВИЖЕНИЯ ДЕНЕЖНЫХ СРЕДСТВ ")
        print("="*60)
        print(forecast_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        if warnings:
            print("\n⚠️ ПРЕДУПРЕЖДЕНИЯ О КАССОВЫХ РАЗРЫВАХ:")
            for w in warnings:
                print(w)
        else:
            print("\n✅ За весь период кассовых разрывов не ожидается.")
        return forecast_df


# ==========================================
# МОДУЛЬ 5: PINN (ТЕКСТОВЫЙ ПРОГНОЗ)
# ==========================================
class TorchPINN:
    def __init__(self, df, end_balance, tax_rate=0.06):
        self.df = df
        self.end_balance = end_balance
        self.tax_rate = tax_rate
        self.growth_rate = None
        self.friction = None
        self.model = None

    def prepare_data(self):
        df = self.df.sort_values('Дата_dt').copy()
        df_op = df[df['Category'] != 'FINANCIAL_FLOW']
        if df_op.empty:
            df_op = df
            print("⚠️ Все транзакции – финансовые потоки. Результат PINN может быть неинформативным.")
        df_op['cumsum'] = df_op['Сумма'].cumsum()
        if self.end_balance is not None:
            total_op_effect = df_op['cumsum'].iloc[-1]
            df_op['balance'] = self.end_balance - total_op_effect + df_op['cumsum']
        else:
            df_op['balance'] = df_op['cumsum'] - df_op['cumsum'].iloc[0]
        start_date = df_op['Дата_dt'].iloc[0]
        df_op['days'] = (df_op['Дата_dt'] - start_date).dt.days
        t_raw = df_op['days'].values.astype(np.float32)
        y_raw = df_op['balance'].values.astype(np.float32)
        t_norm = (t_raw - t_raw.min()) / (t_raw.max() - t_raw.min() + 1e-6)
        y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-6)
        self.t_data = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1)
        self.y_data = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)
        self.t_raw = t_raw
        self.y_raw = y_raw
        self.y_min = y_raw.min()
        self.y_max = y_raw.max()

    class PINNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 20), nn.Tanh(),
                nn.Linear(20, 20), nn.Tanh(),
                nn.Linear(20, 1)
            )
        def forward(self, t):
            return self.net(t)

    def physics_loss(self, model, t, growth_rate, friction):
        balance = model(t)
        d_balance_dt = torch.autograd.grad(balance, t, torch.ones_like(balance), create_graph=True)[0]
        residual = d_balance_dt - (growth_rate * balance - friction * balance**2)
        return torch.mean(residual**2)

    def train(self, epochs=800):
        self.prepare_data()
        self.growth_rate = torch.tensor([0.5], requires_grad=True)
        self.friction = torch.tensor([0.1], requires_grad=True)
        model = self.PINNModel()
        optimizer = torch.optim.Adam(list(model.parameters()) + [self.growth_rate, self.friction], lr=0.01)
        print("\nОбучение PINN (экспериментальный модуль) ...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(self.t_data)
            loss_data = torch.mean((y_pred - self.y_data)**2)
            t_physics = torch.linspace(0, 2, 30).view(-1, 1).requires_grad_(True)
            loss_phys = self.physics_loss(model, t_physics, self.growth_rate, self.friction)
            loss = loss_data + loss_phys
            loss.backward()
            optimizer.step()
            if epoch % 300 == 0:
                print(f"Epoch {epoch:4d}: r={self.growth_rate.item():.4f}, α={self.friction.item():.4f}, Loss={loss.item():.6f}")
        self.model = model
        self.growth_rate_value = self.growth_rate.item()
        self.friction_value = self.friction.item()
        if self.friction_value > 1e-4:
            K = self.growth_rate_value / self.friction_value
            print(f"Предельный масштаб (K): {K:.3f} (в нормализ. единицах)")
        else:
            K = float('inf')
        print(f"✅ Обучение завершено. r = {self.growth_rate_value:.4f}, α = {self.friction_value:.4f}")

    def forecast_text_table(self, months_ahead=12):
        """Текстовая таблица прогноза операционного баланса на указанное количество месяцев."""
        if self.model is None:
            raise RuntimeError("Сначала обучите модель.")
        t_max = self.t_data.max().item()
        t_future_norm = torch.linspace(t_max, t_max * 2.0, months_ahead).view(-1, 1)
        with torch.no_grad():
            y_future_norm = self.model(t_future_norm).numpy()
        y_future = y_future_norm * (self.y_max - self.y_min) + self.y_min
        t_days_future = np.linspace(self.t_raw.max(), self.t_raw.max() * 2.0, months_ahead)
        start_date = self.df['Дата_dt'].min() + timedelta(days=float(self.t_raw.max()))
        dates = [start_date + timedelta(days=float(d)) for d in t_days_future]
        print("\n📈 ПРОГНОЗ PINN (операционный баланс):")
        print(f"{'Дата':<12} {'Баланс, руб.':>15}")
        for dt, bal in zip(dates, y_future):
            print(f"{dt.strftime('%Y-%m-%d'):<12} {bal[0]:>15,.2f}")

    def verdict(self):
        print("\n⚠️ Внимание: PINN – экспериментальный инструмент. Выводы могут быть недостоверны.")
        if self.friction_value > 0.5:
            print("🛑 Высокое внутреннее трение – возможны ограничения роста.")
        elif self.growth_rate_value < 0.2:
            print("⚠️ Низкая базовая скорость роста.")
        else:
            print("✅ Параметры роста выглядят благоприятно (согласно модели).")


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
    print("2 - УСН Доходы минус расходы (15%)")
    print("3 - Другая ставка")
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
    print("AI CFO v2.1 – Финальная версия (текстовый PINN)")
    config = load_config()

    user_path = input("Путь к файлу 1C (.txt): ").strip().strip('"')
    if not user_path or not os.path.exists(user_path):
        print("❌ Файл не найден.")
        return

    try:
        parser = BankParser(user_path)
        df, balance = parser.parse()
        if df.empty:
            print("❌ Файл не содержит транзакций.")
            return
        print(f"✅ Загружено {len(df)} транзакций.")

        classifier = AIClassifier()
        df_classified = classifier.classify(df)

        pd.set_option('display.max_colwidth', 60)
        print("\n[Пример последних транзакций]:")
        print(df_classified[['Назначение', 'Сумма', 'Category']].tail(5).to_string(index=False))
        print_classification_summary(df_classified)

        tax_regime, custom_rate = select_tax_regime()

        scale_factor = config.get('scale_factor', 5.0)
        fixed_growth = config.get('fixed_exp_growth', 3.5)
        engine_base = ForecastEngine(df_classified, balance, tax_regime, custom_rate,
                                     scale_factor, fixed_growth)
        base_profit = engine_base.run_simulation()
        engine_base.get_verdict(base_profit)

        run_dynamic = input("\nЗапустить исправленный динамический прогноз? (y/n): ").strip().lower()
        if run_dynamic == 'y':
            dyn_config = config.get('dynamic_forecast', {})
            dyn_config['tax_regime'] = tax_regime
            dyn_config['custom_tax_rate'] = custom_rate
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
                except ValueError:
                    print("⚠️ Ошибка ввода, используются значения по умолчанию.")
            dyn_engine = DynamicForecastEngine(df_classified, balance, dyn_config)
            dyn_engine.run()

        run_pinn = input("\nЗапустить PINN-анализ (экспериментальный)? (y/n): ").strip().lower()
        if run_pinn == 'y':
            pinn = TorchPINN(df_classified, balance, tax_rate=0.06)
            pinn.train(epochs=800)
            pinn.forecast_text_table(months_ahead=12)   # текстовый прогноз баланса
            pinn.verdict()
        else:
            print("Анализ завершён.")

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