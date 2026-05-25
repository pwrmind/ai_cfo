# -*- coding: utf-8 -*-
"""
AI CFO v3.0 – Rule‑based финансовый аналитик
Упрощённая, интерпретируемая и надёжная система анализа банковских выписок.

Особенности:
- Классификация транзакций по правилам с весами и регулярными выражениями
- Два режима прогноза: базовый (масштабирование) и динамический (с учётом сезонности)
- Понятные текстовые рекомендации в стиле экспертной системы
"""

import os
import re
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ==========================================
# МОДУЛЬ 1: ПАРСЕР ФОРМАТА 1C (без изменений)
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
        except Exception:
            df['Дата_dt'] = pd.NaT
        return df, self.real_end_balance


# ==========================================
# МОДУЛЬ 2: ПРАВИЛОВЫЙ КЛАССИФИКАТОР
# ==========================================
class RuleBasedClassifier:
    """
    Классификатор транзакций на основе приоритетных правил (regexp + ключевые слова).
    Каждое правило имеет вес (от 0 до 100). Правило с максимальным весом выигрывает.
    """
    def __init__(self):
        # Правила: (регулярное выражение, категория, вес)
        self.rules = [
            # Абсолютный приоритет (вес 100)
            (r"уставн(ый|ого)?\s+капитал|внесение\s+капитала", "FINANCIAL_FLOW", 100),
            (r"card2card|corpcards|p2p", "FINANCIAL_FLOW", 100),
            (r"перевод собственных средств", "FINANCIAL_FLOW", 100),

            (r"алимент|ндфл|страховые взносы|пфр|фсс|исполнительный лист", "FIXED_EXPENSE", 100),
            (r"налог на имущество|транспортный налог|земельный налог", "FIXED_EXPENSE", 100),
            (r"штраф|пеня", "FIXED_EXPENSE", 100),

            (r"обслуживание счета|ведение счета|информирование об операциях|абонентская плата", "FIXED_EXPENSE", 100),
            (r"комиссия за .*?sms", "FIXED_EXPENSE", 100),
            (r"комиссия за обслуживание карты", "FIXED_EXPENSE", 100),

            # Высокий приоритет (вес 90)
            (r"аренд[аыуе]|арендн", "FIXED_EXPENSE", 90),
            (r"зарплат[аыуе]|заработн", "FIXED_EXPENSE", 90),
            (r"бухгалтер", "FIXED_EXPENSE", 90),
            (r"интернет|связь", "FIXED_EXPENSE", 90),

            (r"закупк[аи]|товар|логистик", "OPERATING_EXPENSE", 90),
            (r"гсм|топливо", "OPERATING_EXPENSE", 90),
            (r"хозтовар|хозяйственные материалы|канцеляр", "OPERATING_EXPENSE", 90),

            (r"оборудован[аи]|компьютер|мебел|автомобил", "CAPEX", 90),
            (r"основные средства|станок|техника|капитальные вложения", "CAPEX", 90),

            (r"выручк[аи]|оплат[аы] от клиента|поступление оплаты", "OPERATING_INCOME", 90),
            (r"эквайринг|терминал|розничная выручка", "OPERATING_INCOME", 90),

            # Средний приоритет (вес 70)
            (r"услуги по договору|оказание услуг", "OPERATING_EXPENSE", 70),
            (r"реализация|отгрузка", "OPERATING_INCOME", 70),
            (r"расходные материалы", "OPERATING_EXPENSE", 70),

            # Низкий приоритет (вес 50) – общие леммы
            (r"за услуги", "OPERATING_EXPENSE", 50),
            (r"по счету", "OPERATING_EXPENSE", 50),
            (r"возврат", "FINANCIAL_FLOW", 50),
        ]

        # MCC-коды (прямое отображение)
        self.mcc_map = {
            "5411": "OPERATING_EXPENSE", "5499": "OPERATING_EXPENSE",
            "5812": "OPERATING_EXPENSE", "5814": "OPERATING_EXPENSE",
            "5999": "OPERATING_EXPENSE", "5300": "OPERATING_EXPENSE",
            "4111": "OPERATING_EXPENSE", "3990": "OPERATING_EXPENSE",
            "4814": "FIXED_EXPENSE", "7372": "FIXED_EXPENSE",
            "7394": "OPERATING_EXPENSE", "8299": "OPERATING_EXPENSE",
            "5732": "CAPEX", "6011": "FINANCIAL_FLOW", "6538": "FINANCIAL_FLOW",
        }

    def get_mcc(self, text):
        if not isinstance(text, str):
            return None
        match = re.search(r'MCC[:\s]*(\d{4})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def classify(self, df):
        print("[Система] Классификация транзакций по правилам с весами...")
        categories = []
        for idx, row in df.iterrows():
            txt = row['Назначение']
            amount = row['Сумма']

            # 1. Проверка MCC
            mcc = self.get_mcc(txt)
            if mcc and mcc in self.mcc_map:
                cat = self.mcc_map[mcc]
                # Коррекция знака: если MCC даёт OPERATING_INCOME, но сумма отрицательная → расход
                if cat == "OPERATING_INCOME" and amount < 0:
                    cat = "OPERATING_EXPENSE"
                categories.append(cat)
                continue

            # 2. Правила
            best_weight = -1
            best_cat = "OTHER"
            for pattern, cat, weight in self.rules:
                if re.search(pattern, txt, re.IGNORECASE):
                    if weight > best_weight:
                        best_weight = weight
                        best_cat = cat

            # 3. Fallback по знаку суммы (если не удалось определить)
            if best_cat == "OTHER":
                if amount > 0:
                    best_cat = "OPERATING_INCOME"   # по умолчанию доход
                else:
                    best_cat = "OPERATING_EXPENSE"  # по умолчанию расход

            # Дополнительная коррекция: если получили OPERATING_INCOME, но сумма отрицательная
            if best_cat == "OPERATING_INCOME" and amount < 0:
                best_cat = "OPERATING_EXPENSE"

            categories.append(best_cat)

        df['Category'] = categories
        return df


# ==========================================
# МОДУЛЬ 3: БАЗОВАЯ СИМУЛЯЦИЯ (масштабирование)
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
        print("\n" + "="*60)
        print(" БАЗОВЫЙ АНАЛИЗ И СИМУЛЯЦИЯ МАСШТАБИРОВАНИЯ ")
        print("="*60)
        sums = self.df.groupby('Category')['Сумма'].sum()
        op_income = sums.get('OPERATING_INCOME', 0.0)
        op_expense = abs(sums.get('OPERATING_EXPENSE', 0.0))
        fixed_expense = abs(sums.get('FIXED_EXPENSE', 0.0))
        capex = abs(sums.get('CAPEX', 0.0))
        fin_flow = sums.get('FINANCIAL_FLOW', 0.0)

        print(f"📊 ФАКТИЧЕСКИЕ ДАННЫЕ (за анализируемый период):")
        print(f"   Выручка (операционные доходы):        {op_income:>12,.2f} руб.")
        print(f"   Переменные расходы:                   {op_expense:>12,.2f} руб.")
        print(f"   Постоянные расходы:                   {fixed_expense:>12,.2f} руб.")
        print(f"   Капитальные вложения (CAPEX):          {capex:>12,.2f} руб.")
        print(f"   Финансовые потоки (нетто):             {fin_flow:>12,.2f} руб.")
        if self.end_balance is not None:
            print(f"   💰 Конечный остаток на счёте:        {self.end_balance:>12,.2f} руб.")
        print(f"   -----------------------------------")
        unit_profit = op_income - op_expense
        if unit_profit > 0:
            print(f"⚖️  ЮНИТ-ЭКОНОМИКА: ✅ Положительная (каждая продажа приносит прибыль)")
        else:
            print(f"⚖️  ЮНИТ-ЭКОНОМИКА: ❌ ОТРИЦАТЕЛЬНАЯ – переменные расходы превышают выручку")
        actual_tax = self._calculate_tax(op_income, op_expense, fixed_expense, capex/12)
        print(f"   💼 Расчётный налог (по факту):        {actual_tax:,.2f} руб. (режим: {self.tax_regime})")

        print(f"\n🚀 ПРОГНОЗ ПРИ МАСШТАБИРОВАНИИ В {self.scale_factor} РАЗ:")
        sim_income = op_income * self.scale_factor
        sim_op_expense = op_expense * self.scale_factor
        sim_fixed_expense = fixed_expense * self.fixed_exp_growth if fixed_expense > 0 else 0.0
        sim_amortization = (capex * self.scale_factor) / 12
        sim_tax = self._calculate_tax(sim_income, sim_op_expense, sim_fixed_expense, sim_amortization)
        sim_net_profit = sim_income - sim_op_expense - sim_fixed_expense - sim_tax - sim_amortization

        print(f"   Прогнозируемая выручка:            {sim_income:>12,.2f}")
        print(f"   Переменные расходы:                {sim_op_expense:>12,.2f}")
        print(f"   Постоянные расходы:                {sim_fixed_expense:>12,.2f}")
        print(f"   Налоги:                            {sim_tax:>12,.2f}")
        print(f"   Амортизация (CAPEX/12):            {sim_amortization:>12,.2f}")
        print(f"   -----------------------------------")
        print(f"   📉 ПРОГНОЗ ЧИСТОЙ ПРИБЫЛИ:          {sim_net_profit:>12,.2f} руб.")
        return sim_net_profit

    def get_verdict(self, net_profit):
        print("\n" + "="*60)
        print(" ЭКСПЕРТНОЕ ЗАКЛЮЧЕНИЕ ")
        print("="*60)
        if net_profit < 0:
            print("🛑 ВЫСОКИЙ РИСК МАСШТАБИРОВАНИЯ")
            print(f"   Прогнозируемый убыток: {net_profit:,.2f} руб.")
            print("   Рекомендация:")
            print("   - Снизить переменные издержки (искать новых поставщиков).")
            print("   - Увеличить наценку на товар/услугу.")
            print("   - Пересмотреть необходимость капитальных вложений.")
        else:
            print("✅ ЗЕЛЁНЫЙ СВЕТ. Модель масштабируема.")
            print(f"   Прогнозируемая прибыль: {net_profit:,.2f} руб.")
            print("   Рекомендуемый шаг:")
            print("   - Постепенное увеличение оборота с контролем маржинальности.")


# ==========================================
# МОДУЛЬ 4: УПРОЩЁННЫЙ ДИНАМИЧЕСКИЙ ПРОГНОЗ
# ==========================================
class DynamicForecastEngine:
    def __init__(self, df_classified, end_balance, config):
        self.df = df_classified
        self.end_balance = end_balance if end_balance is not None else 0.0
        self.config = config
        self.months = config.get('forecast_months', 12)
        self.growth_rate_monthly = config.get('growth_rate_monthly', 0.02)  # 2% рост в месяц
        self.inflation_monthly = config.get('inflation_rate_monthly', 0.005)
        self.receivables_days = config.get('receivables_days', 0)
        self.payables_days = config.get('payables_days', 0)
        self.tax_schedule = config.get('tax_payment_schedule', 'quarterly')
        self.tax_regime = config.get('tax_regime', 'income')
        self.custom_tax_rate = config.get('custom_tax_rate', None)
        self.capex_monthly = config.get('capex_monthly', 0.0)

    def _calculate_tax(self, income, expense, fixed, amort):
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

    def _estimate_seasonal_factors(self):
        """Упрощённая оценка сезонности: если нет данных, возвращает [1]*12"""
        return [1.0] * 12   # без сезонности

    def run(self):
        # Группировка по месяцам для расчёта средних
        if self.df['Дата_dt'].isna().all():
            print("❌ Нет корректных дат для динамического прогноза. Используйте базовый режим.")
            return None

        monthly = self.df.groupby(self.df['Дата_dt'].dt.to_period('M'))['Сумма'].sum().to_frame()
        if monthly.empty:
            print("⚠️ Недостаточно данных для построения динамического прогноза.")
            return None

        # Оценка среднемесячных показателей (только операционные, без финансовых потоков)
        op_df = self.df[self.df['Category'] != 'FINANCIAL_FLOW']
        if op_df.empty:
            print("⚠️ Нет операционных транзакций.")
            return None

        # Разделяем доходы и расходы
        income_df = op_df[op_df['Сумма'] > 0]
        expense_df = op_df[op_df['Сумма'] < 0]

        avg_monthly_income = income_df.groupby(income_df['Дата_dt'].dt.to_period('M'))['Сумма'].sum().mean()
        avg_monthly_op_exp = abs(expense_df.groupby(expense_df['Дата_dt'].dt.to_period('M'))['Сумма'].sum().mean())
        fixed_df = self.df[self.df['Category'] == 'FIXED_EXPENSE']
        avg_monthly_fixed = abs(fixed_df.groupby(fixed_df['Дата_dt'].dt.to_period('M'))['Сумма'].sum().mean()) if not fixed_df.empty else 0.0
        avg_capex = self.capex_monthly

        if avg_monthly_income <= 0:
            print("❌ Среднемесячная выручка не определена. Прогноз невозможен.")
            return None

        # Прогнозирование
        print("\n" + "="*70)
        print(" ДИНАМИЧЕСКИЙ ПРОГНОЗ ДЕНЕЖНЫХ ПОТОКОВ (упрощённая модель)")
        print("="*70)
        print("Примечание: расчёт основан на среднемесячных значениях + линейный рост.")
        print(f"   Заданный темп роста выручки: {self.growth_rate_monthly*100:.1f}% в месяц")
        print(f"   Инфляция расходов: {self.inflation_monthly*100:.1f}% в месяц\n")

        balance = self.end_balance
        # Переменные для накопления квартальных/годовых налогов
        q_income = 0.0; q_expense = 0.0; q_fixed = 0.0; q_amort = 0.0; quarter_cnt = 0
        tax_paid = 0.0

        rows = []
        for i in range(self.months):
            month_idx = i + 1
            growth_factor = (1 + self.growth_rate_monthly) ** i
            infl_factor = (1 + self.inflation_monthly) ** i

            revenue = avg_monthly_income * growth_factor
            op_expense = avg_monthly_op_exp * infl_factor * growth_factor   # расходы тоже растут с оборотами
            fixed_expense = avg_monthly_fixed * infl_factor
            capex = avg_capex * infl_factor
            amort = capex / 12

            # Учёт лагов (упрощённо: если отсрочка > 0, то поступление/платёж сдвигается на 1 месяц)
            cash_income = revenue if self.receivables_days <= 30 else revenue * 0.5
            cash_op_expense = op_expense if self.payables_days <= 30 else op_expense * 0.5

            # Налоговый учёт
            q_income += revenue
            q_expense += op_expense
            q_fixed += fixed_expense
            q_amort += amort
            quarter_cnt += 1

            tax_payment = 0.0
            if self.tax_schedule == 'monthly':
                tax_payment = self._calculate_tax(revenue, op_expense, fixed_expense, amort)
            elif self.tax_schedule == 'quarterly' and quarter_cnt == 3:
                tax_payment = self._calculate_tax(q_income, q_expense, q_fixed, q_amort)
                q_income = q_expense = q_fixed = q_amort = 0.0
                quarter_cnt = 0
            elif self.tax_schedule == 'annual' and i == self.months - 1:
                total_income = sum(r['Выручка'] for r in rows) + revenue
                total_expense = sum(r['Перем.расходы'] for r in rows) + op_expense
                total_fixed = sum(r['Пост.расходы'] for r in rows) + fixed_expense
                total_amort = sum(r.get('Capex', 0)/12 for r in rows) + amort
                tax_payment = self._calculate_tax(total_income, total_expense, total_fixed, total_amort)

            net_cash = cash_income - cash_op_expense - fixed_expense - capex - tax_payment
            balance += net_cash
            tax_paid += tax_payment

            rows.append({
                'Месяц': i+1,
                'Выручка': revenue,
                'Поступления (cash)': cash_income,
                'Перем.расходы': op_expense,
                'Платежи перем.': cash_op_expense,
                'Пост.расходы': fixed_expense,
                'Capex': capex,
                'Налоги': tax_payment,
                'Чистый поток': net_cash,
                'Баланс на конец': balance
            })

        forecast_df = pd.DataFrame(rows)
        pd.set_option('display.max_columns', 10)
        print(forecast_df.round(2).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        print(f"\n📌 ИТОГО ЗА {self.months} МЕСЯЦЕВ:")
        print(f"   Суммарная выручка:          {forecast_df['Выручка'].sum():,.2f} руб.")
        print(f"   Суммарные расходы (перем):   {forecast_df['Перем.расходы'].sum():,.2f} руб.")
        print(f"   Постоянные расходы:          {forecast_df['Пост.расходы'].sum():,.2f} руб.")
        print(f"   Налоги уплачено:             {tax_paid:,.2f} руб.")
        print(f"   Конечный остаток на счёте:   {balance:,.2f} руб.")

        if balance < 0:
            print("\n⚠️ ВНИМАНИЕ: по прогнозу возникает кассовый разрыв (отрицательный остаток).")
            print("   Рекомендация: увеличить отсрочку по платежам или привлечь дополнительное финансирование.")
        else:
            print("\n✅ Прогнозный баланс положителен – кассовых разрывов не ожидается.")
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
    print("\nВыберите налоговый режим (для расчёта прогнозной нагрузки):")
    print("1 - УСН Доходы (6%)")
    print("2 - УСН Доходы минус расходы (15%)")
    print("3 - Другая ставка (от выручки)")
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

def explain_categories():
    """Выводит человеко-читаемое описание категорий"""
    print("\n📖 РАСШИФРОВКА КАТЕГОРИЙ ТРАНЗАКЦИЙ:")
    print("   OPERATING_INCOME  – Операционные доходы (выручка, оплаты клиентов, эквайринг)")
    print("   OPERATING_EXPENSE – Переменные расходы (закупка товара, логистика, материалы)")
    print("   FIXED_EXPENSE     – Постоянные расходы (аренда, зарплата, налоги, банковское обслуживание)")
    print("   CAPEX             – Капитальные вложения (оборудование, основные средства)")
    print("   FINANCIAL_FLOW    – Финансовые потоки (переводы между счетами, взносы, займы)")


# ==========================================
# ОСНОВНОЙ МОДУЛЬ (ТОЧКА ВХОДА)
# ==========================================
def main():
    print("🤖 AI CFO v3.0 – Rule‑based финансовый аналитик")
    print("   Упрощённая, надёжная и интерпретируемая система.")
    config = load_config()

    user_path = input("\nПуть к файлу выписки 1C (.txt): ").strip().strip('"')
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

        # Классификация
        classifier = RuleBasedClassifier()
        df_classified = classifier.classify(df)

        pd.set_option('display.max_colwidth', 60)
        print("\n📝 ПРИМЕР ПОСЛЕДНИХ ТРАНЗАКЦИЙ С КАТЕГОРИЯМИ:")
        print(df_classified[['Назначение', 'Сумма', 'Category']].tail(5).to_string(index=False))
        print_classification_summary(df_classified)
        explain_categories()

        # Выбор налогового режима
        tax_regime, custom_rate = select_tax_regime()

        # Параметры масштабирования (можно взять из конфига или спросить)
        scale_factor = config.get('scale_factor', 5.0)
        fixed_growth = config.get('fixed_exp_growth', 3.5)

        # Базовый прогноз
        engine_base = ForecastEngine(df_classified, balance, tax_regime, custom_rate,
                                     scale_factor, fixed_growth)
        base_profit = engine_base.run_simulation()
        engine_base.get_verdict(base_profit)

        # Динамический прогноз
        run_dynamic = input("\nЗапустить детальный динамический прогноз (месячный)? (y/n): ").strip().lower()
        if run_dynamic == 'y':
            dyn_config = config.get('dynamic_forecast', {})
            # Переопределение налогового режима из выбора пользователя
            dyn_config['tax_regime'] = tax_regime
            dyn_config['custom_tax_rate'] = custom_rate

            print("\nТекущие параметры динамического прогноза (можно изменить в config.yaml):")
            for k, v in dyn_config.items():
                print(f"  {k}: {v}")
            change = input("Хотите изменить параметры? (y/n): ").strip().lower()
            if change == 'y':
                try:
                    dyn_config['forecast_months'] = int(input("Количество месяцев прогноза: ") or dyn_config.get('forecast_months', 12))
                    dyn_config['growth_rate_monthly'] = float(input("Рост выручки в месяц (например, 0.02 для 2%): ") or dyn_config.get('growth_rate_monthly', 0.02))
                    dyn_config['receivables_days'] = float(input("Отсрочка от клиентов (дней): ") or dyn_config.get('receivables_days', 0))
                    dyn_config['payables_days'] = float(input("Отсрочка поставщикам (дней): ") or dyn_config.get('payables_days', 0))
                except ValueError:
                    print("⚠️ Ошибка ввода, используются значения по умолчанию.")

            dyn_engine = DynamicForecastEngine(df_classified, balance, dyn_config)
            dyn_engine.run()

        print("\n✅ Анализ завершён. Все выводы основаны на заданных правилах и не содержат «чёрных ящиков».")

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