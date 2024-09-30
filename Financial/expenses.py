### monthly expenses
rent = 1050
print(f"Monthly rent: {rent}")
car_payment = 635  # 10 more months
personal_loan = 500  # annual
gas = 50 * 3
insurance = 250
print(f"Monthly car expenses: {car_payment + gas + insurance}")

### monthly subscriptions:
sourcery = 10
one_drive = 10  # unused
twitter = 11
youtube = 12
github = 4
barbershop = 35 / 2
apple_music = 10
total_subscriptions = sourcery + twitter + youtube + github + barbershop + apple_music
print(f"Monthly subscriptions: {total_subscriptions}")

### monthly food and groceries
breakfast = 30 * 10
lunch = 30 * 15
groceries = 50 * 4
print(f"Monthly food and groceries: {breakfast + lunch + groceries}")

## total expenses
total_annual = 9 * car_payment + 6 * personal_loan + 12 * (rent + gas + insurance + total_subscriptions + breakfast + lunch + groceries)
print(f"Total annual expenses: {total_annual}")

## annual income
annual_income = 0
### tax rate
retirement = 0.05
annual_income *= 1 - retirement
tax_rate = 0.20
annual_income *= 1 - tax_rate
print(f"Annual income after tax: {annual_income}")
# savings
savings = annual_income - total_annual
print(f"Annual Savings: {savings}")
print(f"Monthly Savings: {savings / 12:.2f}")
