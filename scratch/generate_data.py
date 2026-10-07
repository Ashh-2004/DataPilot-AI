import random
from datetime import datetime, timedelta
import os

os.makedirs('d:/Projects/DataPilot/data', exist_ok=True)

# Generate sales_data.csv
random.seed(42)

customers = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown", "Charlie Davis", "Eve White", 
             "Frank Miller", "Grace Wilson", "Harry Taylor", "Ivy Moore", "Jack Anderson", "Karen Thomas",
             "Leo Jackson", "Mia Martin", "Noah Lee", "Olivia Perez", "Paul Thompson", "Quinn Garcia",
             "Rachel Martinez", "Sam Robinson", "Tina Clark", "Uma Rodriguez", "Victor Lewis", 
             "Wendy Walker", "Xavier Hall", "Yara Allen", "Zack Young", "Amy King", "Brian Wright", "Cindy Scott"]

categories = {
    "Electronics": [("Smartphone", 599.99), ("Laptop", 999.99), ("Headphones", 49.99), ("Tablet", 299.99)],
    "Clothing": [("T-Shirt", 15.99), ("Jeans", 39.99), ("Jacket", 89.99), ("Sneakers", 59.99)],
    "Home & Garden": [("Planter", 12.99), ("Lawn Mower", 199.99), ("Garden Hose", 25.99), ("Patio Chair", 45.99)],
    "Sports": [("Tennis Racket", 89.99), ("Basketball", 24.99), ("Yoga Mat", 19.99), ("Dumbbells", 34.99)],
    "Books": [("Fiction Novel", 14.99), ("Cookbook", 24.99), ("Biography", 19.99), ("Sci-Fi Book", 12.99)]
}

regions = ["North", "South", "East", "West"]
payments = ["Credit Card", "Debit Card", "Cash", "UPI"]
statuses = ["Completed", "Completed", "Completed", "Completed", "Pending", "Cancelled", "Returned"]

sales_rows = []
sales_rows.append("order_id,order_date,customer_name,product_category,product_name,quantity,unit_price,total_amount,region,payment_method,status,notes")

start_date = datetime(2025, 1, 1)

data_rows = []
for i in range(95):
    order_id = 1001 + i
    order_date = (start_date + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d")
    customer = random.choice(customers)
    category = random.choice(list(categories.keys()))
    product, price = random.choice(categories[category])
    quantity = random.randint(1, 20)
    total = round(quantity * price, 2)
    region = random.choice(regions)
    payment = random.choice(payments)
    status = random.choice(statuses)
    
    if random.random() < 0.05: customer = " " + customer + "  "
    if random.random() < 0.05: category = category + " "
    if random.random() < 0.05: region = ""
    if random.random() < 0.05: payment = ""
    
    notes = ""
    if random.random() < 0.05: notes = "Follow up needed"
    
    data_rows.append(f"{order_id},{order_date},{customer},{category},{product},{quantity},{price},{total},{region},{payment},{status},{notes}")

data_rows.insert(20, ",,,,,,,,,,,")
data_rows.insert(50, ",,,,,,,,,,,")
data_rows.insert(80, ",,,,,,,,,,,")

data_rows.append(data_rows[10])
data_rows.append(data_rows[40])
data_rows.append(data_rows[70])
data_rows.append(data_rows[10])

for row in data_rows:
    sales_rows.append(row)

with open('d:/Projects/DataPilot/data/sales_data.csv', 'w', newline='', encoding='utf-8') as f:
    f.write("\n".join(sales_rows))

# Generate employee_data.csv
depts = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
desigs = ["Junior", "Senior", "Lead", "Manager", "Director"]
cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]

def get_salary(desig):
    base = {"Junior": 40000, "Senior": 70000, "Lead": 100000, "Manager": 120000, "Director": 150000}
    return base[desig] + random.randint(-5000, 10000)

emp_rows = []
emp_rows.append("emp_id, name,department ,designation,join_date, salary ,performance_rating,city,email,is_active,internal_notes")

data_rows = []
start_date = datetime(2018, 1, 15)

for i in range(1, 75):
    emp_id = f"E{i:03d}"
    name = random.choice(customers)
    dept = random.choice(depts)
    desig = random.choices(desigs, weights=[40, 30, 15, 10, 5])[0]
    join_date = (start_date + timedelta(days=random.randint(0, 2500))).strftime("%Y-%m-%d")
    salary = get_salary(desig)
    rating = round(random.uniform(2.5, 5.0), 1)
    city = random.choice(cities)
    email = name.lower().replace(" ", ".") + "@company.com"
    is_active = "TRUE" if random.random() < 0.9 else "FALSE"
    
    if random.random() < 0.1: rating = ""
    if random.random() < 0.1: city = ""
    
    data_rows.append(f"{emp_id},{name},{dept},{desig},{join_date},{salary},{rating},{city},{email},{is_active},")

data_rows.insert(15, ",,,,,,,,,,")
data_rows.insert(45, ",,,,,,,,,,")

data_rows.append(data_rows[5])
data_rows.append(data_rows[25])

for row in data_rows:
    emp_rows.append(row)

with open('d:/Projects/DataPilot/data/employee_data.csv', 'w', newline='', encoding='utf-8') as f:
    f.write("\n".join(emp_rows))
