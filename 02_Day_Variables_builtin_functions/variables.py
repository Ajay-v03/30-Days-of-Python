
# Variables in Python

first_name = 'Ajay'
last_name = 'Verma'
country = 'India'
city = 'Hisar'
age = 250
is_married = False
skills = ['Python','Machine Learning','Data Science']
person_info = {
    'firstname':'Ajay', 
    'lastname':'Verma', 
    'country':'India',
    'city':'Hisar'
    }

# Printing the values stored in the variables

print('First name:', first_name)
print('First name length:', len(first_name))
print('Last name: ', last_name)
print('Last name length: ', len(last_name))
print('Country: ', country)
print('City: ', city)
print('Age: ', age)
print('Married: ', is_married)
print('Skills: ', skills)
print('Person information: ', person_info)

# Declaring multiple variables in one line

first_name, last_name, country, age, is_married = 'Ajay', 'Verma', 'India', 250, False

print(first_name, last_name, country, age, is_married)
print('First name:', first_name)
print('Last name: ', last_name)
print('Country: ', country)
print('Age: ', age)
print('Married: ', is_married)