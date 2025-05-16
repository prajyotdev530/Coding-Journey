import json
employees_string='''{
  "company": "TechNova Solutions",
  "location": "Bangalore",
  "employees": [
    {
      "id": 101,
      "name": "Prajyot More",
      "age": 22,
      "position": "Software Developer",
      "skills": ["Python", "C++", "React"],
      "full_time": true,
      "address": {
        "street": "12 IIT Campus Road",
        "city": "Kharagpur",
        "state": "West Bengal",
        "zip": "721302"
      },
      "projects": [
        {
          "name": "AI Chatbot",
          "status": "Completed",
          "tech_stack": ["Python", "Flask", "TensorFlow"]
        },
        {
          "name": "Employee Dashboard",
          "status": "Ongoing",
          "tech_stack": ["React", "Node.js", "MongoDB"]
        }
      ]
    },
    {
      "id": 102,
      "name": "Sneha Sharma",
      "age": 26,
      "position": "UI/UX Designer",
      "skills": ["Figma", "Photoshop", "HTML", "CSS"],
      "full_time": false,
      "address": {
        "street": "44 MG Road",
        "city": "Pune",
        "state": "Maharashtra",
        "zip": "411001"
      },
      "projects": [
        {
          "name": "E-Commerce Redesign",
          "status": "Completed",
          "tech_stack": ["Figma", "Bootstrap"]
        }
      ]
    }
  ]
}'''

json_data=json.loads(employees_string)
print(json_data)
print(type(json_data['employees']))
for employee in json_data['employees']:
    print(employee['name'])
    del employee['age']

newjsondata=json.dumps(json_data,indent=2,sort_keys=True)
print(format(newjsondata))
print(type(newjsondata))

