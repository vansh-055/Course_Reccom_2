# @title Default title text
import pandas as pd
import numpy as np
import json
import csv
import networkx as nx
G = nx.DiGraph()



cdcs = ["B.E Chemical","B.E Civil","B.E Computer Science","B.E Electrical & Electronic","B.E Electronics & Communication","B.E Electronics & Instrumentation","B.E Mechanical","B.Pharm","M.Sc. Biological Sciences","M.Sc. Chemistry","M.Sc. Economics","M.Sc. Mathematics","M.Sc. Physics","None"]


# Path to your JSON file
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # folder where this .py file is

json_file_path = os.path.join(script_dir, 'data.json')

predicted_gpa = []

# Read data from JSON file
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

data = json_data


be = data['beDegree']
msc = data['mscDegree']


selected_course = data['preferredElective']

data['beDegree'] = []
data['mscDegree'] = []
data['preferredElective'] = []

be_index = cdcs.index(be)
if(msc!='None'):
    msc_index = cdcs.index(msc)
else:
    msc_index = None

csv_file_path = os.path.join(script_dir, 'g.csv')
# Extracting values from the dictionary and flattening the list
courses = [course for sublist in data.values() for course in sublist]


# Writing courses to a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = courses[0].keys() if courses else []
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    csv_writer.writeheader()  # Write CSV header

    # Write each course as a row in the CSV
    for course in courses:
        csv_writer.writerow(course)

# Step 1: Load test data from g.csv and filter columns
test_data = pd.read_csv(r'g.csv')  
test_data = test_data[['subject', 'courseGrade']]  # Keep only 'subject' and 'courseGrade' columns

# Step 2: Load course data from CDCs.csv
csv_path = os.path.join(script_dir, "CDCs.csv")
cdcs_file = pd.read_csv(csv_path)

# Step 3: Convert columns to lists (transpose rows to columns)
columns_as_lists = cdcs_file.values.T.tolist()


# Load the data from the Excel sheet
grade_csv_path = os.path.join(script_dir, "Grade.csv")

# Read the CSV file
data = pd.read_csv(grade_csv_path)


# Create a dictionary to map student IDs to their course IDs and grades
student_courses = {}
for index, row in data.iterrows():
    student_id = row['student_id']
    course_id = row['course_id']
    course_grade = row['course_grade']

    if student_id not in student_courses:
        student_courses[student_id] = []

    student_courses[student_id].append((course_id, course_grade))

student_courses[1112] = []
for index, row in test_data.iterrows():
    student_courses[1112].append((row['subject'], row['courseGrade']))


cgpa_csv_path = os.path.join(script_dir, "CGPA.csv")


def predict_gpa(similarities,this_student,i=25):
    cgpa_data = pd.read_csv(cgpa_csv_path).to_records(index=False)
    gpas = []
    avg_gpa = sum([x[1] for x in this_student])/len(this_student)
    for students_id in similarities[:i]:
        gpas.append(cgpa_data[int(students_id[0])][1])
    return ((sum(gpas)/len(gpas)))
    


# Define a function to calculate the similarity between two sets of courses
def calculate_similarity(courses1, courses2):
    # Convert courses to sets for efficient comparison
    courses1_set = set(courses1)
    courses2_set = set(courses2)

    # Calculate the Jaccard similarity
    jaccard_similarity = len(courses1_set & courses2_set) / len(courses1_set | courses2_set)

    # Calculate the weighted Jaccard similarity based on course grades
    weighted_jaccard_similarity = 0
    for course_id1, grade1 in courses1:
        for course_id2, grade2 in courses2:
            if course_id1 == course_id2 and grade1 == grade2:
                weighted_jaccard_similarity += grade1*grade2

    return jaccard_similarity + weighted_jaccard_similarity

student = 1112

# Split the data into training and test sets
train_students = {}
test_students = {}

for student_id, courses in student_courses.items():
    if student_id!=student:
        train_students[student_id] = courses
test_students[student] = student_courses[student]

# Define a function to recommend courses to a student
def recommend_courses(student_id, train_students, student_courses):

    # Calculate the similarity between the student's courses and all other students' courses
    similarities = []
    for other_student_id, other_student_courses in train_students.items():
          similarity = calculate_similarity(student_courses, other_student_courses)
          # similarity = similarity_matrix[student_id,other_student_id]
          similarities.append((other_student_id, similarity))

    # Sort the similarities in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    def calculate_gpa(student_course_list):
        gp = 0
        cnt = 0
        for course_name, course_grade in student_course_list:
            gp+=course_grade
            cnt+=1
        return gp/cnt

    gpas = []
    for student_id, sim in similarities:
        gpas.append((student_id,calculate_gpa(train_students[student_id]),train_students[student_id]))
        
    gpas.sort(key=lambda x:x[1],reverse=True)

    course_list = []
    for i in gpas[:25]:
        course_list.append(i[2])

    courses = []
    for i in course_list:
        for j in i:
            courses.append(j[0])
    course_count = [(x,courses.count(x)) for x in set(courses)]
    course_count.sort(key=lambda x:x[1],reverse=True)

    print(student_courses)
    
    predicted_gpa = predict_gpa(similarities,student_courses)

    csv_file_gpas = r'output_2.csv'

    with open(csv_file_gpas, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([predicted_gpa])
    

    # Recommend the top 10 courses

    final = []
    #index = cdcs.index(be)
    for course in course_count:
        if course[0] not in columns_as_lists[be_index] and course[0]+"\n" not in columns_as_lists[be_index]:
            final.append(course)

    if(msc != "None"):
        #l = sorted([(x,all_courses.count(x)) for x in set(all_courses)],key=lambda x:x[1],reverse=True)

        f = []
    
        for course in final:
            if course[0] not in columns_as_lists[msc_index] and course[0]+"\n" not in columns_as_lists[msc_index]:
                f.append(course)
        final = f

    return final

# Evaluate the recommender system
recommended_courses = recommend_courses(student, train_students, test_students[student])

print(recommended_courses[:10])


timetable_path = os.path.join(script_dir, "timetable.json")

# Load and preview the JSON data
with open(timetable_path, 'r') as file:
    json_data = json.load(file)


course_names = [course_data['course_name'] for course_data in json_data['courses'].values()]

available_courses = course_names



csv_file_path = r'output.csv'

# Extracting values from the dictionary and flattening the list
# courses = [course for sublist in available_courses.values() for course in sublist]
count_elements = 0
# Writing courses to a CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Course']  # Add more fields if needed
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    csv_writer.writeheader()  # Write CSV header

    for element, count in recommended_courses:
        if count_elements == 5:
          break
        if element:
            csv_writer.writerow({'Course': element})
            count_elements+=1

print("\n\nUse-Case-3\n\n")#################################################################################################

# Create full path to the CSV
csv_path = os.path.join(script_dir, "Grade_Student.csv")

# Load the CSV with proper encoding
student_course = pd.read_csv(csv_path, encoding='latin-1')

directed_graph = {}

for index, row in student_course.iterrows():
    row_tuple = tuple(row)
    if row_tuple[0] in directed_graph:
        directed_graph[row_tuple[0]].append((row_tuple[3],row_tuple[6]))
    else:
        directed_graph[row_tuple[0]] = [(row_tuple[3],row_tuple[6])]


# Add nodes (students) to the graph
for student in directed_graph.keys():
    G.add_node(student)

# Add edges (weighted courses) to the graph
for student, courses in directed_graph.items():
    for course, weight in courses:
        G.add_edge(course, student, weight=weight)

# Calculate PageRank
pagerank_scores = nx.pagerank(G)


# Parse student input to get the courses they have already taken
student_input = []

for index,courses in test_data.iterrows():
    student_input.append(courses['subject'])


# Filter courses that the student has not taken yet
available_courses = [course for course in pagerank_scores.keys() if course not in student_input]

# Sort available courses based on their PageRank scores
recommended_courses = sorted(available_courses, key=lambda x: pagerank_scores[x], reverse=True)

# Print recommended courses
print("Recommended Courses:")

students = recommended_courses[:40]

courses_ = []

for student in students:
    if student in directed_graph:
        courses = directed_graph[student]
        sorted_courses = sorted(courses, key=lambda x: x[1], reverse=True)
        for course, weight in sorted_courses[:3]:
            courses_.append((weight,course))


courses_ = list(set(courses_))


courses_.sort(reverse=True)



final = []
for course in courses_:
    if course[1] not in columns_as_lists[be_index] and course[1]+"\n" not in columns_as_lists[be_index]:
        final.append(course[1])

if(msc != "None"):

    #l = sorted([(x,all_courses.count(x)) for x in set(all_courses)],key=lambda x:x[1],reverse=True)

    f = []

    for course in final:
        if course not in columns_as_lists[msc_index] and course+"\n" not in columns_as_lists[msc_index]:
            f.append(course)
    final = f

print(final)


csv_file_path = r'output_3.csv'


count_elements = 0
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Course']
    csv_writer = csv.DictWriter(csvfile, fieldnames=['Course'])

    csv_writer.writeheader()

    for element in final:
        if count_elements == 5:
          break
        if element:
            csv_writer.writerow({'Course': element})
            count_elements+=1



print("\n\nUse-Case-4\n\n")##################################################

cdcs_path = os.path.join(script_dir, "CDCs.csv")
grade_path = os.path.join(script_dir, "GradeDataWithBranch.csv")

cdcs = pd.read_csv(cdcs_path)
students = pd.read_csv(grade_path)
columns_as_lists = cdcs.values.T.tolist() 

studs = {}


for index,row in students.iterrows():
    if(msc == "None"):
        if(row['branch1'] == be and pd.isna(row['branch2'])):
            if(row['student_id'] in studs):
                studs[row['student_id']].append(row['course_id'])
            else:
                studs[row['student_id']] = [row['course_id']]
                
    else:
        if(row['branch1'] == be and row['branch2'] == msc):
            
            if(row['student_id'] in studs):
                studs[row['student_id']].append(row['course_id'])
            else:
                studs[row['student_id']] = [row['course_id']]



all_courses = []

for student in studs:
    if selected_course in studs[student]:
        all_courses.extend(studs[student])


l = sorted([(x,all_courses.count(x)) for x in set(all_courses)],key=lambda x:x[1],reverse=True)

final = []



for course in l:
    if course[0] not in columns_as_lists[be_index] and course[0]+"\n" not in columns_as_lists[be_index]:
        final.append(course)


if(msc != "None"):

    f = []
    
    for course in final:
        if course[0] not in columns_as_lists[msc_index] and course[0]+"\n" not in columns_as_lists[msc_index]:
            f.append(course)
    final = f

print(final)

csv_file_path = r'output_4.csv'
count_elements = 0
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Course']
    csv_writer = csv.DictWriter(csvfile, fieldnames=['Course'])

    csv_writer.writeheader()

    for element, number in final:
        if count_elements == 5:
          break
        if element:
            csv_writer.writerow({'Course': element})
            count_elements+=1



print('Output Updated')
