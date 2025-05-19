import pandas as pd
import numpy as np
import json
import csv
import networkx as nx
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_json(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def read_csv(file_path):
    return pd.read_csv(file_path)

def write_csv(data, file_path):
    if not data:
        raise ValueError("No data to write.")
    
    # Support list of dicts or list of tuples
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        if isinstance(data[0], dict):
            csv_writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(data)
        elif isinstance(data[0], (tuple, list)) and len(data[0]) == 2:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['course', 'count'])
            csv_writer.writerows(data)
        else:
            # fallback to just writing rows as strings
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows([[str(row)] for row in data])

def get_courses_data(json_file_path, csv_file_path):
    json_data = read_json(json_file_path)
    
    courses = []
    for value in json_data.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    courses.append(item)
    write_csv(courses, csv_file_path)

def load_student_courses(data_file_path):
    data = read_csv(data_file_path)
    student_courses = {}
    for _, row in data.iterrows():
        student_id = row['student_id']
        course_id = row['course_id']
        course_grade = row['course_grade']
        if student_id not in student_courses:
            student_courses[student_id] = []
        student_courses[student_id].append((course_id, course_grade))
    return student_courses

def predict_gpa(similarities, i=25):
    cgpa_data = read_csv(os.path.join(BASE_DIR, 'CGPA.csv')).to_records(index=False)
    # Get GPAs of top i similar students
    gpas = []
    for student_id, _ in similarities[:i]:
        try:
            gpas.append(cgpa_data[int(student_id)][1])  # Assumes student_id corresponds to cgpa_data row index
        except (IndexError, ValueError):
            continue
    if not gpas:
        return 0
    return sum(gpas) / len(gpas)

def calculate_similarity(courses1, courses2):
    # courses1 and courses2 are lists of tuples (course_id, grade)
    courses1_set = set([c[0] for c in courses1])
    courses2_set = set([c[0] for c in courses2])
    
    jaccard_similarity = len(courses1_set & courses2_set) / len(courses1_set | courses2_set) if (courses1_set | courses2_set) else 0
    
    weighted_jaccard_similarity = 0
    # Add weighted similarity based on grades
    for course_id1, grade1 in courses1:
        for course_id2, grade2 in courses2:
            if course_id1 == course_id2:
                weighted_jaccard_similarity += grade1 * grade2
    return jaccard_similarity + weighted_jaccard_similarity

def filter_courses(course_list, cdcs, be, columns_as_lists, msc=""):
    # course_list is list of (course, count)
    final_courses = []
    be_index = cdcs.index(be) if be in cdcs else None
    msc_index = cdcs.index(msc) if msc and msc in cdcs else None

    for course, count in course_list:
        # Use MSC columns if given, else BE
        index = msc_index if msc_index is not None else be_index
        if index is None:
            # if no valid index found, include all courses
            final_courses.append((course, count))
            continue
        # Check if course is NOT in the respective list of courses from CDCs
        course_list_for_branch = columns_as_lists[index]
        if course not in course_list_for_branch and course + "\n" not in course_list_for_branch:
            final_courses.append((course, count))
    return final_courses

def recommend_courses(student_id, train_students, student_courses, cdcs, columns_as_lists, be, msc=""):
    similarities = []
    for other_student_id, other_student_courses in train_students.items():
        similarity = calculate_similarity(student_courses[student_id], other_student_courses)
        similarities.append((other_student_id, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    predicted_gpa_value = predict_gpa(similarities)
    
    output_2_path = os.path.join(BASE_DIR, 'public', 'output_2.csv')
    os.makedirs(os.path.dirname(output_2_path), exist_ok=True)
    with open(output_2_path, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([predicted_gpa_value])
    
    recommended_courses = []
    for student_id_, sim in similarities[:25]:
        recommended_courses.extend([course[0] for course in train_students[student_id_]])
    
    # Count frequency of courses recommended
    course_counts = {}
    for course in recommended_courses:
        course_counts[course] = course_counts.get(course, 0) + 1
    
    sorted_courses = sorted(course_counts.items(), key=lambda x: x[1], reverse=True)
    filtered_courses = filter_courses(sorted_courses, cdcs, be, columns_as_lists, msc)
    return filtered_courses

def main():
    output_dir = os.path.join(BASE_DIR, 'public')
    os.makedirs(output_dir, exist_ok=True)  

    be = "B.E Electrical & Electronic"
    msc = ""
    selected_course = "CINEMATIC ADAPTATION"
    
    json_file_path = os.path.join(BASE_DIR, 'data.json')
    csv_file_path = os.path.join(BASE_DIR, 'g.csv')
    cdcs_file_path = os.path.join(BASE_DIR, 'CDCs.csv')
    grade_csv_path = os.path.join(BASE_DIR, 'Grade.csv')
    grade_student_path = os.path.join(BASE_DIR, 'Grade_Student.csv')
    grade_branch_path = os.path.join(BASE_DIR, 'GradeDataWithBranch.csv')

    get_courses_data(json_file_path, csv_file_path)
    
    test_data = read_csv(csv_file_path)
    test_data = test_data[['subject', 'courseGrade']]

    cdcs = ["B.E Chemical", "B.E Civil", "B.E Computer Science", "B.E Electrical & Electronic",
            "B.E Electronics & Communication", "B.E Electronics & Instrumentation",
            "B.E Mechanical", "B.Pharm", "M.Sc. Biological Sciences", "M.Sc. Chemistry",
            "M. Sc. Economics", "M.Sc. Mathematics", "M. Sc. Physics"]

    columns_as_lists = read_csv(cdcs_file_path).values.T.tolist()

    student_courses = load_student_courses(grade_csv_path)
    # Add test student with id 1112
    student_courses[1112] = [(row['subject'], row['courseGrade']) for _, row in test_data.iterrows()]

    train_students = {sid: courses for sid, courses in student_courses.items() if sid != 1112}
    test_students = {1112: student_courses[1112]}

    recommended_courses = recommend_courses(1112, train_students, student_courses, cdcs, columns_as_lists, be, msc)
    write_csv(recommended_courses, os.path.join(output_dir, 'output.csv'))

    # Build directed graph for PageRank
    directed_graph = {}
    for index, row in pd.read_csv(grade_student_path, encoding='latin-1').iterrows():
        student_id = row['student_id']
        course = row['subject']
        grade = row['courseGrade']
        if student_id not in directed_graph:
            directed_graph[student_id] = []
        directed_graph[student_id].append((course, grade))

    G = nx.DiGraph()
    for student in directed_graph.keys():
        G.add_node(student)
    for student, courses in directed_graph.items():
        for course, weight in courses:
            # Edge from student -> course with weight=grade (or course->student if preferred)
            G.add_edge(student, course, weight=weight)

    pagerank_scores = nx.pagerank(G)
    student_input_courses = [row['subject'] for _, row in test_data.iterrows()]
    available_courses = [course for course in pagerank_scores.keys() if course not in student_input_courses]
    recommended_courses_pagerank = sorted(available_courses, key=lambda x: pagerank_scores[x], reverse=True)[:40]
    write_csv(recommended_courses_pagerank, os.path.join(output_dir, 'output_3.csv'))

    students_df = pd.read_csv(grade_branch_path)
    studs = {}
    for _, row in students_df.iterrows():
        if msc == "":
            if row['branch1'] == be and pd.isna(row.get('branch2', np.nan)):
                studs.setdefault(row['student_id'], []).append(row['course_id'])
        else:
            if row['branch1'] == be and row.get('branch2') == msc:
                studs.setdefault
