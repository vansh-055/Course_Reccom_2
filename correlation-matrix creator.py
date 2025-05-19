import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


csv_file_path = os.path.join(os.path.dirname(__file__), 'Test.csv')

# Read data from the CSV file
data = pd.read_csv(csv_file_path)

average_grades = data.pivot_table(values='course_grade', index='branch1', columns='course_id', aggfunc='mean')

# Create the heatmap
sns.heatmap(average_grades, cmap='coolwarm')  # You can choose a different colormap

# Customize the plot (optional)
plt.xlabel('Course Name')
plt.ylabel('Student Branch')
plt.title('Average Grade by Student Branch and Course')
plt.show()
