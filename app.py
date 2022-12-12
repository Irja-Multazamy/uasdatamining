import streamlit as st
import os

# dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg') 
import joblib

race_label = {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4}
gender_label = {'female': 0, 'male': 1}
level_label = {'bachelors degree': 0, 'some college': 1, 'masters degree': 2, 'associates degree': 3, 'high school': 4}
lunch_label = {'standard': 0, 'free/reduced': 1}
test_label = {'none': 0, 'completed': 1}
label = {'Failed': 0, 'Completed': 1}

# Get the Keys
def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value
# Find the Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key

def main():
	st.title("Web Datamining")
	st.write("""
	# Klasifikasi Student Performance in Exams Dataset
	Applikasi Berbasis Web untuk Mengklasifikasi Kinerja Siswa dalam Ujian
	""")

	activities = ['Dataset','Prediction','About']
	choices = st.sidebar.selectbox("Select Activity",activities)

	if choices == 'Dataset':
		st.subheader("Dataset")
		data = pd.read_csv("https://raw.githubusercontent.com/Irja-Multazamy/datamining/main/StudentsPerformance.csv")
		st.dataframe(data.head(5))

		if st.checkbox("Show Summary of Dataset"):
			st.write(data.describe())

		# Show Columns By Selection
		if st.checkbox("Select Columns To Show"):
			all_columns = data.columns.tolist()
			selected_columns = st.multiselect('Select',all_columns)
			new_df = data[selected_columns]
			st.dataframe(new_df)

	if choices == 'Prediction':
		st.subheader("Prediction")

		gender = st.selectbox('Select Gender :',tuple(gender_label.keys()))
		race = st.selectbox('Select race/ethnicity (kelompok etnis/suku) :',tuple(race_label.keys()))
		level = st.selectbox('Select parental level of education (tingkat pendidikan orang tua) :',tuple(level_label.keys()))
		lunch = st.selectbox('Select lunch (sangu) :',tuple(lunch_label.keys()))
		test = st.selectbox('Select test preparation (kursus pesiapan ujian) :',tuple(test_label.keys()))
		math = st.number_input("Masukkan Nilai Matematika :")
		reading = st.number_input("Masukkan Nilai Kemampuan Membaca :")
		writing = st.number_input("Masukkan Nilai Kemampuan Menulis :")

		k_gender = get_value(gender,gender_label)
		k_race = get_value(race,race_label)
		k_level = get_value(level,level_label)
		k_lunch = get_value(lunch,lunch_label)
		k_test = get_value(test,test_label)

		masukan = {
		"Gender":gender,
		"Race/ethnicity":race,
		"Level":level,
		"Lunch":lunch,
		"Test preparation":test,
		"Math score":math,
		"Reading score":reading,
		"Writing score":writing,
		}
		st.subheader("Data yang dimasukkan:")
		st.json(masukan)

		if (math >= 70):
			math = 1
		elif (math <= 70):
			math = 0

		if (reading >= 70):
			reading = 1
		elif (reading <= 70):
			reading = 0

		if (writing >= 70):
			writing = 1
		elif (writing <= 70):
			writing = 0

		st.subheader("Data Sample")
		sample_data = [k_gender,k_race,k_level,k_lunch,k_test,math,reading,writing]
		st.write(sample_data)

		prep_data = np.array(sample_data).reshape(1, -1)

		st.subheader("Metode yang digunakan:")
		method = st.selectbox("Pilih metode yang digunakan:",['Naive Bayes','Decision Tree','Random Forest'])

		if st.button("Hasil Klasifikasi"):
			if method == 'Naive Bayes':
				predictor = joblib.load("bayes")
				prediction = predictor.predict(prep_data)
				st.write(prediction)
			if method == 'Random Forest':
				predictor = joblib.load("rforest")
				prediction = predictor.predict(prep_data)
				st.write(prediction)
			if method == 'Decision Tree':
				predictor = joblib.load("dtree")
				prediction = predictor.predict(prep_data)
				st.write(prediction)
			final_result = get_key(prediction,label)
			st.success(final_result)

	if choices == 'About':
		st.subheader("Tentang Penulis")

		st.write("Created By. Irja' Multazamy")
		st.write("NIM. 200411100155")
		st.write("Penambangan Data IF 5B")

if __name__ == '__main__':
	main()