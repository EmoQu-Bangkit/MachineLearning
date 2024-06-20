# Lifecycle of Machine-Learning Project

# Classification Current Condition Model 
The app automatically categorizes activities and moods into various conditions using advanced machine learning algorithms. This feature helps users gain deeper insights into their habits and emotional patterns.

# Classification Mood Condition Model 
For the quality of activities, we classify them into mood conditions like Terrible, Bad, Okay, Good, and Excellent. This feature helps users gain deeper insights into their habits and emotional patterns. By accurately classifying their experiences, users can better understand the factors influencing their mental well-being. This structured approach provides valuable context and enhances the overall user experience.

# Visualization of user activities
The app generates detailed statistics and reports on user activity and mood trends. Also recommend activities to improve user moods.

# Activity Recommendation
The app generates detailed statistics and reports on user activity and mood trends. Also recommend activities to improve user moods.

## 1. Data Colection and Labeling
* Data availability and collection: Data collection is done by collecting laundry review data on the internet (specifically on google maps).
  * Current Condition Dataset
  <br> The review dataset consists of two columns, namely Overall_Quality and Category text. This data is then used to model the current condition. </br>
  
  * Mood Condition dataset
  <br> The review dataset consists of 9 columns, which are the text of the activities performed by the user. This data is then used to model the mood condition.  </br>
  ### Data preprocessing Mood Condition Model:
  * Fixing Columns and Format
  ```sh
  # Drop 'Note' columns because its not used in this model
  df_new=df.drop(columns=['Note'])
  df_new['Time Stamp'] = pd.to_datetime(df_new['Time Stamp']).dt.date
  ```
  * Overall Daily Quality Score
  ```sh
  # Define weights for each activity (higher weight means higher significance)
  activity_weights = {
      'Sleep': 0.2,
      'Study': 0.1,
      'Work': 0.35,
      'Dating':0.05,
      'Self Care':0.05,
      'Traveling':0.05,
      'Entertainment':0.1,
      'Eating':0.05,
      'Workout':0.05
  }
  # Calculate overall day quality score for each day
  updated_df['Overall_Quality'] = updated_df['Quality'] * updated_df['Activities'].map(activity_weights)
  daily_quality = updated_df.groupby(['Time Stamp']).agg({'Overall_Quality': 'mean'}).reset_index()
  
  daily_quality
  ```

   * Outlier Cleaning
  ```sh
  # Menghitung IQR untuk mendeteksi outlier
  Q1 = daily_quality['Overall_Quality'].quantile(0.25)
  Q3 = daily_quality['Overall_Quality'].quantile(0.75)
  IQR = Q3 - Q1
  
  # Menentukan batas bawah dan atas untuk deteksi outlier
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  
  # Membersihkan outlier
  df_cleaned = daily_quality[(daily_quality['Overall_Quality'] >= lower_bound) & (daily_quality['Overall_Quality'] <= upper_bound)]
  ```

   ### Data preprocessing Current Condition Model:
  * Fixing Columns and Format
  ```sh
  # Drop 'Note' columns because its not used in this model
  df_new=df.drop(columns=['Note'])
  df_new['Time Stamp'] = pd.to_datetime(df_new['Time Stamp']).dt.date
  ```
  * Grouping Data
  ```sh
  # Aggregate daily data
  daily_summary = df_new.groupby(['Time Stamp', 'Activities']).agg({'Duration': 'sum'}).reset_index()
  
  daily_summary_pivot = daily_summary.pivot(index='Time Stamp', columns='Activities', values='Duration').fillna(0)
  daily_summary_pivot.reset_index(inplace=True)
  daily_summary_pivot
  ```

   * Setting Threshold to Classify
  ```sh
  # Define thresholds
  sleep_threshold = 7 * 60
  study_threshold = 8 * 60
  work_threshold = 8 * 60
  selfcare_threshold = 1 * 60
  traveling_threshold = 1 * 60
  workout_threshold = 1 * 60
  entertainment_threshold = 1 * 60
  eating_threshold = 1 * 60
  
  def classify_day(row):
      sleep = row.get('Sleep', 0)
      study = row.get('Study', 0)
      dating = row.get('Dating', 0)
      work = row.get('Work', 0)
      selfcare = row.get('Self Care', 0)
      traveling = row.get('Traveling', 0)
      workout = row.get('Workout', 0)
      entertainment = row.get('Entertainment', 0)
      eating = row.get('Eating', 0)
  
      if sleep < sleep_threshold and work > work_threshold:
          return 'Stressful Overload'
      elif sleep < sleep_threshold and study > study_threshold:
          return 'Stressful Overload'
      elif workout > workout_threshold and eating > eating_threshold:
          return 'Fitness Fanatic'
      elif traveling > traveling_threshold and entertainment > entertainment_threshold:
          return 'Active Explorer'
      else:
          return 'Balanced Achiever'
  # Apply classification
  daily_summary_pivot['Day_Condition'] = daily_summary_pivot.apply(classify_day, axis=1)
  daily_summary_pivot
  ```
