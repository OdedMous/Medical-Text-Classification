import pandas as pd



def process_raw_data(data_path):
    full_data = pd.read_csv(data_path)

    # drop None values from the relevant columns
    full_data = full_data.dropna(subset=['description', 'transcription', 'medical_specialty']).reset_index(drop=True)

    full_data['description'] = full_data['description'].str.lower() # convert text data to lower case
    full_data['transcription'] = full_data['transcription'].str.lower() # convert text data to lower case
    full_data["medical_specialty"] = full_data["medical_specialty"].str.strip() # delete leading / trailing whitespaces


    #data["words"] =  data["description"].apply(lambda x: nltk.word_tokenize(x))  # description         transcription
    #stop_words = set(stopwords.words('english'))
    #data['words_without_stopwords'] = data["words"].apply(lambda x: [word for word in x if word not in (stop_words)])

    # drop general categories (for example "Surgery" category is kind of superset as there can be surgeries belonging to specializations like cardiology,neurolrogy etc)
    general_categories_rows = full_data["medical_specialty"].isin(["Surgery", 'SOAP / Chart / Progress Notes', 'Office Notes', 'Consult - History and Phy.', 'Emergency Room Reports', 'Discharge Summary', 'Pain Management', 'General Medicine'])
    data = full_data.drop(full_data[general_categories_rows].index)
    data = data.reset_index(drop=True)

    # Combine similar categories
    data["medical_specialty"] = data["medical_specialty"].str.replace("Neurosurgery", "Neurology")

    # add "labels" column
    data['medical_specialty'] = pd.Categorical(data['medical_specialty'])
    data['labels'] = data['medical_specialty'].cat.codes


    categories_mapping = dict(enumerate(data['medical_specialty'].cat.categories))

    # Take only top 5 categories
    top_categories_num = 5
    cause_dist = data['medical_specialty'].value_counts()[0:top_categories_num]
    cause_dist_unseen_cat = data['medical_specialty'].value_counts()[top_categories_num:]

    test_unseen_categories = data[data["medical_specialty"].isin(cause_dist_unseen_cat.keys())]
    test_unseen_categories = test_unseen_categories.reset_index(drop=True)

    # take in unseen data only categories which have more than 50 samples
    unseen_categories_groups  = test_unseen_categories.groupby(test_unseen_categories['medical_specialty'])
    test_unseen_categories = unseen_categories_groups.filter(lambda x:x.shape[0] > 50)

    unseen_categories_mapping = dict(enumerate(test_unseen_categories['medical_specialty'].cat.categories))

    data = data[data["medical_specialty"].isin(cause_dist.keys())]
    data = data.reset_index(drop=True)

    return data, test_unseen_categories



