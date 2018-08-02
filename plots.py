
# ########## Basic data summary plots ##########


def add_age_column(users):
    """
    :param users: a dataframe containing user's data
    :return: a dataframe containing a clean set of the patient's ages
    """
    ages = users.copy()
    ages = ages.dropna(subset=['BirthYear'])
    ages['Age'] = 2018 - ages['BirthYear']
    ages = ages.dropna(subset=['Age'])
    return ages


def keep_only_mild_users(users):
    """
    :param users: a dataframe containing user's data
    :return: a dataframe containing only healthy patients or patients with a mild level of Parkinson
    """
    mild_users = users[(users.Parkinsons == False) | ((users.Parkinsons == True) & (users.Impact == "Mild"))]
    return mild_users


def ages_plot(fig, ages):
    """
    Create a histogram of patient's ages- health vs. sick
    :param fig: a figure
    :param ages: a dataframe containing a clean set of the patient's ages
    """
    # AGES sick vs healthy
    ax = fig.add_subplot(2, 2, 1, title="Participants ages")
    ax.hist([ages.Age[(ages.Parkinsons == True)], ages.Age[(ages.Parkinsons == False)]], bins=15, histtype='bar',
            color=['#FFCC00', '#33CC00'], density=True)
    ax.set_xlabel("Age")
    ax.set_ylabel("")
    ax.legend(["Sick", "Healthy"])


def genders_plot(fig, mild_users):
    """
    create a pie chart of the patient's gender partition
    :param fig: a figure
    :param mild_users: a dataframe containing only healthy patients or patients with a mild level of Parkinson
    """
    ax = fig.add_subplot(2, 2, 2, title="Participants genders")
    ax.pie(mild_users.Gender.value_counts(), labels=["Male", "Female"], colors=["#99CCFF", "#CCFFFF"],
           startangle=90, autopct='%1.1f%%')
    ax.axis('equal')


def diagnosis_plot(fig, mild_users):
    """
    create a pie chart showing the patient's partition to mildly sick and healthy
    :param fig: a figure
    :param mild_users: a dataframe containing only healthy patients or patients with a mild level of Parkinson
    """
    ax = fig.add_subplot(2, 2, 3, title="Parkinsons diagnosis")
    ax.pie(mild_users.Parkinsons.value_counts(), labels=["Sick", "Healthy"], colors=["#c6ecc6", "#40bf40"],
           startangle=90, autopct='%1.1f%%')
    ax.axis('equal')


def sickness_level_plot(fig, users):
    """
    create a pie chart showing the patient's partition by the severity of the sickness
    :param fig: a figure
    :param users: a dataframe containing user's data
    """
    ax = fig.add_subplot(2, 2, 4, title="Sickness Severity")
    sick_lvl = users.loc[(users['Impact'] != "------")]
    sick_lvl = sick_lvl.dropna(subset=['Impact'])

    patches, texts, per_col = ax.pie(sick_lvl.Impact.value_counts(), labels=["Medium", "Mild", "Severe"],
                                     colors=["#ffcccc", "#ff4d4d", "#b30000"], startangle=90, autopct='%1.1f%%',
                                     textprops={'fontsize': 8})
    for per in per_col:
        per.set_color("#FFFFFF")
    ax.axis('equal')


def mit_updrs_distribution(fig, users):
    """
    create a box plot showing the MIT patient's UPDRS scores, divided to healthy vs. sick
    :param fig: a figure
    :param users: a dataframe containing user's data
]    """
    ax = fig.add_subplot(1, 2, 1, title="UPDRS - Healthy vs. Sick")
    b_plt = ax.boxplot([users.UPDRS[users.Parkinsons == False],
                        users.UPDRS[users.Parkinsons == True]], labels=["Healthy", "Sick"],
                       patch_artist=True)
    colors = ['lightgreen', '#FF6666']
    for patch, color in zip(b_plt['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("UPDRS")
    ax.set_xlabel("")


def mit_diagnosis(fig, users):
    """
    create a pie chart showing the patient's partition to sick vs. healthy
    :param fig: a figure
    :param users: a dataframe containing user's data
    """
    ax = fig.add_subplot(1, 2, 2, title="Patients distribution")
    ax.pie(users.Parkinsons.value_counts(), labels=["Sick", "Healthy"], colors=["#99CCFF", "#CCFFFF"],
           startangle=90, autopct='%1.1f%%')
    ax.axis('equal')


# ########## Kaggle features plots ##########

def LR_Hold_Time(fig, healthy, sick):
    """
    create a bar plot showing the differences between the means of the left and right Hold times
        in healthy vs. sick patients
    :param fig: a figure
    :param healthy: a dataframe containing only healthy patient's data
    :param sick: a dataframe containing only sick patient's data
]    """
    ax = fig.add_subplot(1, 3, 1, title="Mean differences of HoldTime between Left/Right Keys")
    healthy_mean = healthy.dropna(subset=['mean_diff_L_R_HoldTime'])
    sick_mean = sick.dropna(subset=['mean_diff_L_R_HoldTime'])
    ax.hist([sick_mean.mean_diff_L_R_HoldTime, healthy_mean.mean_diff_L_R_HoldTime], bins=20, histtype='bar',
            color=['#FF6666', 'lightgreen'], density=True)
    ax.legend(["Sick", "Healthy"])
    ax.set_ylabel("Density")


def LR_RL_Latency_Time(fig, healthy, sick):
    """
    create a bar plot showing the differences between the means of the left-to-right and right-to-left Latency times
        in healthy vs. sick patients
    :param fig: a figure
    :param healthy: a dataframe containing only healthy patient's data
    :param sick: a dataframe containing only sick patient's data
    """
    ax = fig.add_subplot(1, 3, 2, title="Mean differences of LatencyTime between Left-to-Right/Right-to-Left movements")
    healthy_mean = healthy.dropna(subset=['mean_diff_LR_RL_LatencyTime'])
    sick_mean = sick.dropna(subset=['mean_diff_LR_RL_LatencyTime'])
    ax.hist([sick_mean.mean_diff_LR_RL_LatencyTime, healthy_mean.mean_diff_LR_RL_LatencyTime], bins=20,
            histtype='bar', color=['#FF6666', 'lightgreen'], density=True)
    ax.legend(["Sick", "Healthy"])


def LL_RR_Latency_Time(fig, healthy, sick):
    """
    create a bar plot showing the differences between the means of the left-to-left and right-to-right Latency times
        in healthy vs. sick patients
    :param fig: a figure
    :param healthy: a dataframe containing only healthy patient's data
    :param sick: a dataframe containing only sick patient's data
    """
    ax = fig.add_subplot(1, 3, 3, title="Mean differences of LatencyTime between Left-to-Left/Right-to-Right movements")
    healthy_mean = healthy.dropna(subset=['mean_diff_LL_RR_LatencyTime'])
    sick_mean = sick.dropna(subset=['mean_diff_LL_RR_LatencyTime'])
    ax.hist([sick_mean.mean_diff_LL_RR_LatencyTime, healthy_mean.mean_diff_LL_RR_LatencyTime], bins=20,
            histtype='bar', color=['#FF6666', 'lightgreen'], density=True)
    ax.legend(["Sick", "Healthy"])


def lFlight_mean(fig, sick, healthy):
    """
    create a bar plot showing the differences between the means of the left Flight times in healthy vs. sick patients
    :param fig: a figure
    :param healthy: a dataframe containing only healthy patient's data
    :param sick: a dataframe containing only sick patient's data
    """
    ax = fig.add_subplot(2, 2, 1, title="Left Flight Time mean")
    sick_clean = sick.dropna(subset=['L_FlightTime_mean'])
    healthy_clean = healthy.dropna(subset=['L_FlightTime_mean'])
    ax.hist([sick_clean.L_FlightTime_mean, healthy_clean.L_FlightTime_mean], bins=20, histtype='bar',
            color=['#FF6666', 'lightgreen'], density=True)
    ax.legend(["Sick", "Healthy"])
    ax.set_xlabel("Means")
    ax.set_ylabel("Density")


def lFlight_std(fig, sick, healthy):
    """
    create a bar plot showing the differences between the STD's of the left Flight times in healthy vs. sick patients
    :param fig: a figure
    :param healthy: a dataframe containing only healthy patient's data
    :param sick: a dataframe containing only sick patient's data
    """
    ax = fig.add_subplot(2, 2, 2, title="Left Flight Time standard deviation")
    sick_clean = sick.dropna(subset=['L_FlightTime_std'])
    healthy_clean = healthy.dropna(subset=['L_FlightTime_std'])
    ax.hist([sick_clean.L_FlightTime_std, healthy_clean.L_FlightTime_std], bins=20, histtype='bar',
            color=['#FF6666', 'lightgreen'], density=True)
    ax.legend(["Sick", "Healthy"])
    ax.set_xlabel("Means")
    ax.set_ylabel("Density")


def lFlight_kurtosis(fig, sick, healthy):
    """
    create a bar plot showing the differences between the kurtosis of the left Flight times in healthy vs. sick patients
    :param fig: a figure
    :param healthy: a dataframe containing only healthy patient's data
    :param sick: a dataframe containing only sick patient's data
    """
    ax = fig.add_subplot(2, 2, 3, title="Left Flight Time kurtosis")
    sick_clean = sick.dropna(subset=['L_FlightTime_kurtosis'])
    healthy_clean = healthy.dropna(subset=['L_FlightTime_kurtosis'])
    ax.hist([sick_clean.L_FlightTime_kurtosis, healthy_clean.L_FlightTime_kurtosis], bins=20, histtype='bar',
            color=['#FF6666', 'lightgreen'],
            density=True)
    ax.legend(["Sick", "Healthy"])
    ax.set_xlabel("Means")
    ax.set_ylabel("Density")


def lFlight_skew(fig, sick, healthy):
    """
    create a bar plot showing the differences between the skewness of the left Flight times in healthy vs. sick patients
    :param fig: a figure
    :param healthy: a dataframe containing only healthy patient's data
    :param sick: a dataframe containing only sick patient's data
    """
    ax = fig.add_subplot(2, 2, 4, title="Left Flight Time skewness")
    sick_clean = sick.dropna(subset=['L_FlightTime_skew'])
    healthy_clean = healthy.dropna(subset=['L_FlightTime_skew'])
    ax.hist([sick_clean.L_FlightTime_skew, healthy_clean.L_FlightTime_skew], bins=20, histtype='bar',
            color=['#FF6666', 'lightgreen'], density=True)
    ax.legend(["Sick", "Healthy"])
    ax.set_xlabel("Means")
    ax.set_ylabel("Density")


# ########## MIT features plots ##########

def iqr_histogram(fig, data):
    """
    create a histogra, showing the differences between the Hold time IQR of healthy vs. sick patients
    :param fig: a figure
    :param data: a dataframe containing the MIT user's data
    """
    ax = fig.add_subplot(1, 2, 1, title="HoldTime IQR of patients")
    ax.hist([data[data.Parkinsons == True].hold_agg_iqr, data[data.Parkinsons == False].hold_agg_iqr],
            color=['tomato', 'lightgreen'])
    ax.legend(('Sick', 'Healthy'))


def outliers_histogram(fig, data):
    """
    create a histogram showing the outliers proportion of MIT patients
    :param fig: a figure
    :param data: a dataframe containing the MIT user's data
    """
    ax = fig.add_subplot(1, 2, 2, title="Outliers proportion of patients")
    ax.hist([data[data.Parkinsons == True].hold_agg_outliers, data[data.Parkinsons == False].hold_agg_outliers],
            color=['tomato', 'lightgreen'])
    ax.legend(('Sick', 'Healthy'))


def boxplot_nqi_score(fig, position, data, title):
    """
    create a box plot showing the predicted NQI scores of health vs. sick patients
    :param fig: a figure
    :param position: the position of the plot in the figure
    :param data: a dataframe containing the MIT user's data
    :param title: the title for the plot
    """
    ax = fig.add_subplot(1, 2, position, title=title)
    ax.boxplot([data.predicted_nqi[data.Parkinsons == True], data.predicted_nqi[data.Parkinsons == False]],
               labels=["Sick", "Healthy"], patch_artist=True)
    ax.set_ylabel("NQI Score")
