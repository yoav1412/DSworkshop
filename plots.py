from constants import *
import pandas as pd
import matplotlib.pyplot as plt

kaggle_taps = pd.read_csv(KAGGLE_TAPS_INPUT)  # todo: remove
kaggle_users = pd.read_csv(KAGGLE_USERS_INPUT)  # todo: remove


def add_age_column(users):
    ages = users.copy()
    ages = ages.dropna(subset=['BirthYear'])
    ages['Age'] = 2018 - ages['BirthYear']
    ages = ages.dropna(subset=['Age'])
    return ages


def keep_only_mild_users(users):
    mild_users = users[(users.Parkinsons == False) | ((users.Parkinsons == True) & (users.Impact == "Mild"))]
    return mild_users


def ages_plot(fig, ages):
    # AGES sick vs healthy
    ax = fig.add_subplot(2, 2, 1, title="Participants ages")
    ax.hist([ages.Age[(ages.Parkinsons == True)], ages.Age[(ages.Parkinsons == False)]], bins=15, histtype='bar',
             color=['#FFCC00', '#33CC00'], density=True)
    ax.set_xlabel("Age")
    ax.set_ylabel("")
    ax.legend(["Sick", "Healthy"])
    # plt.figure(figsize=(2, 1))


def genders_plot(fig, mild_users):
    ax = fig.add_subplot(2, 2, 2, title="Participants genders")
    ax.pie(mild_users.Gender.value_counts(), labels=["Male", "Female"], colors=["#99CCFF", "#CCFFFF"],
            startangle=90)  # 'autopct='%1.1f%%')
    ax.axis('equal')


def diagnosis_plot(fig, mild_users):
    ax = fig.add_subplot(2, 2, 3, title="Parkinsons diagnosis")
    ax.pie(mild_users.Parkinsons.value_counts(), labels=["Sick", "Healthy"], colors=["#99CCFF", "#CCFFFF"],
            startangle=90, autopct='%1.1f%%')
    ax.axis('equal')


def sickness_level_plot(fig, users):
    ax = fig.add_subplot(2, 2, 4, title="Sickness Severity")
    sick_lvl = users.loc[(users['Impact'] != "------")]
    sick_lvl = sick_lvl.dropna(subset=['Impact'])

    patches, texts, per_col = ax.pie(sick_lvl.Impact.value_counts(), labels=["Medium", "Mild", "Severe"],
                                      colors=["#006666", "#003366", "#006633"], startangle=90, autopct='%1.1f%%',
                                      textprops={'fontsize': 8})
    for per in per_col:
        per.set_color("#FFFFFF")
    ax.axis('equal')


mit_users = pd.read_csv(MIT_USERS_INPUT)  # todo: remove
mit_taps = pd.read_csv(MIT_TAPS_INPUT)  # todo: remove


def mit_updrs_distribution(fig, users):
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
    ax = fig.add_subplot(1, 2, 2, title="Patients distribution")
    ax.pie(users.Parkinsons.value_counts(), labels=["Sick", "Healthy"], colors=["#99CCFF", "#CCFFFF"],
            startangle=90, autopct='%1.1f%%')
    ax.axis('equal')


'''
feat = pd.read_csv(r"C:\\Users\\Nili\\PycharmProjects\\DSworkshop\\Data\\features.csv")

###sick vs. healthy###
sick = feat.loc[(feat['Parkinsons'] == True)]
healthy = feat.loc[(feat['Parkinsons'] == False)]

###means

# plt.hist(sick_mean.L_FlightTime_mean,bins=20,histtype='bar', color='#009999' )
# plt.xlabel("means")
# plt.ylabel("")
# plt.title("sick - Left Flight Time mean\n")
# plt.show()

###use the 2 following to show all wanted values from features table###
####2 ovelapping histograms
plt.subplot(2,2,1)
sick_mean = sick.dropna(subset=['L_FlightTime_mean'])
healthy_mean = healthy.dropna(subset=['L_FlightTime_mean'])
plt.hist(sick_mean.L_FlightTime_mean,bins=20,histtype='bar', color='#FF3300', density=True)
plt.hist(healthy_mean.L_FlightTime_mean,bins=20,histtype='bar', color='#00CC66', density=True, alpha=0.8)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("Left Flight Time mean\n")
#plt.show()

plt.subplot(2, 2, 2)
sick_mean2 = sick.dropna(subset=['L_FlightTime_std'])
healthy_mean2 = healthy.dropna(subset=['L_FlightTime_std'])
plt.hist(sick_mean2.L_FlightTime_std,bins=20,histtype='bar', color='#FF3300', density=True)
plt.hist(healthy_mean2.L_FlightTime_std,bins=20,histtype='bar', color='#00CC66', density=True, alpha=0.8)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("Left Flight Time std\n")

plt.subplot(2, 2, 3)
sick_mean3 = sick.dropna(subset=['L_FlightTime_kurtosis'])
healthy_mean3 = healthy.dropna(subset=['L_FlightTime_kurtosis'])
plt.hist(sick_mean3.L_FlightTime_kurtosis,bins=20,histtype='bar', color='#FF3300', density=True)
plt.hist(healthy_mean3.L_FlightTime_kurtosis,bins=20,histtype='bar', color='#00CC66', density=True, alpha=0.8)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("Left Flight Time kurtosis\n")

plt.subplot(2, 2, 4)
sick_mean4 = sick.dropna(subset=['L_FlightTime_skew'])
healthy_mean4 = healthy.dropna(subset=['L_FlightTime_skew'])
plt.hist(sick_mean4.L_FlightTime_skew,bins=20,histtype='bar', color='#FF3300', density=True)
plt.hist(healthy_mean4.L_FlightTime_skew,bins=20,histtype='bar', color='#00CC66', density=True, alpha=0.8)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("Left Flight Time skewness\n")
# plt = gcf()
#st = plt.suptitle("Title centered above all subplots", fontsize=14)
#st.set_y(0.95)
#plt.subplots_adjust(top=0.85)

plt.tight_layout()
plt.show()

### bars ###L Flight
cols=['#FF6666','lightgreen']
plt.subplot(2, 2, 1)
healthy_mean5 = healthy.dropna(subset=['mean_diff_L_R_HoldTime'])
sick_mean5 = sick.dropna(subset=['mean_diff_L_R_HoldTime'])
plt.hist([sick_mean5.mean_diff_L_R_HoldTime,healthy_mean5.mean_diff_L_R_HoldTime],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
#plt.xlabel("means")
plt.ylabel("Density")
plt.title("mean differences LR Hold Time\n")

plt.subplot(2, 2, 2)
healthy_mean6 = healthy.dropna(subset=['mean_diff_LR_RL_LatencyTime'])
sick_mean6 = sick.dropna(subset=['mean_diff_LR_RL_LatencyTime'])
plt.hist([sick_mean6.mean_diff_LR_RL_LatencyTime,healthy_mean6.mean_diff_LR_RL_LatencyTime],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
#plt.xlabel("means")
plt.ylabel("Density")
plt.title("mean differences LR-RL Latency Time\n")

plt.subplot(2, 2, 3)
healthy_mean7 = healthy.dropna(subset=['mean_diff_LL_RR_LatencyTime'])
sick_mean7 = sick.dropna(subset=['mean_diff_LL_RR_LatencyTime'])
plt.hist([sick_mean7.mean_diff_LL_RR_LatencyTime,healthy_mean7.mean_diff_LL_RR_LatencyTime],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
#plt.xlabel("means")
plt.ylabel("Density")
plt.title("mean differences LL-RR Latency Time\n")

# plt.subplot(2, 2, 4)
# healthy_mean8 = healthy.dropna(subset=['RR_FlightTime_kurtosis'])
# sick_mean8 = sick.dropna(subset=['RR_FlightTime_kurtosis'])
# plt.hist([sick_mean8.RR_FlightTime_kurtosis,healthy_mean8.RR_FlightTime_kurtosis],bins=20,histtype='bar', color=cols,density=True)
# plt.legend(["sick","healthy"])
# #plt.xlabel("means")
# plt.ylabel("")
# plt.title("RR Flight Time kurtosis\n")
# plt = gcf()
# st = plt.suptitle("Title centered above all subplots", fontsize=14)
#st.set_y(0.95)
#plt.subplots_adjust(top=0.85)
plt.tight_layout()
plt.show()





###bars ## LR Latency
cols=['#FF6666','lightgreen']
plt.subplot(2, 2, 1)
healthy_1 = healthy.dropna(subset=['LR_LatencyTime_mean'])
sick_1 = sick.dropna(subset=['LR_LatencyTime_mean'])
plt.hist([sick_1.LR_LatencyTime_mean,healthy_1.LR_LatencyTime_mean],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LR Latency Time Time mean\n")

plt.subplot(2, 2, 2)
healthy_2 = healthy.dropna(subset=['LR_LatencyTime_std'])
sick_2 = sick.dropna(subset=['LR_LatencyTime_std'])
plt.hist([sick_2.LR_LatencyTime_std,healthy_2.LR_LatencyTime_std],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LR Latency Time Time std\n")

plt.subplot(2, 2, 3)
healthy_3 = healthy.dropna(subset=['LR_LatencyTime_skew'])
sick_3 = sick.dropna(subset=['LR_LatencyTime_skew'])
plt.hist([sick_3.LR_LatencyTime_skew,healthy_3.LR_LatencyTime_skew],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LR Latency Time Time skewness\n")

plt.subplot(2, 2, 4)
healthy_4 = healthy.dropna(subset=['LR_LatencyTime_kurtosis'])
sick_4 = sick.dropna(subset=['LR_LatencyTime_kurtosis'])
plt.hist([sick_4.LR_LatencyTime_kurtosis,healthy_4.LR_LatencyTime_kurtosis],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LR Latency Time Time skewness\n")

plt.tight_layout()
plt.show()

##LL Flight Mean
cols=['lightgreen','#FF6666']
plt.subplot(2, 2, 1)
healthy_5 = healthy.dropna(subset=['LL_FlightTime_mean'])
sick_5 = sick.dropna(subset=['LL_FlightTime_mean'])
plt.hist([sick_5.LL_FlightTime_mean,healthy_5.LL_FlightTime_mean],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LL Flight Time mean\n")

plt.subplot(2, 2, 2)
healthy_6 = healthy.dropna(subset=['LL_FlightTime_std'])
sick_6 = sick.dropna(subset=['LL_FlightTime_std'])
plt.hist([sick_6.LL_FlightTime_std,healthy_6.LL_FlightTime_std],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LL Flight Time std\n")

plt.subplot(2, 2, 3)
healthy_7 = healthy.dropna(subset=['LL_FlightTime_skew'])
sick_7 = sick.dropna(subset=['LL_FlightTime_skew'])
plt.hist([sick_7.LL_FlightTime_skew,healthy_7.LL_FlightTime_skew],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LL Flight Time skewness\n")

plt.subplot(2, 2, 4)
healthy_8 = healthy.dropna(subset=['LL_FlightTime_kurtosis'])
sick_8 = sick.dropna(subset=['LL_FlightTime_kurtosis'])
plt.hist([sick_8.LL_FlightTime_kurtosis,healthy_8.LL_FlightTime_kurtosis],bins=20,histtype='bar', color=cols,density=True)
plt.legend(["sick","healthy"])
plt.xlabel("means")
plt.ylabel("Density")
plt.title("LL Flight Time skewness\n")

plt.tight_layout()
plt.show()


##R Hold



"""
##std
sick_std = sick.dropna(subset=['L_FlightTime_std'])
plt.hist(sick_std.L_FlightTime_std,bins=20,histtype='bar', color='#009999' )
plt.xlabel("std")
plt.ylabel("")
plt.title("sick - Left Flight Time std\n")
plt.show()

healthy_std = healthy.dropna(subset=['L_FlightTime_std'])
plt.hist(healthy_std.L_FlightTime_std,bins=20,histtype='bar', color='#009999' )
plt.xlabel("std")
plt.ylabel("")
plt.title("Healthy - Left Flight Time std\n")
plt.show()

##skew
sick_skew = sick.dropna(subset=['L_FlightTime_skew'])
plt.hist(sick_skew.L_FlightTime_skew,bins=20,histtype='bar', color='#009999' )
plt.xlabel("skewness")
plt.ylabel("")
plt.title("sick - Left Flight Time skewness\n")
plt.show()

healthy_skew = healthy.dropna(subset=['L_FlightTime_skew'])
plt.hist(healthy_skew.L_FlightTime_skew,bins=20,histtype='bar', color='#009999' )
plt.xlabel("skewness")
plt.ylabel("")
plt.title("Healthy - Left Flight Time skewness\n")
plt.show()

##kurtosis
sick_kurtosis = sick.dropna(subset=['L_FlightTime_kurtosis'])
plt.hist(sick_kurtosis.L_FlightTime_kurtosis,bins=20,histtype='bar', color='#009999' )
plt.xlabel("kurtosis")
plt.ylabel("")
plt.title("sick - Left Flight Time kurtosis\n")
plt.show()

healthy_kurtosis = healthy.dropna(subset=['L_FlightTime_kurtosis'])
plt.hist(healthy_kurtosis.L_FlightTime_kurtosis,bins=20,histtype='bar', color='#009999' )
plt.xlabel("kurtosis")
plt.ylabel("")
plt.title("Healthy - Left Flight Time kurtosis\n")
plt.show()

#mean diff-hold time
sick_mean_diff = sick.dropna(subset=['mean_diff_L_R_HoldTime'])
plt.hist(sick_mean_diff.mean_diff_L_R_HoldTime,bins=20,histtype='bar', color='#009999' )
plt.xlabel("mean differences")
plt.ylabel("")
plt.title("sick - mean differences in hold times\n")
plt.show()

healthy_mean_diff = healthy.dropna(subset=['mean_diff_L_R_HoldTime'])
plt.hist(healthy_mean_diff.mean_diff_L_R_HoldTime,bins=20,histtype='bar', color='#009999' )
plt.xlabel("mean differences")
plt.ylabel("")
plt.title("Healthy - mean differences in hold times\n")
plt.show()

#mean diff-latency - LR RL
sick_latency_diff1 = sick.dropna(subset=['mean_diff_LR_RL_LatencyTime'])
plt.hist(sick_latency_diff1.mean_diff_LR_RL_LatencyTime,bins=20,histtype='bar', color='#009999' )
plt.xlabel("Latency differences - LR RL")
plt.ylabel("")
plt.title("sick - mean differences in Latency - LR RL\n")
plt.show()

healthy_latency_diff1 = healthy.dropna(subset=['mean_diff_LR_RL_LatencyTime'])
plt.hist(healthy_latency_diff1.mean_diff_LR_RL_LatencyTime,bins=20,histtype='bar', color='#009999' )
plt.xlabel("Latency differences - LR RL")
plt.ylabel("")
plt.title("Healthy - mean differences in Latency - LR RL\n")
plt.show()

#mean diff-latency - LL RR
sick_latency_diff2 = sick.dropna(subset=['mean_diff_LL_RR_LatencyTime'])
plt.hist(sick_latency_diff2.mean_diff_LL_RR_LatencyTime,bins=20,histtype='bar', color='#009999' )
plt.xlabel("Latency differences - LL RR")
plt.ylabel("")
plt.title("sick - mean differences in Latency - LL RR\n")
plt.show()

healthy_latency_diff2 = healthy.dropna(subset=['mean_diff_LL_RR_LatencyTime'])
plt.hist(healthy_latency_diff2.mean_diff_LL_RR_LatencyTime,bins=20,histtype='bar', color='#009999' )
plt.xlabel("Latency differences - LL RR")
plt.ylabel("")
plt.title("Healthy - mean differences in Latency - LL RR\n")
plt.show()
"""

###mild vs. medium & severe###
mild = feat.loc[(feat['Impact'] == "Mild")]
med_sev = feat.loc[(feat['Impact'] == "Medium") | (feat['Impact'] == "Severe")]

###means
mild_mean = mild.dropna(subset=['L_FlightTime_mean'])
plt.hist(mild_mean.L_FlightTime_mean,bins=20,histtype='bar', color='#009999' )
plt.xlabel("means")
plt.ylabel("")
plt.title("mild - Left Flight Time mean\n")
plt.show()

medsev_mean = med_sev.dropna(subset=['L_FlightTime_mean'])
plt.hist(medsev_mean.L_FlightTime_mean,bins=20,histtype='bar', color='#009999' )
plt.xlabel("means")
plt.ylabel("")
plt.title("Medium & Severe - Left Flight Time mean\n")
plt.show()

####ages in sick and in healthy?
####i dont think gender division is relevant


###merging data###
#merged = pd.merge(taps,users, on="ID")
merged= pd.read_csv(r"C:\\Users\\Nili\\PycharmProjects\\DSworkshop\\Data\\merged.csv")


###mereged - means###

###Latency Time###
latency_mean_id = merged.groupby(["Parkinsons","ID"]).LatencyTime.mean().reset_index()
latency_agg_id = merged.groupby(["Parkinsons","ID"]).LatencyTime.agg(stats.entropy).reset_index()
plt.hist([latency_agg_id.LatencyTime[latency_mean_id.Parkinsons==False], latency_agg_id.LatencyTime[latency_mean_id.Parkinsons==True]],bins=20,histtype='bar', color=['#66FF99','#FF6666'], density=True)
plt.hist([latency_mean_id.LatencyTime[latency_mean_id.Parkinsons==False], latency_mean_id.LatencyTime[latency_mean_id.Parkinsons==True]],bins=20,histtype='bar', color=['lightgreen','#FF6666'], density=True)
plt.xlabel("Latency means")
plt.ylabel("Density")
plt.legend(["Sick","Healthy"])
plt.title("Latency Means - Healthy vs. Sick\n")
plt.show()

plt.subplot(2, 2, 1)
b_plt1 = plt.boxplot([latency_mean_id.LatencyTime[latency_mean_id.Parkinsons==False], latency_mean_id.LatencyTime[latency_mean_id.Parkinsons==True]], labels=["Healthy","Sick"],patch_artist=True)
#color the chart
colors = ['lightgreen','#FF6666']
for patch, color in zip(b_plt1['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Latency Time means")
plt.ylabel("")
plt.title("Latency Means - Healthy vs. Sick\n")
plt.show()



###Hold Time###
hold_mean_id = merged.groupby(["Parkinsons","ID"]).HoldTime.mean().reset_index()
plt.hist([hold_mean_id.HoldTime[hold_mean_id.Parkinsons==False], hold_mean_id.HoldTime[hold_mean_id.Parkinsons==True]],bins=20,histtype='bar', color=['lightgreen','#FF6666'], density=True)
plt.xlabel("Hold Time means")
plt.ylabel("Density")
plt.legend(["Healthy","Sick"])
plt.title("Hold Means - Healthy vs. Sick\n")
plt.show()

plt.subplot(2, 2, 2)

b_plt2=plt.boxplot([hold_mean_id.HoldTime[hold_mean_id.Parkinsons==False], hold_mean_id.HoldTime[hold_mean_id.Parkinsons==True]], labels=["Healthy","Sick"],patch_artist=True)
colors = ['lightgreen','#FF6666']
for patch, color in zip(b_plt2['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Hold Time means")
plt.ylabel("")
plt.title("Hold Means - Healthy vs. Sick\n")
plt.show()

###Flight Time###
flight_mean_id = merged.groupby(["Parkinsons","ID"]).FlightTime.mean().reset_index()
plt.hist([flight_mean_id.FlightTime[flight_mean_id.Parkinsons==False], flight_mean_id.FlightTime[flight_mean_id.Parkinsons==True]],bins=20,histtype='bar', color=['lightgreen','#FF6666'], density=True)
plt.xlabel("Flight Time means")
plt.ylabel("Density")
plt.legend(["Healthy","Sick"])
plt.title("Flight Means - Healthy vs. Sick\n")
plt.show()

plt.subplot(2, 2, 3)
b_plt3 = plt.boxplot([flight_mean_id.FlightTime[flight_mean_id.Parkinsons==False], flight_mean_id.FlightTime[flight_mean_id.Parkinsons==True]], labels=["Healthy","Sick"],patch_artist=True)
colors = ['lightgreen','#FF6666']
for patch, color in zip(b_plt3['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Flight Time means")
plt.ylabel("")
plt.title("Flight Means - Healthy vs. Sick\n")
plt.tight_layout()

plt.show()


###Latency vs. Hold+Flight###
merged['Hold_plus_Flight'] = merged["HoldTime"] + merged["FlightTime"]
merged_smaple=merged.sample(2000)
plt.scatter(merged_smaple.LatencyTime ,merged_smaple.Hold_plus_Flight, color="#3333FF")
plt.plot([0,800],[0,800], color="#00FFFF")
plt.xlabel("Latency Times")
plt.ylabel("")
plt.title("Latency vs. Hold+Flight\n")
plt.show()

####features - only mild, >2000###
#final = pd.read_csv(r"C:\\Users\\Nili\\PycharmProjects\\DSworkshop\\Data\\final.csv")
#
#final_filtered = final[(final['Impact'] == "Mild")]
#final_filtered = final_filtered[(final_filtered['total_count'] >= 2000)]
## final_filtered.to_csv(r"C:\\Users\\Nili\\PycharmProjects\\DSworkshop\\Data\\final_filtered.csv")

###health vs. sick - only mild, >2000###

###latency###
cleaned_Latency = merged[(merged.Parkinsons == False) | ((merged.Parkinsons == True) & (merged.Impact == "Mild"))].groupby(["Parkinsons","ID"]).LatencyTime.agg([np.mean,np.count_nonzero]).reset_index()
cleaned_Latency = cleaned_Latency[(cleaned_Latency.count_nonzero >= 2000)]
cleaned_Latency = cleaned_Latency.rename(columns = {"mean":"Latencymean"})

plt.subplot(2, 2, 1)
plt.hist([cleaned_Latency.Latencymean[cleaned_Latency.Parkinsons==False], cleaned_Latency.Latencymean[cleaned_Latency.Parkinsons==True]],bins=20,histtype='bar', color=['lightgreen','#FF6666'], density=True)
plt.xlabel("Latency means")
plt.ylabel("")
plt.legend(["Healthy","Sick"])
plt.title("Latency Means - Healthy vs. Sick\n")
plt.show()

b_plt4 = plt.boxplot([cleaned_Latency.Latencymean[cleaned_Latency.Parkinsons==False], cleaned_Latency.Latencymean[cleaned_Latency.Parkinsons==True]], labels=["Healthy","Sick"],patch_artist=True)
colors = ['lightgreen','#FF6666']
for patch, color in zip(b_plt4['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Latency Time means")
plt.ylabel("")
plt.title("Latency Means - Healthy vs. Sick\n")
plt.show()

###hold###
cleaned_Hold = merged[(merged.Parkinsons == False) | ((merged.Parkinsons == True) & (merged.Impact == "Mild"))].groupby(["Parkinsons","ID"]).LatencyTime.agg([np.mean,np.count_nonzero]).reset_index()
cleaned_Hold = cleaned_Hold[(cleaned_Hold.count_nonzero >= 2000)]
cleaned_Hold = cleaned_Hold.rename(columns = {"mean":"holdmean"})

plt.subplot(2, 2, 2)
plt.hist([cleaned_Hold.holdmean[cleaned_Hold.Parkinsons==False], cleaned_Hold.holdmean[cleaned_Hold.Parkinsons==True]],bins=20,histtype='bar', color=['lightgreen','#FF6666'], density=True)
plt.xlabel("Hold means")
plt.ylabel("")
plt.legend(["Healthy","Sick"])
plt.title("Hold Means - Healthy vs. Sick\n")
plt.show()

b_plt5=plt.boxplot([cleaned_Hold.holdmean[cleaned_Hold.Parkinsons==False], cleaned_Hold.holdmean[cleaned_Hold.Parkinsons==True]], labels=["Healthy","Sick"],patch_artist=True)
colors = ['lightgreen','#FF6666']
for patch, color in zip(b_plt5['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Hold Time means")
plt.ylabel("")
plt.title("Hold Means - Healthy vs. Sick\n")
plt.show()

###Flight###
cleaned_Flight = merged[(merged.Parkinsons == False) | ((merged.Parkinsons == True) & (merged.Impact == "Mild"))].groupby(["Parkinsons","ID"]).LatencyTime.agg([np.mean,np.count_nonzero]).reset_index()
cleaned_Flight = cleaned_Flight[(cleaned_Flight.count_nonzero >= 2000)]
cleaned_Flight = cleaned_Flight.rename(columns = {"mean":"Flightmean"})

plt.subplot(2, 2, 3)
plt.hist([cleaned_Flight.Flightmean[cleaned_Flight.Parkinsons==False], cleaned_Flight.Flightmean[cleaned_Flight.Parkinsons==True]],bins=20,histtype='bar', color=['lightgreen','#FF6666'], density=True)
plt.xlabel("Flight means")
plt.ylabel("")
plt.legend(["Healthy","Sick"])
plt.title("Flight Means - Healthy vs. Sick\n")
plt.show()

b_plt6 = plt.boxplot([cleaned_Flight.Flightmean[cleaned_Flight.Parkinsons==False], cleaned_Flight.Flightmean[cleaned_Flight.Parkinsons==True]], labels=["Healthy","Sick"],patch_artist=True)
colors = ['lightgreen','#FF6666']
for patch, color in zip(b_plt6['boxes'], colors):
    patch.set_facecolor(color)
plt.xlabel("Flight Time means")
plt.ylabel("")
plt.title("Flight Means - Healthy vs. Sick\n")
plt.tight_layout()
plt.show()

new = pd.read_csv(r"C:\\Users\\Nili\\PycharmProjects\\DSworkshop\\Data\\MIT-CSXPD_v2\\MIT-CS2PD\\data_MIT-CS2PD\\1424946827.1000_001_014.csv")


R = ['y','u','i','o','p','h','j','k','l',';','n','m',',','.','?','/','comma','period','colon']
L = ['q','w','e','r','t','a','s','d','f','g','z','x','c','v','b']
L_to_del = ['tab','escape','control_l','alt_l','shift_l','caps_lock']
R_to_del = ['underscore','semicolon','question','plus','apostrophe','right','num_lock','left','insert','end','down','delete','control_r','shift_r','return','minus','backspace','7','8','9','0']
S = ['space','6']



def dir_column(row):
    letter = row['Key'].lower()
    if letter in R:
        return 'R'
    if letter in L:
        return 'L'
    if letter in S:
        return 'S'
    if letter in L_to_del:
        return 'L_to_del'
    if letter in R_to_del:
        return 'R_to_del'
    return "bad val"

try:
    user_file_df['Hand'] = user_file_df.apply(lambda row: dir_column(row), axis=1)
except:
    pass
user_file_df['Direction'] = user_file_df['Hand'].shift(+1) + user_file_df['Hand']
'''
