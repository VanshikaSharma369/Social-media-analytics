import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler,MinMaxScaler
df = pd.read_csv("C:/Users/hp/Downloads/Social_Media_Advertising.csv~/Social_Media_Advertising.csv")
df["Acquisition_Cost"] = df["Acquisition_Cost"].replace('[\$,]','',regex=True).astype(float)
df["Date"] = pd.to_datetime(df["Date"])
print(df.head(20))
print(df.describe())
print(df.isnull().sum())

def findoutliers(df,col):
    I1 = df[col].quantile(0.25)
    I3 = df[col].quantile(0.75)
    val = I3-I1
    l = I1 - 1.5*val
    u = I3 + 1.5*val
    ans = df[(df[col]<l) | (df[col]>u)]
    return ans
    

hd = ["Conversion_Rate","ROI","Clicks","Impressions","Engagement_Score"]

iqr = {col:findoutliers(df,col) for col in hd}
iqrcnt = {col: len(outlier)for col,outlier in iqr.items()}
print(iqrcnt)
plt.figure(figsize=(15, 10))
for i, col in enumerate(hd, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


sns.set_style("whitegrid")
fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(10,10))
fig.suptitle("Feature distribution",fontsize=18)
for i,col in enumerate(hd):
    row,col_pos = divmod(i,3)
    sns.histplot(df[col],bins=30,kde=True,ax=axes[row,col_pos])
    axes[row,col_pos].set_title(f"Distribution of{col}")
fig.delaxes(axes[1,2])
plt.tight_layout()
plt.show()

corr_matrix = df[hd].corr()
plt.figure(figsize=(11,11))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("Correlation of numerical features")
plt.show()

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
df["Conversion_rate_std"] = standard_scaler.fit_transform(df[["Conversion_Rate"]])
df["Roi_std"] = standard_scaler.fit_transform(df[["ROI"]])
df["click_norm"] = minmax_scaler.fit_transform(df[["Clicks"]])
df["impression_norm"] = minmax_scaler.fit_transform(df[["Impressions"]])
print(df[["Conversion_Rate","Conversion_rate_std","ROI","Roi_std","Clicks","click_norm","Impressions","impression_norm"]].head())
sns.pairplot(df[["Conversion_rate_std","Roi_std","click_norm","impression_norm"]],diag_kind="hist")
plt.suptitle("Pair plot between all features")
plt.show()


cat_cols = ["Campaign_Goal", "Channel_Used", "Location", "Language", "Customer_Segment"]
plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=df[col], y=df["Conversion_Rate"],hue=df["Channel_Used"],palette="coolwarm")
    plt.xticks(rotation=45)
    plt.title(f"Conversion Rate by {col}")

plt.tight_layout()
plt.show()

