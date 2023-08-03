# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# %%
tips = sns.load_dataset("tips")
tips.head()

# %%
plt.Figure(figsize=(10, 10), dpi=1000)
ax = sns.barplot(x="day",
                 y="total_bill",
                 hue="sex",
                 data=tips)
plt.show()

# %%
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(10, 10), dpi=1000)
g = sns.catplot(x="sex",
                y="total_bill",
                hue="smoker",
                col="time",
                data=tips,
                kind="bar",
                height=4,
                aspect=.7)
plt.show()

# %%
df = pd.read_excel('./plot/metric.xlsx',sheet_name='Sheet2')
df.head()
# sns.set_context({'figure.figsize':[20, 20]})
plt.style.use('seaborn')

sns.set("paper", font_scale=1.2)
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.figure(figsize=(20, 10), dpi=1000)
plt.figure(figsize = [20,10],dpi=1000)
m = sns.catplot(x="Model",
                y="Value",
                hue="Metric",
                col="Time",
                data=df,
                kind="bar",
                height=4,
                aspect=1.2,
                )
plt.savefig('./plot/metric.svg', format='svg', dpi=1000)
plt.show()

