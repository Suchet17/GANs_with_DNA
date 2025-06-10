import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.size']=13

df = pd.read_csv("GAN/FashionMNIST/logs/try5/output.log", sep='\t',
                 names = ['Timestamp', 'Epoch', 'Batch', 'LossD', 'LossG', 'nan'])
df = df.loc[range(1,len(df),6)][:35]
df['LossD'] = df['LossD'].apply(lambda i: float(i[8:]))
df['LossG'] = df['LossG'].apply(lambda i: float(i[8:]))

x = list(range(1,len(df)+1))
plt.plot(x, df['LossD'], 'r.-', label = "Discriminator Loss")
plt.plot(x, df['LossG'], 'b.-', label = "Generator Loss")

plt.plot(x, [np.log(2)]*len(x), '-k')
plt.title("Learning Curve for FashionMNIST", fontsize=16)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.xticks([0]+x[4::5], [0]+x[4::5], fontsize=11)
plt.legend()
plt.tight_layout()
# plt.ylim(0.4,1)
