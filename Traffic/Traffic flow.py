import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('smart_traffic.csv')
print(df.head())



# Convert 'timestamp' to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Resample data to daily and weekly traffic volume
daily_traffic = df['traffic_volume'].resample('D').sum()
weekly_traffic = df['traffic_volume'].resample('W').sum()

# Plot daily and weekly traffic patterns
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(daily_traffic, label='Daily Traffic Volume', color='blue')
plt.title('Daily Traffic Volume')
plt.ylabel('Volume')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(weekly_traffic, label='Weekly Traffic Volume', color='green')
plt.title('Weekly Traffic Volume')
plt.ylabel('Volume')
plt.grid(True)

plt.tight_layout()
plt.show()