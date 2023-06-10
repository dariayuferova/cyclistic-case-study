import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import seaborn as sns

## STEP 1. COLLECTING DATA

# Upload files:
trips1 = pd.read_csv('../data/202201-divvy-tripdata.csv')
trips2 = pd.read_csv('../data/202202-divvy-tripdata.csv')
trips3 = pd.read_csv('../data/202203-divvy-tripdata.csv')
trips4 = pd.read_csv('../data/202204-divvy-tripdata.csv')
trips5 = pd.read_csv('../data/202205-divvy-tripdata.csv')
trips6 = pd.read_csv('../data/202206-divvy-tripdata.csv')
trips7 = pd.read_csv('../data/202207-divvy-tripdata.csv')
trips8 = pd.read_csv('../data/202208-divvy-tripdata.csv')
trips9 = pd.read_csv('../data/202209-divvy-tripdata.csv')
trips10 = pd.read_csv('../data/202210-divvy-tripdata.csv')
trips11 = pd.read_csv('../data/202211-divvy-tripdata.csv')
trips12 = pd.read_csv('../data/202212-divvy-tripdata.csv')

# Check if column names are the same across all files:
print('Checking column names:',
      list(trips1.columns) == list(trips2.columns) == list(trips3.columns) ==
      list(trips4.columns) == list(trips5.columns) == list(trips6.columns) ==
      list(trips7.columns) == list(trips8.columns) == list(trips9.columns) ==
      list(trips10.columns) == list(trips11.columns) == list(trips12.columns))

# Check data types of columns:
print('Data types are:', trips1.dtypes, sep='\n')

# Concatenate all files into one (one under the other):
trips = pd.concat([trips1, trips2, trips3, trips4, trips5, trips6,
                   trips7, trips8, trips9, trips10, trips11, trips12], axis=0, ignore_index=True)

## STEP 2. CLEANING AND PREPARING DATA FOR ANALYSIS

# Delete unnecessary columns, rename column containing customer types, reset indices:
clean_trips = trips \
    .drop(columns=['start_station_id', 'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng']) \
    .rename(columns={'member_casual': 'customer_type'}) \
    .reset_index()

# Check if the values are appropriate and consistent with what we're expecting:
print('Types of customers are:', clean_trips['customer_type'].unique(), sep='\n')
print('Types of bikes are:', clean_trips['rideable_type'].unique(), sep='\n')
print('Min and max of ride IDs are:', clean_trips['ride_id'].min(), clean_trips['ride_id'].max(), sep='\n')
print('Min and max of starting date are:', clean_trips['started_at'].min(), clean_trips['started_at'].max(), sep='\n')
print('Min and max of ending date are:', clean_trips['ended_at'].min(), clean_trips['ended_at'].max(), sep='\n')

# Change the type of date columns to datetime:
clean_trips['started_at'] = pd.to_datetime(clean_trips['started_at'])
clean_trips['ended_at'] = pd.to_datetime(clean_trips['ended_at'])

# Extract time, day of week, and month of ride:
clean_trips['time_of_start'] = clean_trips['started_at'].dt.strftime('%H')
clean_trips['day_of_week_start'] = clean_trips['started_at'].dt.strftime('%a')
clean_trips['month_of_start'] = clean_trips['started_at'].dt.strftime('%b')

# Sort days of the week and months in order that we need:
cat_day_of_week = CategoricalDtype(['Mon', 'Tue', 'Wed',
                                    'Thu', 'Fri', 'Sat', 'Sun'], ordered=True)
cat_month = CategoricalDtype(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], ordered=True)

clean_trips['day_of_week_start'] = clean_trips['day_of_week_start'].astype(cat_day_of_week)
clean_trips['month_of_start'] = clean_trips['month_of_start'].astype(cat_month)

# Create new column ride_length, representing the duration of rides in minutes, and set it to integer:
clean_trips['ride_length'] = ((clean_trips['ended_at'] - clean_trips['started_at']).dt.total_seconds() / 60).astype(int)

# Check the quality of data in new column by examining min and max:
print('Min and max of ride length are:', clean_trips['ride_length'].min(), clean_trips['ride_length'].max(), sep='\n')

# Clean up negative values:
clean_trips = clean_trips.drop(clean_trips[clean_trips.ride_length < 0].index)
print('New min of ride length is:', clean_trips['ride_length'].min(), sep='\n')

## STEP 3. DATA EXPLORATION

# Let's look closer at the maximum value of new column ride_length:
print(clean_trips.loc[clean_trips['ride_length'].idxmax()])

# Notice that max ride length is several days, not hours - which is strange.
# After investigating further, a rideable type 'docked_bike' is found to be associated with the longest rides:
print(clean_trips.sort_values('ride_length', ascending=False) \
      .head(20)[['rideable_type', 'ride_length']])

# Let's see how ride length is distributed for different types of bikes.
# Theoretically, we expect to see a kind of exponentially decreasing ride length, within maybe one-day scope:
sns.histplot(data=clean_trips, x='ride_length', hue='rideable_type', bins=100)
plt.show()

# Unfortunately, due to the high number of rides concentrated close to zero, we can't really see any distribution.
# However, the limit of x-axis (40000 mins) tells us that there are some longer trips,
# so let's change the plot type to see it clearer:
sns.stripplot(data=clean_trips, y='rideable_type', x='ride_length',
              hue='rideable_type', alpha=0.5, sizes=[2])
plt.show()

# We can see that docked_bike rides take up to almost one month, which differentiates largely
# from what we see with other bike types.
# Let's plot ride length distribution without docked bikes:
without_docked = clean_trips[clean_trips.rideable_type != 'docked_bike']
sns.histplot(data=without_docked,
             x='ride_length', hue='rideable_type', bins=100)
plt.show()

# Now, that looks like more natural distribution.
# Let's double-check if ride duration statistics for docked bikes differs significantly
# - from all other types:
docked_stats = clean_trips \
    .groupby(~clean_trips['rideable_type'].str.contains('docked_bike')) \
    .ride_length.agg(['count', 'min', 'mean', 'max'])
docked_stats.index = ['docked_bike', 'non_docked_bike']

# - across all types:
diff_bikes_stats = clean_trips \
    .groupby('rideable_type') \
    .ride_length.agg(['count', 'min', 'mean', 'max'])

# - and from all dataset:
all_bikes_stats = pd.DataFrame({
    'count': clean_trips['ride_length'].count(),
    'min': clean_trips['ride_length'].min(),
    'mean': clean_trips['ride_length'].mean(),
    'max': clean_trips['ride_length'].max()}, index=['all_bike_types'])

print('Ride length statistics for different types of bikes:',
      docked_stats, diff_bikes_stats, all_bikes_stats, sep='\n')

# Despite having small number of rides (~3% of all rides), docked bikes rides do skew average
# and max statistics of ride length. Let's therefore remove those instances from our analysis,
# create another version of clean_trips dataframe without docked_bike:
clean_trips_v2 = without_docked

## STEP 4. DESCRIPTIVE ANALYSIS

# Let's first get the main statistics of rides for casual users and members:
rides_stats = clean_trips_v2.groupby('customer_type')\
    .ride_length.agg(['count', 'min', 'max', 'median', 'mean'])
print('Rides statistics:', rides_stats, sep='\n')

# Let's also check if different groups have strong preferences regarding type of bikes, electric or classic:
print(clean_trips_v2.groupby(['customer_type', 'rideable_type']).rideable_type.agg('count'))

# As part of our business task, we dive into the specifics of customers behaviour
# by looking into number and average duration of rides,
# - by time of the day:
time_df = clean_trips_v2.groupby(['customer_type', 'time_of_start']) \
    .ride_length.agg(['count', 'mean']) \
    .reset_index()
print('Rides by hour:', time_df, sep='\n')

# - by day of the week:
days_df = clean_trips_v2.groupby(['customer_type', 'day_of_week_start']) \
    .ride_length.agg(['count', 'mean']) \
    .reset_index()
print('Rides by day of the week:', days_df, sep='\n')

# - by month:
months_df = clean_trips_v2.groupby(['customer_type', 'month_of_start']) \
    .ride_length.agg(['count', 'mean']) \
    .reset_index()
print('Rides by month:', months_df, sep='\n')


## STEP 5. VISUALIZATIONS

# Let's visualize the findings. First, set a color pallete and style for all visualizations:
sns.set_palette(sns.color_palette('Set2'))
sns.set_style('whitegrid')

# Plot two subplots next to each other, to see a number of rides and average duration of rides by customer type
# - and time of the day:
fig1, ax = plt.subplots(1, 2)
sns.barplot(data=time_df, x='time_of_start', y='count', hue='customer_type', ax=ax[0]) \
    .set(title='Number of rides by customer type throughout the day',
         xlabel='Hours',
         ylabel='Number of rides')

sns.lineplot(data=time_df, x='time_of_start', y='mean', hue='customer_type', ax=ax[1]) \
    .set(title='Average duration of rides by customer type throughout the day',
         xlabel='Hours',
         ylabel='Average ride duration, in minutes',
         xlim=(0, 23),
         ylim=(10, 24))

# Rename legends:
ax[0].legend(title='Customer type')
ax[1].legend(title='Customer type')

# Then plot two subplots next to each other, to see a number of rides and average duration of rides by customer type
# - and day of the week:
fig2, ax = plt.subplots(1, 2)
sns.barplot(data=days_df, x='day_of_week_start', y='count', hue='customer_type', ax=ax[0]) \
    .set(title='Number of rides by customer type throughout the week',
         xlabel='Days of the week',
         ylabel='Number of rides')

sns.lineplot(data=days_df, x='day_of_week_start', y='mean', hue='customer_type', ax=ax[1]) \
    .set(title='Average duration of rides by customer type throughout the week',
         xlabel='Days of the week',
         ylabel='Average ride duration, in minutes',
         xlim=('Mon', 'Sun'),
         ylim=(11, 24))

# Rename legends:
ax[0].legend(title='Customer type')
ax[1].legend(title='Customer type')

# Finally, plot two subplots next to each other, to see a number of rides and average duration of rides by customer type
# - and month:
fig3, ax = plt.subplots(1, 2)
sns.barplot(data=months_df, x='month_of_start', y='count', hue='customer_type', ax=ax[0]) \
    .set(title='Number of rides by customer type throughout the year',
         xlabel='Months',
         ylabel='Number of rides')

sns.lineplot(data=months_df, x='month_of_start', y='mean', hue='customer_type', ax=ax[1]) \
    .set(title='Average duration of rides by customer type throughout the year',
         xlabel='Months',
         ylabel='Average ride duration, in minutes',
         xlim=('Jan', 'Dec'),
         ylim=(10, 25))

# Rename legends:
ax[0].legend(title='Customer type')
ax[1].legend(title='Customer type')

# Present all the visualizations:
plt.show()
