from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        min_per_day = self.data.min(dim=1).values
        max_per_day = self.data.max(dim=1).values
        return min_per_day, max_per_day

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        # Calculate the daily average temperatures
        # Calculate the day-over-day differences
        # Find the largest drop (most negative difference)
        daily_averages = self.data.mean(dim=1)
        day_over_day_diff = daily_averages[1:] - daily_averages[:-1]
        return day_over_day_diff.min()

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        # Calculate the daily average temperatures
        # Calculate the absolute differences from the daily averages
        # Find the indices of the maximum absolute differences for each day
        # Use these indices to extract the measurements that caused the largest deviations
        
        daily_averages = self.data.mean(dim=1, keepdim=True)
        abs_diff = torch.abs(self.data - daily_averages)
        max_diff_indices = abs_diff.argmax(dim=1)
        return self.data[torch.arange(self.data.size(0)), max_diff_indices]

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        # Select the last k days from the data
        # Find the maximum temperature for each of the last k days
        
        last_k_days = self.data[-k:]
        return last_k_days.max(dim=1).values

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        # Select the last k days from the data
        # Calculate the daily average temperatures for the last k days
        # Calculate the average temperature over the past k days
        
        last_k_days = self.data[-k:]
        daily_averages_last_k_days = last_k_days.mean(dim=1)
        return daily_averages_last_k_days.mean()

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        # Calculate the absolute difference between t and each day's measurements
        # Calculate the sum of absolute differences for each day
        # Find the index of the day with the smallest sum of differences
        differences = torch.abs(self.data - t)
        sum_of_differences = differences.sum(dim=1)
        return sum_of_differences.argmin()
