# import numpy as np
# from abc import abstractmethod
# from .utils import data_loader, display_accuracy, Position


# CONFIG = {
#     "position": 'long',
#     "entry_ratio": 0.1,
#     "commission": 0.15,
#     "tradable_coins_path": "../../storage/dataset/dataset_10m/tradable_coins.txt"
# }


# class BasicBacktester:

#     def display_accuracy(self):
#         display_accuracy(
#             historical_predictions=self.historical_predictions,
#             historical_labels=self.historical_labels,
#         )

#     @abstractmethod
#     def run(self):
#         pass
