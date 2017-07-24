"""Small library that points to data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from dataset import Dataset


class NewDataset(Dataset):
  """data set"""

  def __init__(self, subset):
    super(NewDataset, self).__init__('NewDataset', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 2

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    # Bounding box data consists of 615299 bounding boxes for 544546 images.
    if self.subset == 'train':
      return 1625
    if self.subset == 'validation':
      return 466
