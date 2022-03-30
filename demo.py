import torch
from roi_align import RoIAlign

feature_map = torch.rand(1, 3, 56, 56).cuda()
roi_align_part = RoIAlign(output_size=7, sampling_ratio=2)
input_ = torch.tensor([[0, 1.0, 1.0, 25.0, 25.0], [0, 24.0, 24.0, 55.0, 55.0]]).cuda()
results = roi_align_part(feature_map, input_)
print(results)