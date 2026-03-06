// EMD approximation module (based on auction algorithm)
// author: Minghua Liu
#include <torch/extension.h>
#include <vector>

int emd_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist, at::Tensor assignment, at::Tensor price,
					 at::Tensor assignment_inv, at::Tensor bid, at::Tensor bid_increments, at::Tensor max_increments,
					 at::Tensor unass_idx, at::Tensor unass_cnt, at::Tensor unass_cnt_sum, at::Tensor cnt_tmp, at::Tensor max_idx, float eps, int iters);

int emd_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx);

int emd_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist, at::Tensor assignment, at::Tensor price,
				at::Tensor assignment_inv, at::Tensor bid, at::Tensor bid_increments, at::Tensor max_increments,
				at::Tensor unass_idx, at::Tensor unass_cnt, at::Tensor unass_cnt_sum, at::Tensor cnt_tmp, at::Tensor max_idx, float eps, int iters)
{
	return emd_cuda_forward(xyz1, xyz2, dist, assignment, price, assignment_inv, bid, bid_increments, max_increments, unass_idx, unass_cnt, unass_cnt_sum, cnt_tmp, max_idx, eps, iters);
}

int emd_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz, at::Tensor graddist, at::Tensor idx)
{

	return emd_cuda_backward(xyz1, xyz2, gradxyz, graddist, idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("forward", &emd_forward, "emd forward (CUDA)");
	m.def("backward", &emd_backward, "emd backward (CUDA)");
}