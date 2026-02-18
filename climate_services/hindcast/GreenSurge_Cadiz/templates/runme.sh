ulimit -s unlimited
dimr dimr_config.xml

source /opt/conda/bin/activate

case_dir=${1:-$(pwd)}
output_file="${case_dir}/dflowfmoutput/GreenSurge_GFDcase_map.nc"
output_file_raw="${case_dir}/dflowfmoutput/GreenSurge_GFDcase_map.raw"

python3 - <<EOF
import os, xarray as xr, numpy as np, struct

output_file = r"${output_file}"
output_file_raw = r"${output_file_raw}"

ds = xr.open_dataset(output_file)
data = ds["mesh2d_s1"].values.astype(np.float32)
shape = list(data.shape)
shape += [0] * (4 - len(shape))
header = struct.pack("4i", *shape) + bytes(256 - 16)

with open(output_file_raw, "wb") as f:
    f.write(header)
    f.write(data.tobytes())

# if ${SLURM_ARRAY_TASK_ID} != 1:
#     os.remove(output_file)
EOF