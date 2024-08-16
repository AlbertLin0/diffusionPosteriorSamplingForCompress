# test
section_counts = 10
section_counts = [section_counts]
start_idx = 0
all_steps = []
num_timesteps = 1000 
extra = num_timesteps % len(section_counts)
size_per = num_timesteps // len(section_counts)
for i, section_count in enumerate(section_counts):
    size = size_per + (1 if i < extra else 0)
    if size < section_count:
        raise ValueError(
            f"cannot divide section of {size} steps into {section_count}"
        )
    if section_count <= 1:
        frac_stride = 1
    else:
        frac_stride = (size - 1) / (section_count - 1)
    cur_idx = 0.0
    taken_steps = []
    for _ in range(section_count):
        taken_steps.append(start_idx + round(cur_idx))
        cur_idx += frac_stride
    all_steps += taken_steps
    start_idx += size
print(all_steps)