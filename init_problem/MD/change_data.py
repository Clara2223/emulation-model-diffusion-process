import numpy as np

#path='train_data/3D_chain_1e8steps_5e4burn_20261021_2133.npz'
path='train_data/3D_double_well_1e8steps_5e4burn_20261021_2056.npz'

with np.load(path, allow_pickle=True) as data:
    # Subsample every 10th element (retaining 10% of the data)
    positions = data['positions'][::10]
    md_time = data['md_time'][::10]
    velocities = data['velocities'][::10]
    energy = data['energy'][::10]
    
    # Safely handle metadata if they exist in the file
    new_stride = None
    if 'save_stride' in data.files:
        old_stride = int(data['save_stride'].item() if data['save_stride'].ndim == 0 else data['save_stride'][0])
        new_stride = old_stride * 10

    potential_type = data['potential_type'].item() if 'potential_type' in data.files and data['potential_type'].ndim == 0 else data.get('potential_type', None)
    well_amplitude = data['well_amplitude'].item() if 'well_amplitude' in data.files and data['well_amplitude'].ndim == 0 else data.get('well_amplitude', None)
    
np.savez_compressed(
    path,
    positions=positions,
    md_time=md_time,
    velocities=velocities,
    energy=energy,
    save_stride=new_stride,
    potential_type=potential_type,
    well_amplitude=well_amplitude
)