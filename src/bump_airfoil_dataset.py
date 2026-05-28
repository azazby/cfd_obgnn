import pandas as pd
import numpy as np
import os

class BumpComboAirfoilDataset:
    def __init__(self, name, data, extra_data, feature_names, bumped_dir,
                 split='train_A', split_csv='bumped_dataset_split.csv'):
        self.name = name
        self.data = data.copy() # dictionary
        self.extra_data = extra_data.copy() # dictionary with keys simulation_names, surface
        self.feature_names = feature_names
        self.split = split # string: trainA, trainB, trainC, trainD, val

        self._add_bumped_samples(split_csv, bumped_dir)


    def _add_bumped_samples(self, split_csv, bumped_dir):
        df = pd.read_csv(split_csv)
        if self.split == 'test':
            sim_names = list(df[df['split']=='test']['name'])
        elif self.split == 'test_cat':
            sim_names = list(df[df['split']=='test_cat']['name'])
        else:
            sim_names = list(df[df[self.split]]['name'])

        for name in sim_names:
            # read file
            bump_data = np.load(os.path.join(bumped_dir, (name+'.npz')), allow_pickle=True)
            size = bump_data['x-position'].shape[0]
            
            # update data
            for feature in self.feature_names:
                # handle naming mismatches
                if feature in ('x-velocity', 'y-velocity','pressure', 'turbulent_viscosity'):
                    bump_feature_arr = bump_data['gt_'+feature.replace('-','_')]
                else:   
                    bump_feature_arr = bump_data[feature]
                
                if feature in self.data:
                    self.data[feature] = np.concatenate((self.data[feature],bump_feature_arr))
                else:
                    self.data[feature] = np.array(bump_feature_arr)
            
            # update extra_data
            if 'simulation_names' in self.extra_data:
                self.extra_data['simulation_names'] = np.vstack((self.extra_data['simulation_names'], [name, str(size)]))
            else:
                self.extra_data['simulation_names'] = np.array([name, str(size)])
            if 'surface' in self.extra_data:
                self.extra_data['surface'] = np.concatenate((self.extra_data['surface'],bump_data['surface mask']))
            else:
                self.extra_data['surface'] = np.array(bump_data['surface mask'])